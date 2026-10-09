using Evo.Core;
using Evo.Core.Compiled;
using Evo.Core.Functions;
using Evo.Core.Values;
using Xunit;

namespace Evo.Core.Tests;

// Corpus determinístico de genomas DAG válidos (EVO-012/EVO-013). O oráculo abaixo é independente de
// Genome<T>.Analyze: recalcula fecho ativo, profundidade e entradas usadas a partir dos genes brutos.
// Cobre double, int, double[] e int[]; a geração de cada genoma depende apenas de (seed, domínio, índice).
public sealed class DagGenomeCorpusTests
{
    private const int GenomesPerDomain = 2_500;
    private const int TotalGenomes = 4 * GenomesPerDomain;
    private const ulong CorpusSeed = 0xE0012UL;
    private const int ArrayLength = 3;
    private const int SampleSize = 64;

    [Fact]
    public void Corpus_of_10000_valid_genomes_matches_independent_oracle_on_all_domains()
    {
        var report = new CorpusReport();

        RunDomain(report, "double", 0, DoubleFunctionSet.Instance, DoubleValueSemantics.Instance,
            (r, _) => r.NextInt(-5, 6) * 0.5);
        RunDomain(report, "int", 1, StandardInt32FunctionSet.Instance, Int32ValueSemantics.Instance,
            (r, _) => r.NextInt(-5, 6));
        RunDomain(report, "double[]", 2, StandardDoubleArrayFunctionSet.Instance, DoubleArrayValueSemantics.Instance,
            (r, _) => Enumerable.Range(0, ArrayLength).Select(_ => r.NextInt(-5, 6) * 0.5).ToArray());
        RunDomain(report, "int[]", 3, StandardInt32ArrayFunctionSet.Instance, Int32ArrayValueSemantics.Instance,
            (r, _) => Enumerable.Range(0, ArrayLength).Select(_ => r.NextInt(-5, 6)).ToArray());

        Assert.Equal(TotalGenomes, report.Total);
        foreach (var domain in new[] { "double", "int", "double[]", "int[]" })
        {
            Assert.Equal(GenomesPerDomain, report.PerDomain[domain]);
            Assert.True(report.SharedNodeByDomain[domain] > 0, $"{domain}: corpus must include shared nodes");
            Assert.True(report.ReconvergenceByDomain[domain] > 0, $"{domain}: corpus must include reconvergent nodes");
            Assert.True(report.RepeatedOutputsByDomain[domain] > 0, $"{domain}: corpus must include repeated outputs");
        }

        Assert.True(report.ActiveNodeHistogram.Count > 1);
        Assert.True(report.DepthHistogram.Count > 1);
        Assert.True(report.LevelsBackHistogram.Count > 1);
    }

    [Fact]
    public void Corpus_generation_is_reproducible_from_seed()
    {
        var first = new CorpusReport();
        var second = new CorpusReport();
        RunDomain(first, "double", 0, DoubleFunctionSet.Instance, DoubleValueSemantics.Instance, (r, _) => r.NextInt(-5, 6) * 0.5, 200);
        RunDomain(second, "double", 0, DoubleFunctionSet.Instance, DoubleValueSemantics.Instance, (r, _) => r.NextInt(-5, 6) * 0.5, 200);
        Assert.Equal(first.Signature, second.Signature);
    }

    [Fact]
    public void Corpus_sample_matches_between_reference_and_compiled_evaluators()
    {
        SampleEquivalence(DoubleFunctionSet.Instance, DoubleValueSemantics.Instance, 0, (r, _) => r.NextInt(-5, 6) * 0.5, SameDouble);
        SampleEquivalence(StandardInt32FunctionSet.Instance, Int32ValueSemantics.Instance, 1, (r, _) => r.NextInt(-5, 6), (a, b) => a == b);
        SampleEquivalence(StandardDoubleArrayFunctionSet.Instance, DoubleArrayValueSemantics.Instance, 2,
            (r, _) => Enumerable.Range(0, ArrayLength).Select(_ => r.NextInt(-5, 6) * 0.5).ToArray(),
            (a, b) => a.Length == b.Length && a.Zip(b).All(p => SameDouble(p.First, p.Second)));
        SampleEquivalence(StandardInt32ArrayFunctionSet.Instance, Int32ArrayValueSemantics.Instance, 3,
            (r, _) => Enumerable.Range(0, ArrayLength).Select(_ => r.NextInt(-5, 6)).ToArray(),
            (a, b) => a.SequenceEqual(b));
    }

    private static void SampleEquivalence<T>(
        IFunctionSet<T> functions,
        IValueSemantics<T> values,
        int domainId,
        ParamFactory<T> param,
        Func<T, T, bool> same)
    {
        var reference = new ReferenceGenomeEvaluator<T>(functions);
        var compiled = new CompiledGenomeEvaluator<T>(functions, values, 16);
        var shapeInputs = 2;
        for (var index = 0; index < SampleSize; index++)
        {
            var genome = BuildGenome(functions, values, domainId, index, param, out var inputCount);
            var inputs = new T[inputCount];
            var random = RandomStream.Create(CorpusSeed ^ 0xABCDUL, (ulong)domainId, (ulong)index, 1);
            for (var i = 0; i < inputCount; i++)
            {
                inputs[i] = param(random, i);
            }

            var outRef = new T[genome.Shape.OutputCount];
            var outComp = new T[genome.Shape.OutputCount];
            reference.Evaluate(genome, inputs, outRef);
            compiled.Evaluate(genome, inputs, outComp);
            for (var o = 0; o < outRef.Length; o++)
            {
                Assert.True(same(outRef[o], outComp[o]), $"domain {domainId} genome {index} output {o} differs between reference and compiled");
            }
        }

        Assert.True(shapeInputs > 0);
    }

    private delegate T ParamFactory<T>(RandomStream random, int index);

    private static void RunDomain<T>(
        CorpusReport report,
        string domain,
        int domainId,
        IFunctionSet<T> functions,
        IValueSemantics<T> values,
        ParamFactory<T> param,
        int count = GenomesPerDomain)
    {
        for (var index = 0; index < count; index++)
        {
            var genome = BuildGenome(functions, values, domainId, index, param, out _);
            var shape = genome.Shape;
            var expected = Oracle(shape, genome.FunctionGenes.ToArray(), genome.ConnectionGenes.ToArray(), genome.OutputGenes.ToArray(), functions);
            var actual = genome.Analyze(functions);

            Assert.Equal(expected.ActiveNodes.Length, actual.ActiveNodeCount);
            Assert.Equal(expected.Depth, actual.ActiveDepth);
            Assert.Equal(expected.UsedInputs, actual.UsedInputs);
            Assert.Equal(expected.ActiveNodes, Sorted(actual.ActiveNodes));

            var fingerprint = genome.ComputePhenotypeFingerprint(functions, values);
            Assert.NotEmpty(fingerprint);

            report.Record(domain, shape, expected);
        }
    }

    private static Genome<T> BuildGenome<T>(
        IFunctionSet<T> functions,
        IValueSemantics<T> values,
        int domainId,
        int index,
        ParamFactory<T> param,
        out int inputCount)
    {
        var random = RandomStream.Create(CorpusSeed, (ulong)domainId, (ulong)index, 0);
        var shape = RandomShape(ref random);
        inputCount = shape.InputCount;
        var fn = new int[shape.NodeCount];
        var conn = new int[shape.NodeCount * shape.MaxArity];
        var outs = new int[shape.OutputCount];
        var parameters = new T[shape.NodeCount];

        for (var node = 0; node < shape.NodeCount; node++)
        {
            do
            {
                fn[node] = random.NextInt(0, functions.Count);
            }
            while (functions.Get(fn[node]).Arity > shape.MaxArity);
            for (var slot = 0; slot < shape.MaxArity; slot++)
            {
                conn[node * shape.MaxArity + slot] = RandomAddress(ref random, shape, node);
            }

            parameters[node] = param(random, node);
        }

        for (var output = 0; output < shape.OutputCount; output++)
        {
            outs[output] = random.NextInt(0, shape.InputCount + shape.NodeCount);
        }

        return Genome<T>.FromGenes(shape, fn, conn, parameters, outs, values, functions);
    }

    private static GenomeShape RandomShape(ref RandomStream random)
    {
        var inputs = random.NextInt(1, 4);
        var nodes = random.NextInt(1, 13);
        var outputs = random.NextInt(1, 4);
        var arity = random.NextInt(1, 5);
        var levelsBack = random.NextInt(1, nodes + 2);
        return new GenomeShape(inputs, nodes, outputs, arity, levelsBack);
    }

    // Endereço válido para a conexão de `node`: entrada ou nó anterior dentro de LevelsBack.
    private static int RandomAddress(ref RandomStream random, GenomeShape shape, int node)
    {
        var lowest = Math.Max(0, node - shape.LevelsBack);
        var candidates = shape.InputCount + (node - lowest);
        var pick = random.NextInt(0, candidates);
        return pick < shape.InputCount ? pick : shape.InputCount + lowest + (pick - shape.InputCount);
    }

    private sealed record Expected(int[] ActiveNodes, int Depth, int[] UsedInputs, bool SharedNode, bool Reconvergence, bool RepeatedOutputs);

    // Recalcula o grafo ativo por percurso próprio, sem usar Genome<T>.Analyze.
    private static Expected Oracle<T>(GenomeShape shape, int[] fn, int[] conn, int[] outs, IFunctionSet<T> functions)
    {
        var inputs = shape.InputCount;
        var active = new bool[shape.NodeCount];
        var refCount = new int[inputs + shape.NodeCount];
        var depth = new int[shape.NodeCount];

        var pending = new Stack<int>();
        foreach (var output in outs)
        {
            refCount[output]++;
            pending.Push(output);
        }

        while (pending.Count > 0)
        {
            var address = pending.Pop();
            if (address < inputs || active[address - inputs])
            {
                continue;
            }

            var node = address - inputs;
            active[node] = true;
            var arity = functions.Get(fn[node]).Arity;
            for (var slot = 0; slot < arity; slot++)
            {
                var arg = conn[node * shape.MaxArity + slot];
                refCount[arg]++;
                pending.Push(arg);
            }
        }

        // Profundidade em arestas: entradas têm 0; aridade zero tem 0; senão 1 + máximo dos argumentos.
        for (var node = 0; node < shape.NodeCount; node++)
        {
            if (active[node])
            {
                depth[node] = ComputeDepth(node, shape, fn, conn, functions, depth, active);
            }
        }

        var activeNodes = new List<int>();
        for (var node = 0; node < shape.NodeCount; node++)
        {
            if (active[node])
            {
                activeNodes.Add(node);
            }
        }

        var usedInputs = new SortedSet<int>();
        for (var address = 0; address < inputs; address++)
        {
            if (refCount[address] > 0)
            {
                usedInputs.Add(address);
            }
        }

        var maxOutputDepth = 0;
        foreach (var output in outs)
        {
            if (output >= inputs)
            {
                maxOutputDepth = Math.Max(maxOutputDepth, depth[output - inputs]);
            }
        }

        var shared = false;
        var reconvergence = false;
        for (var address = 0; address < refCount.Length; address++)
        {
            if (refCount[address] > 1)
            {
                shared = true;
                if (address >= inputs)
                {
                    reconvergence = true;
                }
            }
        }

        var repeated = outs.Distinct().Count() < outs.Length;
        return new Expected(activeNodes.ToArray(), maxOutputDepth, usedInputs.ToArray(), shared, reconvergence, repeated);
    }

    private static int ComputeDepth<T>(int node, GenomeShape shape, int[] fn, int[] conn, IFunctionSet<T> functions, int[] depth, bool[] active)
    {
        var arity = functions.Get(fn[node]).Arity;
        if (arity == 0)
        {
            return 0;
        }

        var best = 0;
        for (var slot = 0; slot < arity; slot++)
        {
            var arg = conn[node * shape.MaxArity + slot];
            if (arg >= shape.InputCount)
            {
                var target = arg - shape.InputCount;
                best = Math.Max(best, active[target] ? depth[target] : ComputeDepth(target, shape, fn, conn, functions, depth, active));
            }
        }

        return best + 1;
    }

    private static bool SameDouble(double a, double b) => double.IsFinite(a) && double.IsFinite(b)
        ? BitConverter.DoubleToInt64Bits(a) == BitConverter.DoubleToInt64Bits(b)
        : double.IsNaN(a) == double.IsNaN(b) && double.IsPositiveInfinity(a) == double.IsPositiveInfinity(b) && double.IsNegativeInfinity(a) == double.IsNegativeInfinity(b);

    private static int[] Sorted(ReadOnlySpan<int> values)
    {
        var copy = values.ToArray();
        Array.Sort(copy);
        return copy;
    }

    private sealed class CorpusReport
    {
        public int Total { get; private set; }
        public Dictionary<string, int> PerDomain { get; } = new();
        public Dictionary<string, int> SharedNodeByDomain { get; } = new();
        public Dictionary<string, int> ReconvergenceByDomain { get; } = new();
        public Dictionary<string, int> RepeatedOutputsByDomain { get; } = new();
        public Dictionary<int, int> ActiveNodeHistogram { get; } = new();
        public Dictionary<int, int> DepthHistogram { get; } = new();
        public Dictionary<int, int> LevelsBackHistogram { get; } = new();
        public long Signature { get; private set; }

        public void Record(string domain, GenomeShape shape, Expected expected)
        {
            Total++;
            Bump(PerDomain, domain);
            if (expected.SharedNode) Bump(SharedNodeByDomain, domain);
            if (expected.Reconvergence) Bump(ReconvergenceByDomain, domain);
            if (expected.RepeatedOutputs) Bump(RepeatedOutputsByDomain, domain);
            Bump(ActiveNodeHistogram, expected.ActiveNodes.Length);
            Bump(DepthHistogram, expected.Depth);
            Bump(LevelsBackHistogram, shape.LevelsBack);
            Signature = unchecked(Signature * 31 + expected.ActiveNodes.Length * 1009 + expected.Depth * 17 + expected.UsedInputs.Length);
        }

        private static void Bump<K>(Dictionary<K, int> map, K key) where K : notnull =>
            map[key] = map.GetValueOrDefault(key) + 1;
    }
}

