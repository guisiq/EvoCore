using Evo.Core;
using Evo.Core.Values;
using Xunit;

namespace Evo.Core.Tests;

// Corpus determinístico de genomas DAG válidos (EVO-012). O oráculo abaixo é independente de
// Genome<T>.Analyze: recalcula fecho ativo, profundidade e entradas usadas a partir dos genes brutos.
public sealed class DagGenomeCorpusTests
{
    private const int CorpusSize = 10_000;
    private const ulong CorpusSeed = 0xE0012UL;

    [Fact]
    public void Corpus_of_10000_valid_double_genomes_matches_independent_oracle()
    {
        var functions = DoubleFunctionSet.Instance;
        var stats = new CorpusStats();

        for (var index = 0; index < CorpusSize; index++)
        {
            var random = RandomStream.Create(CorpusSeed, 0, (ulong)index, 0);
            var shape = RandomShape(ref random);
            var fn = new int[shape.NodeCount];
            var conn = new int[shape.NodeCount * shape.MaxArity];
            var outs = new int[shape.OutputCount];

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
            }

            for (var output = 0; output < shape.OutputCount; output++)
            {
                outs[output] = random.NextInt(0, shape.InputCount + shape.NodeCount);
            }

            var genome = Genome<double>.FromGenes(
                shape, fn, conn, new double[shape.NodeCount], outs,
                DoubleValueSemantics.Instance, functions);

            var expected = Oracle(shape, fn, conn, outs, functions);
            var actual = genome.Analyze(functions);

            Assert.Equal(expected.ActiveNodes.Length, actual.ActiveNodeCount);
            Assert.Equal(expected.Depth, actual.ActiveDepth);
            Assert.Equal(expected.UsedInputs, actual.UsedInputs);
            Assert.Equal(expected.ActiveNodes, Sorted(actual.ActiveNodes));

            stats.Record(shape, expected);
        }

        Assert.Equal(CorpusSize, stats.Total);
        Assert.True(stats.WithSharedNode > 0, "corpus must include shared (multi-referenced) nodes");
        Assert.True(stats.WithReconvergence > 0, "corpus must include reconvergent nodes");
        Assert.True(stats.WithRepeatedOutputs > 0, "corpus must include repeated outputs");
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
    private static Expected Oracle(GenomeShape shape, int[] fn, int[] conn, int[] outs, IFunctionSet<double> functions)
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
            if (!active[node])
            {
                continue;
            }

            depth[node] = ComputeDepth(node, shape, fn, conn, functions, depth, active);
        }

        var activeNodes = new List<int>();
        var usedInputs = new SortedSet<int>();
        for (var node = 0; node < shape.NodeCount; node++)
        {
            if (active[node])
            {
                activeNodes.Add(node);
            }
        }

        foreach (var address in refCount.Select((count, address) => (count, address)).Where(p => p.count > 0 && p.address < inputs).Select(p => p.address))
        {
            usedInputs.Add(address);
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

    private static int ComputeDepth(int node, GenomeShape shape, int[] fn, int[] conn, IFunctionSet<double> functions, int[] depth, bool[] active)
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

    private static int[] Sorted(ReadOnlySpan<int> values)
    {
        var copy = values.ToArray();
        Array.Sort(copy);
        return copy;
    }

    private sealed class CorpusStats
    {
        public int Total { get; private set; }
        public int WithSharedNode { get; private set; }
        public int WithReconvergence { get; private set; }
        public int WithRepeatedOutputs { get; private set; }

        public void Record(GenomeShape shape, Expected expected)
        {
            Total++;
            if (expected.SharedNode)
            {
                WithSharedNode++;
            }

            if (expected.Reconvergence)
            {
                WithReconvergence++;
            }

            if (expected.RepeatedOutputs)
            {
                WithRepeatedOutputs++;
            }
        }
    }
}
