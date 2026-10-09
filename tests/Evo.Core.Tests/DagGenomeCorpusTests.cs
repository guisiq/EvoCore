using System.Text;
using Evo.Core;
using Evo.Core.Compiled;
using Evo.Core.Functions;
using Evo.Core.Values;
using Xunit;

namespace Evo.Core.Tests;

// Corpus determinístico de genomas DAG válidos (EVO-012/EVO-013). O oráculo é independente de
// Genome<T>.Analyze: recalcula arestas, referências, fecho ativo, profundidade e entradas usadas a partir
// dos genes brutos, e avalia as saídas por travessia própria. Cobre double, int, double[] e int[].
public sealed class DagGenomeCorpusTests
{
    private const int GenomesPerDomain = 2_500;
    private const int ArrayLength = 3;
    private const int SampleSize = 64;
    private const ulong CorpusSeed = 0xE0012UL;
    private const double SimdTolerance = 1e-12;

    private static readonly string[] Patterns =
    [
        "compartilhado", "reconvergencia", "saidas-repetidas",
        "entrada-nao-usada", "levelsback-minimo", "levelsback-maximo",
    ];

    private static readonly string[] DomainNames = ["double", "int", "double[]", "int[]"];

    private delegate T ParamDraw<T>(ref RandomStream random);

    private sealed class Dom<T>
    {
        public Dom(string name, int id, IFunctionSet<T> fs, IValueSemantics<T> vs, ParamDraw<T> draw, Func<T, T, bool> same)
        {
            Name = name;
            Id = id;
            Fs = fs;
            Vs = vs;
            Draw = draw;
            Same = same;
        }

        public string Name { get; }
        public int Id { get; }
        public IFunctionSet<T> Fs { get; }
        public IValueSemantics<T> Vs { get; }
        public ParamDraw<T> Draw { get; }
        public Func<T, T, bool> Same { get; }
    }

    private static Dom<double> DoubleDom() => new("double", 0, DoubleFunctionSet.Instance, DoubleValueSemantics.Instance, DrawDouble, SameDouble);
    private static Dom<int> IntDom() => new("int", 1, StandardInt32FunctionSet.Instance, Int32ValueSemantics.Instance, DrawInt, (a, b) => a == b);
    private static Dom<double[]> DoubleArrayDom() => new("double[]", 2, StandardDoubleArrayFunctionSet.Instance, DoubleArrayValueSemantics.Instance, DrawDoubleArray, SameDoubleArray);
    private static Dom<int[]> IntArrayDom() => new("int[]", 3, StandardInt32ArrayFunctionSet.Instance, Int32ArrayValueSemantics.Instance, DrawIntArray, (a, b) => a.SequenceEqual(b));

    private static double DrawDouble(ref RandomStream r) => r.NextInt(-5, 6) * 0.5;
    private static int DrawInt(ref RandomStream r) => r.NextInt(-5, 6);

    private static double[] DrawDoubleArray(ref RandomStream r)
    {
        var a = new double[ArrayLength];
        for (var i = 0; i < a.Length; i++)
        {
            a[i] = DrawDouble(ref r);
        }

        return a;
    }

    private static int[] DrawIntArray(ref RandomStream r)
    {
        var a = new int[ArrayLength];
        for (var i = 0; i < a.Length; i++)
        {
            a[i] = DrawInt(ref r);
        }

        return a;
    }

    [Fact]
    public void Corpus_of_10000_valid_genomes_matches_independent_oracle_on_all_domains()
    {
        var report = new Report();
        RunCorpus(DoubleDom(), report, GenomesPerDomain);
        RunConstructed(DoubleDom(), report);
        RunCorpus(IntDom(), report, GenomesPerDomain);
        RunConstructed(IntDom(), report);
        RunCorpus(DoubleArrayDom(), report, GenomesPerDomain);
        RunConstructed(DoubleArrayDom(), report);
        RunCorpus(IntArrayDom(), report, GenomesPerDomain);
        RunConstructed(IntArrayDom(), report);

        Assert.True(report.Total >= 10_000);
        foreach (var domain in DomainNames)
        {
            Assert.Equal(GenomesPerDomain + Patterns.Length, report.Get(domain, "genomes"));
            Assert.True(report.Get(domain, "shared") > 0, $"{domain}: corpus must include shared nodes");
            Assert.True(report.Get(domain, "reconv") > 0, $"{domain}: corpus must include reconvergent nodes");
            Assert.True(report.Get(domain, "repeated") > 0, $"{domain}: corpus must include repeated outputs");
            Assert.True(report.Get(domain, "unused") > 0, $"{domain}: corpus must include unused inputs");
            Assert.True(report.Get(domain, "lbmin") > 0, $"{domain}: corpus must include LevelsBack minimum");
            Assert.True(report.Get(domain, "lbmax") > 0, $"{domain}: corpus must include LevelsBack maximum");
            Assert.True(report.Get(domain, "lbmid") > 0, $"{domain}: corpus must include intermediate LevelsBack");
            foreach (var pattern in Patterns)
            {
                Assert.Equal(1, report.Get(domain, "pat:" + pattern));
            }
        }

        File.WriteAllText(Path.Combine(AppContext.BaseDirectory, "dag-corpus-coverage.md"), report.ToMarkdown());
    }

    [Fact]
    public void Corpus_generation_is_reproducible_from_seed()
    {
        var first = new Report();
        var second = new Report();
        RunCorpus(DoubleDom(), first, 200);
        RunCorpus(DoubleDom(), second, 200);
        Assert.Equal(first.Signature, second.Signature);
    }

    [Fact]
    public void Constructed_patterns_have_the_declared_property_on_every_domain()
    {
        RunConstructed(DoubleDom(), new Report());
        RunConstructed(IntDom(), new Report());
        RunConstructed(DoubleArrayDom(), new Report());
        RunConstructed(IntArrayDom(), new Report());
    }

    private static void RunCorpus<T>(Dom<T> d, Report report, int count)
    {
        for (var index = 0; index < count; index++)
        {
            var genome = BuildRandom(d, index);
            var oracle = Verify(d, report, genome, null);
            if (index < SampleSize)
            {
                SampleEvaluate(d, genome, oracle, (ulong)index);
                if (d.Fs is IFunctionSet<double>)
                {
                    SimdCheck((Dom<double>)(object)d, (Genome<double>)(object)genome, oracle, (ulong)index);
                }
            }
        }
    }

    private static void RunConstructed<T>(Dom<T> d, Report report)
    {
        for (var k = 0; k < Patterns.Length; k++)
        {
            var genome = BuildConstructed(d, k);
            var oracle = Verify(d, report, genome, Patterns[k]);
            var shape = genome.Shape;
            var holds = k switch
            {
                0 => oracle.Shared,
                1 => oracle.Reconvergence,
                2 => oracle.RepeatedOutputs,
                3 => oracle.UnusedInput,
                4 => shape.LevelsBack == 1,
                _ => shape.LevelsBack == shape.NodeCount,
            };
            Assert.True(holds, $"{d.Name}: pattern {Patterns[k]} does not exhibit its property");
            SampleEvaluate(d, genome, oracle, 5_000UL + (ulong)k);
            if (d.Fs is IFunctionSet<double>)
            {
                SimdCheck((Dom<double>)(object)d, (Genome<double>)(object)genome, oracle, 5_000UL + (ulong)k);
            }
        }
    }

    // Verifica um genoma contra o oráculo estrutural e as propriedades de fingerprint.
    private static Oracle Verify<T>(Dom<T> d, Report report, Genome<T> g, string? pattern)
    {
        var s = g.Shape;
        var fn = g.FunctionGenes.ToArray();
        var conn = g.ConnectionGenes.ToArray();
        var p = g.ParameterGenes.ToArray();
        var outs = g.OutputGenes.ToArray();
        var o = OracleAnalyze(s, fn, conn, outs, d.Fs);
        var actual = g.Analyze(d.Fs);

        Assert.Equal(0, o.EdgeViolations);
        Assert.Equal(o.Active.Length, actual.ActiveNodeCount);
        Assert.Equal(o.Depth, actual.ActiveDepth);
        Assert.Equal(o.UsedInputs, actual.UsedInputs);
        Assert.Equal(o.Active, actual.ActiveNodes);
        Assert.Equal(actual.ActiveNodes.Length, actual.ActiveNodes.Distinct().Count());

        var fp = g.ComputePhenotypeFingerprint(d.Fs, d.Vs);
        Assert.NotEmpty(fp);
        Assert.Equal(fp, g.ComputePhenotypeFingerprint(d.Fs, d.Vs));
        Assert.Equal(fp, Genome<T>.FromGenes(s, fn, conn, p, outs, d.Vs, d.Fs).ComputePhenotypeFingerprint(d.Fs, d.Vs));
        Assert.Equal(fp, g.Clone(d.Vs).ComputePhenotypeFingerprint(d.Fs, d.Vs));

        if (o.Active.Length > 0)
        {
            var mutated = (int[])fn.Clone();
            var node = o.Active[0];
            mutated[node] = OtherFunction(d.Fs, fn[node], s.MaxArity);
            var other = Genome<T>.FromGenes(s, mutated, conn, p, outs, d.Vs, d.Fs).ComputePhenotypeFingerprint(d.Fs, d.Vs);
            Assert.NotEqual(fp, other);
        }

        var inactive = Enumerable.Range(0, s.NodeCount).Where(n => !o.Active.Contains(n)).ToArray();
        if (inactive.Length > 0)
        {
            var mutated = (int[])fn.Clone();
            mutated[inactive[0]] = OtherFunction(d.Fs, fn[inactive[0]], s.MaxArity);
            var same = Genome<T>.FromGenes(s, mutated, conn, p, outs, d.Vs, d.Fs).ComputePhenotypeFingerprint(d.Fs, d.Vs);
            Assert.Equal(fp, same);
        }

        report.Add(d.Name, o, s, pattern);
        return o;
    }

    // Avalia a amostra com o avaliador de referência e o compilado; contagens por nó via ScalarExecutions.
    private static void SampleEvaluate<T>(Dom<T> d, Genome<T> g, Oracle o, ulong key)
    {
        var s = g.Shape;
        var r = RandomStream.Create(CorpusSeed ^ 0xABCDUL, (ulong)d.Id, key, 1);
        var inputs = new T[s.InputCount];
        for (var i = 0; i < inputs.Length; i++)
        {
            inputs[i] = d.Draw(ref r);
        }

        var execs = new int[s.NodeCount];
        var expected = OracleEvaluate(s, g.FunctionGenes.ToArray(), g.ConnectionGenes.ToArray(), g.ParameterGenes.ToArray(),
            g.OutputGenes.ToArray(), inputs, d.Fs, o.Active, execs);

        var outRef = new T[s.OutputCount];
        new ReferenceGenomeEvaluator<T>(d.Fs).Evaluate(g, inputs, outRef);
        var counters = new CompiledExecutionCounters();
        var outComp = new T[s.OutputCount];
        new CompiledGenomeEvaluator<T>(d.Fs, d.Vs, 16, counters).Evaluate(g, inputs, outComp);

        for (var output = 0; output < s.OutputCount; output++)
        {
            Assert.True(d.Same(expected[output], outRef[output]), $"{d.Name} key {key} output {output}: reference differs from oracle");
            Assert.True(d.Same(expected[output], outComp[output]), $"{d.Name} key {key} output {output}: compiled differs from oracle");
        }

        for (var node = 0; node < s.NodeCount; node++)
        {
            Assert.Equal(o.Active.Contains(node) ? 1 : 0, execs[node]);
            Assert.Equal(execs[node], counters.ScalarExecutions[node]);
        }
    }

    // Caminho SIMD (apenas double): um caso, contagens somadas dos três caminhos por nó.
    private static void SimdCheck(Dom<double> d, Genome<double> g, Oracle o, ulong key)
    {
        var s = g.Shape;
        var r = RandomStream.Create(CorpusSeed ^ 0xABCDUL, (ulong)d.Id, key, 1);
        var inputs = new double[s.InputCount];
        for (var i = 0; i < inputs.Length; i++)
        {
            inputs[i] = d.Draw(ref r);
        }

        var execs = new int[s.NodeCount];
        var expected = OracleEvaluate(s, g.FunctionGenes.ToArray(), g.ConnectionGenes.ToArray(), g.ParameterGenes.ToArray(),
            g.OutputGenes.ToArray(), inputs, d.Fs, o.Active, execs);

        var counters = new CompiledExecutionCounters();
        var outSimd = new double[s.OutputCount];
        new SimdGenomeEvaluator<double>(d.Fs, d.Vs, 16, counters).Evaluate(g, inputs, 1, outSimd);
        for (var output = 0; output < s.OutputCount; output++)
        {
            Assert.True(NearDouble(expected[output], outSimd[output]), $"double key {key} output {output}: SIMD differs from oracle");
        }

        for (var node = 0; node < s.NodeCount; node++)
        {
            var total = counters.ScalarExecutions[node] + counters.KernelExecutions[node] + counters.TailExecutions[node];
            Assert.Equal(execs[node], total);
        }
    }

    // Oráculo de avaliação: travessia própria em ordem topológica (índices crescentes de nó).
    private static T[] OracleEvaluate<T>(GenomeShape s, int[] fn, int[] conn, T[] p, int[] outs, T[] inputs,
        IFunctionSet<T> fs, int[] activeNodes, int[] execs)
    {
        var inputCount = s.InputCount;
        var values = new T[inputCount + s.NodeCount];
        Array.Copy(inputs, values, inputCount);
        foreach (var node in activeNodes)
        {
            var function = fs.Get(fn[node]);
            var args = new T[function.Arity];
            for (var slot = 0; slot < args.Length; slot++)
            {
                args[slot] = values[conn[node * s.MaxArity + slot]];
            }

            values[inputCount + node] = function.Invoke(args, p[node]);
            execs[node]++;
        }

        var result = new T[outs.Length];
        for (var output = 0; output < outs.Length; output++)
        {
            result[output] = values[outs[output]];
        }

        return result;
    }

    // Recalcula o grafo ativo por percurso próprio, sem usar Genome<T>.Analyze.
    private static Oracle OracleAnalyze<T>(GenomeShape s, int[] fn, int[] conn, int[] outs, IFunctionSet<T> fs)
    {
        var inputs = s.InputCount;
        var nodes = s.NodeCount;
        var active = new bool[nodes];
        var pending = new Stack<int>();
        foreach (var output in outs)
        {
            if (output >= inputs)
            {
                pending.Push(output - inputs);
            }
        }

        while (pending.Count > 0)
        {
            var node = pending.Pop();
            if (active[node])
            {
                continue;
            }

            active[node] = true;
            var arity = fs.Get(fn[node]).Arity;
            for (var slot = 0; slot < arity; slot++)
            {
                var address = conn[node * s.MaxArity + slot];
                if (address >= inputs)
                {
                    pending.Push(address - inputs);
                }
            }
        }

        var refs = new int[inputs + nodes];
        foreach (var output in outs)
        {
            refs[output]++;
        }

        var depth = new int[nodes];
        var violations = 0;
        var reconvergence = false;
        for (var node = 0; node < nodes; node++)
        {
            if (!active[node])
            {
                continue;
            }

            var arity = fs.Get(fn[node]).Arity;
            var best = 0;
            var nodeArgs = new HashSet<int>();
            for (var slot = 0; slot < arity; slot++)
            {
                var address = conn[node * s.MaxArity + slot];
                refs[address]++;
                if (address < inputs)
                {
                    continue;
                }

                var target = address - inputs;
                if (target >= node || node - target > s.LevelsBack)
                {
                    violations++;
                }

                best = Math.Max(best, depth[target]);
                nodeArgs.Add(target);
            }

            if (nodeArgs.Count >= 2)
            {
                reconvergence = true;
            }

            depth[node] = arity == 0 ? 0 : best + 1;
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
        for (var address = inputs; address < refs.Length; address++)
        {
            shared |= refs[address] > 1;
        }

        var usedInputs = Enumerable.Range(0, inputs).Where(i => refs[i] > 0).ToArray();
        var activeNodes = Enumerable.Range(0, nodes).Where(n => active[n]).ToArray();
        var repeated = outs.Distinct().Count() < outs.Length;
        return new Oracle(activeNodes, maxOutputDepth, usedInputs, shared, reconvergence, repeated,
            usedInputs.Length < inputs, violations);
    }

    private static Genome<T> BuildRandom<T>(Dom<T> d, int index)
    {
        var random = RandomStream.Create(CorpusSeed, (ulong)d.Id, (ulong)index, 0);
        var shape = RandomShape(ref random);
        var fn = new int[shape.NodeCount];
        var conn = new int[shape.NodeCount * shape.MaxArity];
        var outs = new int[shape.OutputCount];
        var parameters = new T[shape.NodeCount];

        for (var node = 0; node < shape.NodeCount; node++)
        {
            do
            {
                fn[node] = random.NextInt(0, d.Fs.Count);
            }
            while (d.Fs.Get(fn[node]).Arity > shape.MaxArity);

            for (var slot = 0; slot < shape.MaxArity; slot++)
            {
                conn[node * shape.MaxArity + slot] = RandomAddress(ref random, shape, node);
            }

            parameters[node] = d.Draw(ref random);
        }

        for (var output = 0; output < shape.OutputCount; output++)
        {
            outs[output] = random.NextInt(0, shape.InputCount + shape.NodeCount);
        }

        return Genome<T>.FromGenes(shape, fn, conn, parameters, outs, d.Vs, d.Fs);
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

    // Padrões construídos (aridade 2 em todos os nós). Conexões e saídas são endereços brutos.
    private static Genome<T> BuildConstructed<T>(Dom<T> d, int k) => k switch
    {
        0 => Make(d, 2, 3, 2, 3, [0, 1, 2, 1, 2, 0], [3, 4], k),
        1 => Make(d, 2, 4, 1, 4, [0, 1, 2, 0, 2, 1, 3, 4], [5], k),
        2 => Make(d, 2, 2, 3, 2, [0, 1, 2, 0], [3, 3, 2], k),
        3 => Make(d, 3, 2, 1, 2, [0, 1, 3, 1], [4], k),
        4 => Make(d, 2, 3, 1, 1, [0, 1, 2, 0, 3, 3], [4], k),
        _ => Make(d, 2, 3, 1, 3, [0, 1, 1, 0, 2, 3], [4], k),
    };

    private static Genome<T> Make<T>(Dom<T> d, int inputs, int nodes, int outputs, int levelsBack, int[] conn, int[] outs, int k)
    {
        var shape = new GenomeShape(inputs, nodes, outputs, 2, levelsBack);
        var fn = new int[nodes];
        Array.Fill(fn, Arity2(d.Fs));
        var r = RandomStream.Create(CorpusSeed, (ulong)d.Id, 1_000UL + (ulong)k, 2);
        var p = new T[nodes];
        for (var i = 0; i < nodes; i++)
        {
            p[i] = d.Draw(ref r);
        }

        return Genome<T>.FromGenes(shape, fn, conn, p, outs, d.Vs, d.Fs);
    }

    // Outra função de aridade <= maxArity, diferente de `current`, para mutar um gene de função sem violar o shape.
    private static int OtherFunction<T>(IFunctionSet<T> fs, int current, int maxArity)
    {
        for (var step = 1; step < fs.Count; step++)
        {
            var candidate = (current + step) % fs.Count;
            if (fs.Get(candidate).Arity <= maxArity)
            {
                return candidate;
            }
        }

        return current;
    }

    private static int Arity2<T>(IFunctionSet<T> fs)
    {
        for (var i = 0; i < fs.Count; i++)
        {
            if (fs.Get(i).Arity == 2)
            {
                return i;
            }
        }

        throw new InvalidOperationException("Function set has no arity-2 function.");
    }

    private static bool SameDouble(double a, double b) => double.IsFinite(a) && double.IsFinite(b)
        ? BitConverter.DoubleToInt64Bits(a) == BitConverter.DoubleToInt64Bits(b)
        : double.IsNaN(a) == double.IsNaN(b) && double.IsPositiveInfinity(a) == double.IsPositiveInfinity(b) && double.IsNegativeInfinity(a) == double.IsNegativeInfinity(b);

    private static bool SameDoubleArray(double[] a, double[] b) => a.Length == b.Length && a.Zip(b).All(pair => SameDouble(pair.First, pair.Second));

    // Tolerância SIMD: 1e-12 absoluto para |b| <= 1 e relativo acima disso; não finitos devem ter a mesma classe.
    private static bool NearDouble(double a, double b)
    {
        if (double.IsFinite(a) && double.IsFinite(b))
        {
            return Math.Abs(a - b) <= SimdTolerance * Math.Max(1.0, Math.Abs(b));
        }

        return SameDouble(a, b) || (double.IsNaN(a) && double.IsNaN(b));
    }

    private sealed record Oracle(
        int[] Active,
        int Depth,
        int[] UsedInputs,
        bool Shared,
        bool Reconvergence,
        bool RepeatedOutputs,
        bool UnusedInput,
        int EdgeViolations);

    private sealed class Report
    {
        private readonly Dictionary<string, int> _counts = new();
        private readonly Dictionary<string, HashSet<string>> _shapes = new();

        public int Total { get; private set; }
        public long Signature { get; private set; }

        public int Get(string domain, string metric) => _counts.GetValueOrDefault(domain + "|" + metric);

        public void Add(string domain, Oracle o, GenomeShape s, string? pattern)
        {
            Total++;
            Bump(domain, "genomes");
            if (o.Shared) Bump(domain, "shared");
            if (o.Reconvergence) Bump(domain, "reconv");
            if (o.RepeatedOutputs) Bump(domain, "repeated");
            if (o.UnusedInput) Bump(domain, "unused");
            if (s.LevelsBack == 1) Bump(domain, "lbmin");
            else if (s.LevelsBack == s.NodeCount) Bump(domain, "lbmax");
            else Bump(domain, "lbmid");
            if (pattern is not null) Bump(domain, "pat:" + pattern);

            if (!_shapes.TryGetValue(domain, out var set))
            {
                _shapes[domain] = set = new HashSet<string>();
            }

            set.Add($"{s.InputCount}x{s.NodeCount}x{s.OutputCount}/LB{s.LevelsBack}");
            Signature = unchecked(Signature * 31 + o.Active.Length * 1009 + o.Depth * 17 + o.UsedInputs.Length);
        }

        public string ToMarkdown()
        {
            var sb = new StringBuilder();
            sb.AppendLine("| domínio | genomas | formas×LB distintas | LB mín | LB intermediário | LB máx | compartilhado | reconvergência | saídas repetidas | entrada não usada |");
            sb.AppendLine("|---|---|---|---|---|---|---|---|---|---|");
            foreach (var d in DomainNames)
            {
                var shapes = _shapes.GetValueOrDefault(d)?.Count ?? 0;
                sb.AppendLine($"| {d} | {Get(d, "genomes")} | {shapes} | {Get(d, "lbmin")} | {Get(d, "lbmid")} | {Get(d, "lbmax")} | {Get(d, "shared")} | {Get(d, "reconv")} | {Get(d, "repeated")} | {Get(d, "unused")} |");
            }

            sb.AppendLine();
            sb.Append("| domínio |");
            foreach (var p in Patterns) sb.Append($" {p} |");
            sb.AppendLine();
            sb.Append("|---|");
            foreach (var _ in Patterns) sb.Append("---|");
            sb.AppendLine();
            foreach (var d in DomainNames)
            {
                sb.Append($"| {d} |");
                foreach (var p in Patterns) sb.Append($" {Get(d, "pat:" + p)} |");
                sb.AppendLine();
            }

            return sb.ToString();
        }

        private void Bump(string domain, string metric)
        {
            var key = domain + "|" + metric;
            _counts[key] = _counts.GetValueOrDefault(key) + 1;
        }
    }

    [Fact]
    public void Invalid_forward_connection_is_rejected_with_identified_address()
    {
        var shape = new GenomeShape(2, 2, 1, 2, 2);
        ThrowsWith("connection to node 1", () => Genome<double>.FromGenes(shape, new int[] { 0, 0 }, new int[] { 0, 3, 0, 1 }, new double[2], new int[] { 2 }, DoubleValueSemantics.Instance, DoubleFunctionSet.Instance));
    }

    [Fact]
    public void Invalid_connection_outside_levels_back_window_is_rejected()
    {
        var shape = new GenomeShape(2, 3, 1, 2, 1);
        ThrowsWith("2", () => Genome<double>.FromGenes(shape, new int[] { 0, 0, 0 }, new int[] { 0, 1, 2, 0, 2, 0 }, new double[3], new int[] { 4 }, DoubleValueSemantics.Instance, DoubleFunctionSet.Instance));
    }

    [Fact]
    public void Invalid_output_index_and_address_are_rejected()
    {
        var shape = new GenomeShape(2, 2, 1, 2, 2);
        var genome = new Genome<double>(shape);
        ThrowsWith("Output index 5", () => genome.SetOutputGene(5, 0));
        ThrowsWith("Output address 4", () => Genome<double>.FromGenes(shape, new int[2], new int[] { 0, 1, 0, 1 }, new double[2], new int[] { 4 }, DoubleValueSemantics.Instance, DoubleFunctionSet.Instance));
    }

    [Fact]
    public void Invalid_function_gene_is_rejected()
    {
        var shape = new GenomeShape(2, 2, 1, 2, 2);
        var genome = new Genome<double>(shape);
        ThrowsWith("must be >= 0", () => genome.SetFunctionGene(0, -1));
        var outOfSet = new int[] { DoubleFunctionSet.Instance.Count, 0 };
        ThrowsWith("function id", () => Genome<double>.FromGenes(shape, outOfSet, new int[] { 0, 1, 0, 1 }, new double[2], new int[] { 2 }, DoubleValueSemantics.Instance, DoubleFunctionSet.Instance));
    }

    [Fact]
    public void Invalid_gene_vector_lengths_are_rejected()
    {
        var shape = new GenomeShape(2, 2, 1, 2, 2);
        var values = DoubleValueSemantics.Instance;
        var functions = DoubleFunctionSet.Instance;
        ThrowsWith("FunctionGenes length", () => Genome<double>.FromGenes(shape, new int[3], new int[] { 0, 1, 0, 1 }, new double[2], new int[] { 2 }, values, functions));
        ThrowsWith("ConnectionGenes length", () => Genome<double>.FromGenes(shape, new int[2], new int[] { 0, 1, 0 }, new double[2], new int[] { 2 }, values, functions));
        ThrowsWith("ParameterGenes length", () => Genome<double>.FromGenes(shape, new int[2], new int[] { 0, 1, 0, 1 }, new double[3], new int[] { 2 }, values, functions));
        ThrowsWith("OutputGenes length", () => Genome<double>.FromGenes(shape, new int[2], new int[] { 0, 1, 0, 1 }, new double[2], new int[] { 2, 2 }, values, functions));
    }

    [Fact]
    public void Invalid_null_array_parameter_is_rejected()
    {
        var shape = new GenomeShape(2, 2, 1, 2, 2);
        var parameters = new double[][] { null!, null! };
        ThrowsWith("null parameter gene", () => Genome<double[]>.FromGenes(shape, new int[2], new int[] { 0, 1, 0, 1 }, parameters, new int[] { 2 }, DoubleArrayValueSemantics.Instance, StandardDoubleArrayFunctionSet.Instance));
    }

    [Fact]
    public void Invalid_levels_back_shape_is_rejected()
    {
        ThrowsWith("LevelsBack", () => new Genome<double>(new GenomeShape(2, 2, 1, 2, 0)));
    }

    [Fact]
    public void Inconsistent_array_parameter_shape_is_rejected()
    {
        var random = RandomStream.Create(1, 2, 3, 0);
        var doubles = new DoubleArrayParameterStrategy(ArrayLength);
        var ex = Assert.Throws<ArgumentException>(() => doubles.Mutate(new double[2], ref random));
        Assert.Equal("current", ex.ParamName);

        var ints = new IntArrayParameterStrategy(ArrayLength);
        var ex2 = Assert.Throws<ArgumentException>(() => ints.Mutate(new int[2], ref random));
        Assert.Equal("current", ex2.ParamName);
    }

    [Fact]
    public void Wrong_input_count_is_rejected_by_reference_evaluator()
    {
        var shape = new GenomeShape(2, 2, 1, 2, 2);
        var genome = new Genome<double>(shape);
        var ex = Assert.ThrowsAny<ArgumentException>(() => new ReferenceGenomeEvaluator<double>(DoubleFunctionSet.Instance).Evaluate(genome, new double[1], new double[1]));
        Assert.False(string.IsNullOrWhiteSpace(ex.Message));
    }

    private static void ThrowsWith(string fragment, Action action)
    {
        var ex = Assert.Throws<InvalidGenomeException>(action);
        Assert.Contains(fragment, ex.Message);
    }
}