using System.Numerics;
using Evo.Core.Compiled;
using Evo.Core.Functions;
using Evo.Core.Values;
using Xunit;

namespace Evo.Core.Tests;

/// <summary>
/// EVO-012: real per-address execution counts on the compiled and SIMD paths, reconvergence,
/// repeated arguments/outputs and isolation between calls. The genome is y = (x + x) + (x + x) = 4x,
/// with node 2 inactive, both outputs pointing at node 1 (repeated output).
/// </summary>
public sealed class DagExecutionAuditTests
{
    private const int NodeCount = 3;
    private const int InactiveNode = 2;

    private static Genome<T> BuildGenome<T>(IFunctionSet<T> functions, IValueSemantics<T> values, int constantIndex, T constant)
        => BuildGenome(functions, values, constantIndex, constant, constant);

    private static Genome<T> BuildGenome<T>(
        IFunctionSet<T> functions,
        IValueSemantics<T> values,
        int constantIndex,
        T constant,
        T addParameter)
    {
        var shape = new GenomeShape(InputCount: 1, NodeCount: NodeCount, OutputCount: 2, MaxArity: 2, LevelsBack: 3);
        // Node 0 = Add(in0, in0); node 1 = Add(n0, n0) (reconvergence and repeated argument); node 2 = Constant (inactive).
        return Genome<T>.FromGenes(
            shape,
            functionGenes: [0, 0, constantIndex],
            connectionGenes: [0, 0, 1, 1, 0, 0],
            parameterGenes: [addParameter, addParameter, constant],
            outputGenes: [2, 2],
            values,
            functions);
    }

    /// <summary>R1-1: the reference evaluator counts one execution per active node per call and zero for inactive nodes.</summary>
    private static void AssertReferenceCounts<T>(
        IFunctionSet<T> functions,
        IValueSemantics<T> values,
        int constantIndex,
        T constant,
        T[] cases,
        Func<T, T> oracle)
    {
        var genome = BuildGenome(functions, values, constantIndex, constant);
        var counters = new ReferenceExecutionCounters();
        var evaluator = new ReferenceGenomeEvaluator<T>(functions, counters);
        var outputs = new T[2];

        foreach (var x in cases)
        {
            evaluator.Evaluate(genome, new[] { x }, outputs);
            var expected = oracle(x);
            Assert.True(values.ValueEquals(outputs[0], expected));
            Assert.True(values.ValueEquals(outputs[1], expected));
        }

        Assert.Equal(new[] { cases.Length, cases.Length, 0 }, counters.NodeExecutions);
    }

    [Fact]
    public void Reference_counts_one_execution_per_active_node_per_case_for_all_domains()
    {
        var doubles = new[] { 0.5, -1.25, 3.0 };
        AssertReferenceCounts(StandardDoubleFunctionSet.Instance, DoubleValueSemantics.Instance, 13, 7.0, doubles, x => 4.0 * x);

        var ints = new[] { 0, 3, -7 };
        AssertReferenceCounts(StandardInt32FunctionSet.Instance, Int32ValueSemantics.Instance, 11, 7, ints, x => 4 * x);

        var arrays = new[] { new[] { 1.0, -2.0 }, new[] { 0.5, 0.25 } };
        AssertReferenceCounts(
            StandardDoubleArrayFunctionSet.Instance,
            DoubleArrayValueSemantics.Instance,
            13,
            new[] { 7.0 },
            arrays,
            x => x.Select(v => 4.0 * v).ToArray());

        var intArrays = new[] { new[] { 1, -2 }, new[] { 5, 0 } };
        AssertReferenceCounts(
            StandardInt32ArrayFunctionSet.Instance,
            Int32ArrayValueSemantics.Instance,
            11,
            new[] { 7 },
            intArrays,
            x => x.Select(v => 4 * v).ToArray());
    }

    [Fact]
    public void Reference_without_counters_keeps_counters_disabled_and_outputs_equal()
    {
        var genome = BuildGenome(StandardDoubleFunctionSet.Instance, DoubleValueSemantics.Instance, 13, 7.0);
        var outputs = new double[2];
        new ReferenceGenomeEvaluator<double>(StandardDoubleFunctionSet.Instance).Evaluate(genome, new[] { 2.0 }, outputs);
        Assert.Equal(8.0, outputs[0]);
        Assert.Equal(8.0, outputs[1]);
    }

    private static void AssertScalarCounts<T>(
        IFunctionSet<T> functions,
        IValueSemantics<T> values,
        int constantIndex,
        T constant,
        T[] cases,
        Func<T, T> oracle)
    {
        var genome = BuildGenome(functions, values, constantIndex, constant);
        var counters = new CompiledExecutionCounters();
        var evaluator = new CompiledGenomeEvaluator<T>(functions, values, 256, counters);
        var outputs = new T[2];

        foreach (var x in cases)
        {
            evaluator.Evaluate(genome, new[] { x }, outputs);
            var expected = oracle(x);
            Assert.True(values.ValueEquals(outputs[0], expected));
            Assert.True(values.ValueEquals(outputs[1], expected));
        }

        // Active node = 1 per case; inactive node = 0.
        Assert.Equal(new[] { cases.Length, cases.Length, 0 }, counters.ScalarExecutions);
        Assert.Equal(new int[NodeCount], counters.KernelExecutions);
        Assert.Equal(new int[NodeCount], counters.TailExecutions);
    }

    private static void AssertSimdCounts<T>(
        IFunctionSet<T> functions,
        IValueSemantics<T> values,
        int constantIndex,
        T constant,
        Func<int, T> caseValue,
        Func<T, T> oracle) where T : struct
    {
        var genome = BuildGenome(functions, values, constantIndex, constant);
        var width = Vector<T>.Count;
        var caseCount = 2 * width + 1;
        var inputs = new T[caseCount];
        for (var c = 0; c < caseCount; c++)
        {
            inputs[c] = caseValue(c);
        }

        var outputs = new T[caseCount * 2];
        var counters = new CompiledExecutionCounters();
        var simd = new SimdGenomeEvaluator<T>(functions, values, 256, counters);

        var report = simd.Evaluate(genome, inputs, caseCount, outputs);

        Assert.Equal(EvaluationBackend.Simd, report.Backend);
        Assert.Null(report.FallbackReason);
        Assert.Equal(2, report.VectorBlocks);
        Assert.Equal(1, report.TailCases);
        // Kernel and tail counts: one per active node per block / per tail case; inactive node = 0.
        Assert.Equal(new[] { 2, 2, 0 }, counters.KernelExecutions);
        Assert.Equal(new[] { 1, 1, 0 }, counters.TailExecutions);
        Assert.Equal(new int[NodeCount], counters.ScalarExecutions);

        for (var c = 0; c < caseCount; c++)
        {
            var expected = oracle(inputs[c]);
            Assert.True(values.ValueEquals(outputs[c * 2], expected));
            Assert.True(values.ValueEquals(outputs[c * 2 + 1], expected));
        }
    }

    [Fact]
    public void Double_scalar_counts_one_execution_per_active_node_per_case()
    {
        AssertScalarCounts(
            StandardDoubleFunctionSet.Instance,
            DoubleValueSemantics.Instance,
            constantIndex: 13,
            constant: 2.5,
            cases: [0.5, -1.25, 3.0],
            oracle: x => 4 * x);
    }

    [Fact]
    public void Int32_scalar_counts_one_execution_per_active_node_per_case()
    {
        AssertScalarCounts(
            StandardInt32FunctionSet.Instance,
            Int32ValueSemantics.Instance,
            constantIndex: 11,
            constant: 2,
            cases: [3, -5, 7],
            oracle: x => 4 * x);
    }

    [Fact]
    public void DoubleArray_scalar_counts_one_execution_per_active_node_per_case()
    {
        var cases = new[] { new[] { 1.0, 2.0, 3.0 }, new[] { -0.5, 0.0, 4.0 } };
        AssertScalarCounts(
            StandardDoubleArrayFunctionSet.Instance,
            DoubleArrayValueSemantics.Instance,
            constantIndex: 13,
            constant: new[] { 2.0, -1.0, 0.5 },
            cases: cases,
            oracle: x => x.Select(v => 4 * v).ToArray());
    }

    [Fact]
    public void Int32Array_scalar_counts_one_execution_per_active_node_per_case()
    {
        var cases = new[] { new[] { 1, 2, 3 }, new[] { -5, 0, 7 } };
        AssertScalarCounts(
            StandardInt32ArrayFunctionSet.Instance,
            Int32ArrayValueSemantics.Instance,
            constantIndex: 11,
            constant: new[] { 2, -1, 9 },
            cases: cases,
            oracle: x => x.Select(v => 4 * v).ToArray());
    }

    [Fact]
    public void Double_simd_counts_kernel_per_block_and_tail_per_case()
    {
        AssertSimdCounts(
            StandardDoubleFunctionSet.Instance,
            DoubleValueSemantics.Instance,
            constantIndex: 13,
            constant: 2.5,
            caseValue: c => c * 0.25 - 1.0,
            oracle: x => 4 * x);
    }

    [Fact]
    public void Int32_simd_counts_kernel_per_block_and_tail_per_case()
    {
        AssertSimdCounts(
            StandardInt32FunctionSet.Instance,
            Int32ValueSemantics.Instance,
            constantIndex: 11,
            constant: 2,
            caseValue: c => c - 3,
            oracle: x => 4 * x);
    }

    [Fact]
    public void DoubleArray_reused_plan_keeps_results_isolated_between_calls()
    {
        var functions = StandardDoubleArrayFunctionSet.Instance;
        var values = DoubleArrayValueSemantics.Instance;
        var add = new[] { 1.0, 2.0, 3.0 };
        var genomeA = BuildGenome(functions, values, 13, new[] { 1.0, 2.0, 3.0 }, add);
        var genomeB = BuildGenome(functions, values, 13, new[] { -1.0, 0.5, 8.0 }, add);
        var evaluator = new CompiledGenomeEvaluator<double[]>(functions, values);

        var inputA = new[] { new[] { 1.0, 2.0, 3.0 } };
        var outputsA = new double[2][];
        evaluator.Evaluate(genomeA, inputA, outputsA);
        var snapshotA = (double[])outputsA[0].Clone();

        var inputB = new[] { new[] { 5.0, 5.0, 5.0 } };
        var outputsB = new double[2][];
        evaluator.Evaluate(genomeB, inputB, outputsB);

        // Constant is inactive, so genome B reuses genome A's plan (same phenotype).
        Assert.Equal(1, evaluator.CacheHits);
        Assert.True(values.ValueEquals(outputsA[0], snapshotA));
        Assert.True(values.ValueEquals(outputsA[0], new[] { 4.0, 8.0, 12.0 }));
        Assert.True(values.ValueEquals(outputsB[0], new[] { 20.0, 20.0, 20.0 }));
        Assert.NotSame(outputsA[0], outputsB[0]);
        Assert.NotSame(outputsA[0], inputA[0]);

        // Mutating returned arrays must not affect later calls.
        outputsA[0][0] = 999.0;
        var outputsA2 = new double[2][];
        evaluator.Evaluate(genomeA, inputA, outputsA2);
        Assert.True(values.ValueEquals(outputsA2[0], snapshotA));
    }

    // R2-1..R2-5: matrix materialization, oracle-based outputs, reference/compiled/SIMD comparisons and isolation.
    private const int AddGene = 0;
    private const int NegateGene = 4;

    private sealed record Domain<T>(
        IFunctionSet<T> Functions,
        IValueSemantics<T> Values,
        int ConstantGene,
        Func<T, T, T> Add,
        Func<T, T> Negate,
        Func<int, int, T> Konst,
        Func<int, int, T> Input,
        Func<T, T, bool> Compiled,
        Func<T, T, bool>? Simd);

    private static readonly DagExpectation IsolationExpectation = new([0, 1], 2, [0], []);

    private static readonly DagFixture IsolationA = new(
        "isolation_add_add", 1, 2, 3, [2, 2],
        [new DagNode(DagFunctions.Add, 0, [0, 0]), new DagNode(DagFunctions.Add, 0, [1, 1]), new DagNode(DagFunctions.Constant, 0, [])],
        3, IsolationExpectation);

    private static readonly DagFixture IsolationB = new(
        "isolation_add_negate", 1, 2, 3, [2, 2],
        [new DagNode(DagFunctions.Add, 0, [0, 0]), new DagNode(DagFunctions.Negate, 0, [1]), new DagNode(DagFunctions.Constant, 0, [])],
        3, IsolationExpectation);

    private static Domain<double> DoubleDomain() => new(
        StandardDoubleFunctionSet.Instance, DoubleValueSemantics.Instance, 13,
        (a, b) => a + b, a => -a,
        (p, _) => p + 0.5, (c, _) => c * 0.75 - 1.25,
        SameDouble, DoubleWithinTolerance);

    private static Domain<int> IntDomain() => new(
        StandardInt32FunctionSet.Instance, Int32ValueSemantics.Instance, 11,
        (a, b) => a + b, a => -a,
        (p, _) => p, (c, _) => c * 4 - 5,
        (a, b) => a == b, (a, b) => a == b);

    private static Domain<double[]> DoubleArrayDomain() => new(
        StandardDoubleArrayFunctionSet.Instance, DoubleArrayValueSemantics.Instance, 13,
        (a, b) => a.Zip(b, (u, v) => u + v).ToArray(), a => a.Select(u => -u).ToArray(),
        (p, len) => Enumerable.Repeat(p + 0.5, len).ToArray(),
        (c, len) => Enumerable.Range(0, len).Select(i => c * 1.5 + i * 0.25 - 2.0).ToArray(),
        DoubleArraysSame, null);

    private static Domain<int[]> IntArrayDomain() => new(
        StandardInt32ArrayFunctionSet.Instance, Int32ArrayValueSemantics.Instance, 11,
        (a, b) => a.Zip(b, (u, v) => u + v).ToArray(), a => a.Select(u => -u).ToArray(),
        (p, len) => Enumerable.Repeat(p, len).ToArray(),
        (c, len) => Enumerable.Range(0, len).Select(i => c * 3 + i - 4).ToArray(),
        (a, b) => a.SequenceEqual(b), null);

    private static bool SameNonFiniteClass(double a, double b) =>
        double.IsNaN(a) == double.IsNaN(b)
        && double.IsPositiveInfinity(a) == double.IsPositiveInfinity(b)
        && double.IsNegativeInfinity(a) == double.IsNegativeInfinity(b);

    private static bool SameDouble(double a, double b) => double.IsFinite(a) && double.IsFinite(b)
        ? BitConverter.DoubleToInt64Bits(a) == BitConverter.DoubleToInt64Bits(b)
        : SameNonFiniteClass(a, b);

    private static bool DoubleWithinTolerance(double a, double b) => double.IsFinite(a) && double.IsFinite(b)
        ? Math.Abs(a - b) <= 1e-12 * Math.Max(1.0, Math.Abs(b))
        : SameNonFiniteClass(a, b);

    private static bool DoubleArraysSame(double[] a, double[] b) =>
        a.Length == b.Length && a.Zip(b).All(p => SameDouble(p.First, p.Second));

    private static int GeneFor(int descriptorFunction, int constantGene) => descriptorFunction switch
    {
        DagFunctions.Add => AddGene,
        DagFunctions.Negate => NegateGene,
        DagFunctions.Constant => constantGene,
        _ => throw new ArgumentOutOfRangeException(nameof(descriptorFunction)),
    };

    private static Genome<T> FromFixture<T>(DagFixture fx, IFunctionSet<T> functions, IValueSemantics<T> values, int constantGene, Func<int, T> konst)
    {
        var functionGenes = fx.Nodes.Select(n => GeneFor(n.Function, constantGene)).ToArray();
        var parameterGenes = fx.Nodes.Select(n => konst(n.Function == DagFunctions.Constant ? n.Parameter : 0)).ToArray();
        return Genome<T>.FromGenes(fx.Shape, functionGenes, fx.ConnectionGenes(), parameterGenes, fx.Outputs, values, functions);
    }

    /// <summary>Independent oracle over the descriptor (does not use Genome&lt;T&gt; or production evaluators).</summary>
    private static T[] Oracle<T>(DagFixture fx, T[] inputs, Func<T, T, T> add, Func<T, T> negate, Func<int, T> konst)
    {
        var v = new T[fx.InputCount + fx.NodeCount];
        Array.Copy(inputs, v, fx.InputCount);
        for (var n = 0; n < fx.NodeCount; n++)
        {
            var node = fx.Nodes[n];
            v[fx.InputCount + n] = node.Function switch
            {
                DagFunctions.Add => add(v[node.Args[0]], v[node.Args[1]]),
                DagFunctions.Negate => negate(v[node.Args[0]]),
                DagFunctions.Constant => konst(node.Parameter),
                _ => throw new ArgumentOutOfRangeException(nameof(fx)),
            };
        }

        return fx.Outputs.Select(a => v[a]).ToArray();
    }

    private static void AssertNodeCounts(int[] actual, DagFixture fx, int perActive, string what)
    {
        var expected = Enumerable.Range(0, fx.NodeCount)
            .Select(n => fx.Expected.ActiveNodes.Contains(n) ? perActive : 0)
            .ToArray();
        Assert.True(expected.SequenceEqual(actual), $"{fx.Name} {what}: esperado [{string.Join(",", expected)}], obtido [{string.Join(",", actual)}]");
    }

    private static void AuditScalar<T>(DagFixture fx, Domain<T> d)
    {
        Func<int, T> konst = p => d.Konst(p, fx.VectorLength);
        var genome = FromFixture(fx, d.Functions, d.Values, d.ConstantGene, konst);
        var cases = Enumerable.Range(0, 3).Select(c => d.Input(c, fx.VectorLength)).ToArray();
        var refCounters = new ReferenceExecutionCounters();
        var reference = new ReferenceGenomeEvaluator<T>(d.Functions, refCounters);
        var compiledCounters = new CompiledExecutionCounters();
        var compiled = new CompiledGenomeEvaluator<T>(d.Functions, d.Values, 256, compiledCounters);
        var refOut = new T[fx.OutputCount];
        var cmpOut = new T[fx.OutputCount];

        foreach (var x in cases)
        {
            var input = new[] { x };
            reference.Evaluate(genome, input, refOut);
            compiled.Evaluate(genome, input, cmpOut);
            var expected = Oracle(fx, input, d.Add, d.Negate, konst);
            for (var i = 0; i < fx.OutputCount; i++)
            {
                Assert.True(d.Values.ValueEquals(refOut[i], expected[i]), $"{fx.Name}: reference difere do oráculo");
                Assert.True(d.Values.ValueEquals(cmpOut[i], expected[i]), $"{fx.Name}: compiled difere do oráculo");
                Assert.True(d.Compiled(cmpOut[i], refOut[i]), $"{fx.Name}: compiled difere do reference");
            }
        }

        AssertNodeCounts(refCounters.NodeExecutions, fx, cases.Length, "reference");
        AssertNodeCounts(compiledCounters.ScalarExecutions, fx, cases.Length, "compiled escalar");
        Assert.True(new int[fx.NodeCount].SequenceEqual(compiledCounters.KernelExecutions), $"{fx.Name}: kernels escalares devem ser zero");
        Assert.True(new int[fx.NodeCount].SequenceEqual(compiledCounters.TailExecutions), $"{fx.Name}: caudas escalares devem ser zero");
    }

    private static void AuditSimd<T>(DagFixture fx, Domain<T> d) where T : struct
    {
        Func<int, T> konst = p => d.Konst(p, fx.VectorLength);
        var genome = FromFixture(fx, d.Functions, d.Values, d.ConstantGene, konst);
        var caseCount = 2 * Vector<T>.Count + 1;
        var inputs = Enumerable.Range(0, caseCount).Select(c => d.Input(c, fx.VectorLength)).ToArray();
        var outputs = new T[caseCount * fx.OutputCount];
        var counters = new CompiledExecutionCounters();
        var simd = new SimdGenomeEvaluator<T>(d.Functions, d.Values, 256, counters);

        var report = simd.Evaluate(genome, inputs, caseCount, outputs);

        Assert.Equal(EvaluationBackend.Simd, report.Backend);
        Assert.Null(report.FallbackReason);
        Assert.Equal(2, report.VectorBlocks);
        Assert.Equal(1, report.TailCases);
        AssertNodeCounts(counters.KernelExecutions, fx, 2, "kernel SIMD por bloco");
        AssertNodeCounts(counters.TailExecutions, fx, 1, "cauda SIMD por caso");
        Assert.True(new int[fx.NodeCount].SequenceEqual(counters.ScalarExecutions), $"{fx.Name}: escalar deve ser zero no SIMD");

        var reference = new ReferenceGenomeEvaluator<T>(d.Functions);
        var refOut = new T[fx.OutputCount];
        for (var c = 0; c < caseCount; c++)
        {
            var input = new[] { inputs[c] };
            reference.Evaluate(genome, input, refOut);
            var expected = Oracle(fx, input, d.Add, d.Negate, konst);
            for (var i = 0; i < fx.OutputCount; i++)
            {
                var value = outputs[c * fx.OutputCount + i];
                Assert.True(d.Simd!(value, expected[i]), $"{fx.Name}: SIMD difere do oráculo no caso {c}");
                Assert.True(d.Simd!(value, refOut[i]), $"{fx.Name}: SIMD difere do reference no caso {c}");
            }
        }
    }

    [Fact]
    public void Double_fixture_matrix_counts_per_address_on_reference_and_compiled()
    {
        foreach (var fx in DagFixtureMatrix.All)
        {
            AuditScalar(fx, DoubleDomain());
        }
    }

    [Fact]
    public void Int32_fixture_matrix_counts_per_address_on_reference_and_compiled()
    {
        foreach (var fx in DagFixtureMatrix.All)
        {
            AuditScalar(fx, IntDomain());
        }
    }

    [Fact]
    public void DoubleArray_fixture_matrix_counts_per_address_on_reference_and_compiled()
    {
        foreach (var fx in DagFixtureMatrix.All)
        {
            AuditScalar(fx, DoubleArrayDomain());
        }
    }

    [Fact]
    public void Int32Array_fixture_matrix_counts_per_address_on_reference_and_compiled()
    {
        foreach (var fx in DagFixtureMatrix.All)
        {
            AuditScalar(fx, IntArrayDomain());
        }
    }

    [Fact]
    public void Double_fixture_matrix_counts_kernel_and_tail_per_address_on_simd()
    {
        foreach (var fx in DagFixtureMatrix.All)
        {
            AuditSimd(fx, DoubleDomain());
        }
    }

    [Fact]
    public void Int32_fixture_matrix_counts_kernel_and_tail_per_address_on_simd()
    {
        foreach (var fx in DagFixtureMatrix.All)
        {
            AuditSimd(fx, IntDomain());
        }
    }

    private static void AssertScalarIsolation<T>(Domain<T> d, T x1, T x2, Action<T> mutateReturned)
    {
        Func<int, T> konst = p => d.Konst(p, IsolationA.VectorLength);
        var genomeA = FromFixture(IsolationA, d.Functions, d.Values, d.ConstantGene, konst);
        var genomeB = FromFixture(IsolationB, d.Functions, d.Values, d.ConstantGene, konst);
        var evaluator = new CompiledGenomeEvaluator<T>(d.Functions, d.Values);
        var inA1 = new[] { x1 };
        var inB = new[] { x2 };
        T[] OracleA(T[] input) => Oracle(IsolationA, input, d.Add, d.Negate, konst);
        T[] OracleB(T[] input) => Oracle(IsolationB, input, d.Add, d.Negate, konst);

        var outA1 = new T[2];
        evaluator.Evaluate(genomeA, inA1, outA1);
        var outB = new T[2];
        evaluator.Evaluate(genomeB, inB, outB);
        var outA2 = new T[2];
        evaluator.Evaluate(genomeA, inB, outA2);
        var outA3 = new T[2];
        evaluator.Evaluate(genomeA, inA1, outA3);

        for (var i = 0; i < 2; i++)
        {
            Assert.True(d.Values.ValueEquals(outA1[i], OracleA(inA1)[i]), "isolamento: saída A1 inicial");
            Assert.True(d.Values.ValueEquals(outB[i], OracleB(inB)[i]), "isolamento: saída B");
            Assert.True(d.Values.ValueEquals(outA2[i], OracleA(inB)[i]), "isolamento: saída A2");
            Assert.True(d.Values.ValueEquals(outA3[i], OracleA(inA1)[i]), "isolamento: saída A3");
        }

        // Mutating returned values must not leak into later calls.
        mutateReturned(outA1[0]);
        mutateReturned(outB[0]);
        var outA4 = new T[2];
        evaluator.Evaluate(genomeA, inA1, outA4);
        for (var i = 0; i < 2; i++)
        {
            Assert.True(d.Values.ValueEquals(outA2[i], OracleA(inB)[i]), "isolamento: valor anterior A2 alterado");
            Assert.True(d.Values.ValueEquals(outA3[i], OracleA(inA1)[i]), "isolamento: valor anterior A3 alterado");
            Assert.True(d.Values.ValueEquals(outA4[i], OracleA(inA1)[i]), "isolamento: nova chamada A4 incorreta");
        }
    }

    private static void AssertSimdIsolation<T>(Domain<T> d) where T : struct
    {
        Func<int, T> konst = p => d.Konst(p, IsolationA.VectorLength);
        var genomeA = FromFixture(IsolationA, d.Functions, d.Values, d.ConstantGene, konst);
        var genomeB = FromFixture(IsolationB, d.Functions, d.Values, d.ConstantGene, konst);
        var caseCount = 2 * Vector<T>.Count + 1;
        var in1 = Enumerable.Range(0, caseCount).Select(c => d.Input(c, IsolationA.VectorLength)).ToArray();
        var in2 = Enumerable.Range(0, caseCount).Select(c => d.Input(c + 7, IsolationA.VectorLength)).ToArray();
        var simd = new SimdGenomeEvaluator<T>(d.Functions, d.Values, 256, null);

        var out1 = new T[caseCount * 2];
        var report1 = simd.Evaluate(genomeA, in1, caseCount, out1);
        var snap1 = (T[])out1.Clone();
        var outB = new T[caseCount * 2];
        simd.Evaluate(genomeB, in2, caseCount, outB);
        var out2 = new T[caseCount * 2];
        var report2 = simd.Evaluate(genomeA, in2, caseCount, out2);

        Assert.Equal(EvaluationBackend.Simd, report1.Backend);
        Assert.Equal(EvaluationBackend.Simd, report2.Backend);
        for (var c = 0; c < caseCount; c++)
        {
            var e1 = Oracle(IsolationA, new[] { in1[c] }, d.Add, d.Negate, konst);
            var e2 = Oracle(IsolationA, new[] { in2[c] }, d.Add, d.Negate, konst);
            var eb = Oracle(IsolationB, new[] { in2[c] }, d.Add, d.Negate, konst);
            for (var i = 0; i < 2; i++)
            {
                Assert.True(d.Simd!(out1[c * 2 + i], e1[i]), $"isolamento SIMD: A1 caso {c}");
                Assert.True(d.Simd!(outB[c * 2 + i], eb[i]), $"isolamento SIMD: B caso {c}");
                Assert.True(d.Simd!(out2[c * 2 + i], e2[i]), $"isolamento SIMD: A2 caso {c}");
                Assert.True(d.Values.ValueEquals(out1[c * 2 + i], snap1[c * 2 + i]), $"isolamento SIMD: valor anterior alterado no caso {c}");
            }
        }
    }

    [Fact]
    public void Double_isolation_between_genomes_and_calls_keeps_previous_values()
    {
        AssertScalarIsolation(DoubleDomain(), 0.5, -2.0, _ => { });
    }

    [Fact]
    public void Int32_isolation_between_genomes_and_calls_keeps_previous_values()
    {
        AssertScalarIsolation(IntDomain(), 3, -6, _ => { });
    }

    [Fact]
    public void DoubleArray_isolation_between_genomes_and_calls_keeps_arrays_intact()
    {
        AssertScalarIsolation(DoubleArrayDomain(), new[] { 1.0, 2.0, 3.0 }, new[] { -4.0, 0.5, 8.0 }, a => a[0] += 1000.0);
    }

    [Fact]
    public void Int32Array_isolation_between_genomes_and_calls_keeps_arrays_intact()
    {
        AssertScalarIsolation(IntArrayDomain(), new[] { 1, 2, 3 }, new[] { -4, 5, 8 }, a => a[0] += 1000);
    }

    [Fact]
    public void Double_simd_isolation_between_genomes_and_calls_keeps_previous_values()
    {
        AssertSimdIsolation(DoubleDomain());
    }

    [Fact]
    public void Int32_simd_isolation_between_genomes_and_calls_keeps_previous_values()
    {
        AssertSimdIsolation(IntDomain());
    }
}
