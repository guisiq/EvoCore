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
}
