using Evo.Core.Compiled;
using Evo.Core.Functions;
using Evo.Core.Values;
using Xunit;

namespace Evo.Core.Tests;

public sealed class CompiledDomainParityTests
{
    private static readonly GenomeShape Shape = new(InputCount: 2, NodeCount: 8, OutputCount: 2, MaxArity: 3, LevelsBack: 8);

    private static Genome<double[]> RandomDoubleArrayGenome(ulong seed, IFunctionSet<double[]> functions)
    {
        var rng = RandomStream.Create(seed, 0, 0, 0);
        var fg = new int[Shape.NodeCount];
        var cg = new int[Shape.NodeCount * Shape.MaxArity];
        var pg = new double[Shape.NodeCount][];
        for (var n = 0; n < Shape.NodeCount; n++)
        {
            fg[n] = rng.NextInt(0, functions.Count);
            for (var a = 0; a < Shape.MaxArity; a++)
            {
                cg[n * Shape.MaxArity + a] = rng.NextInt(0, Shape.InputCount + n);
            }

            pg[n] = [rng.NextDouble() * 4 - 2, rng.NextDouble() * 4 - 2, rng.NextDouble() * 4 - 2];
        }

        var og = new[] { rng.NextInt(0, Shape.InputCount + Shape.NodeCount), rng.NextInt(0, Shape.InputCount + Shape.NodeCount) };
        return Genome<double[]>.FromGenes(Shape, fg, cg, pg, og, DoubleArrayValueSemantics.Instance, functions);
    }

    private static Genome<int[]> RandomInt32ArrayGenome(ulong seed, IFunctionSet<int[]> functions)
    {
        var rng = RandomStream.Create(seed, 0, 0, 0);
        var fg = new int[Shape.NodeCount];
        var cg = new int[Shape.NodeCount * Shape.MaxArity];
        var pg = new int[Shape.NodeCount][];
        for (var n = 0; n < Shape.NodeCount; n++)
        {
            fg[n] = rng.NextInt(0, functions.Count);
            for (var a = 0; a < Shape.MaxArity; a++)
            {
                cg[n * Shape.MaxArity + a] = rng.NextInt(0, Shape.InputCount + n);
            }

            pg[n] = [rng.NextInt(-20, 20), rng.NextInt(-20, 20), rng.NextInt(-20, 20)];
        }

        var og = new[] { rng.NextInt(0, Shape.InputCount + Shape.NodeCount), rng.NextInt(0, Shape.InputCount + Shape.NodeCount) };
        return Genome<int[]>.FromGenes(Shape, fg, cg, pg, og, Int32ArrayValueSemantics.Instance, functions);
    }

    [Fact]
    public void CompiledDoubleArray_matches_reference_bitwise()
    {
        var functions = StandardDoubleArrayFunctionSet.Instance;
        var reference = new ReferenceGenomeEvaluator<double[]>(functions);
        var compiled = new CompiledGenomeEvaluator<double[]>(functions, DoubleArrayValueSemantics.Instance);
        var inputs = new[] { new[] { 0.5, -1.25, 3.0 }, new[] { -2.0, 0.0, 1e3 } };
        for (var seed = 0UL; seed < 30; seed++)
        {
            var genome = RandomDoubleArrayGenome(seed, functions);
            var expected = new double[Shape.OutputCount][];
            var actual = new double[Shape.OutputCount][];
            reference.Evaluate(genome, inputs, expected);
            compiled.Evaluate(genome, inputs, actual);
            for (var o = 0; o < expected.Length; o++)
            {
                Assert.Equal(expected[o].Length, actual[o].Length);
                for (var i = 0; i < expected[o].Length; i++)
                {
                    Assert.Equal(BitConverter.DoubleToInt64Bits(expected[o][i]), BitConverter.DoubleToInt64Bits(actual[o][i]));
                }
            }
        }
    }

    [Fact]
    public void CompiledInt32Array_matches_reference_exactly()
    {
        var functions = StandardInt32ArrayFunctionSet.Instance;
        var reference = new ReferenceGenomeEvaluator<int[]>(functions);
        var compiled = new CompiledGenomeEvaluator<int[]>(functions, Int32ArrayValueSemantics.Instance);
        var inputs = new[] { new[] { 5, -7, int.MinValue }, new[] { 0, 3, 9 } };
        for (var seed = 0UL; seed < 30; seed++)
        {
            var genome = RandomInt32ArrayGenome(seed, functions);
            var expected = new int[Shape.OutputCount][];
            var actual = new int[Shape.OutputCount][];
            reference.Evaluate(genome, inputs, expected);
            compiled.Evaluate(genome, inputs, actual);
            for (var o = 0; o < expected.Length; o++)
            {
                Assert.Equal(expected[o], actual[o]);
            }
        }
    }

    [Fact]
    public void SimdDouble_counts_kernel_blocks_and_tail_cases_per_active_node()
    {
        var functions = StandardDoubleFunctionSet.Instance;
        var counters = new CompiledExecutionCounters();
        var simd = new SimdGenomeEvaluator<double>(functions, DoubleValueSemantics.Instance, counters: counters);
        var width = System.Numerics.Vector<double>.Count;
        var caseCount = 2 * width + 3;
        var rng = RandomStream.Create(23, 1, 0, 0);
        var inputs = new double[caseCount * Shape.InputCount];
        for (var i = 0; i < inputs.Length; i++)
        {
            inputs[i] = rng.NextDouble() * 6 - 3;
        }

        var genome = Genome<double>.FromGenes(
            new GenomeShape(InputCount: 2, NodeCount: 3, OutputCount: 1, MaxArity: 2, LevelsBack: 3),
            new[] { 0, 1, 2 },
            new[] { 0, 1, 2, 2, 0, 1 },
            new[] { 0.0, 0.0, 0.0 },
            new[] { 4 },
            DoubleValueSemantics.Instance,
            functions);
        var output = new double[caseCount];
        counters.Reset();
        var report = simd.Evaluate(genome, inputs, caseCount, output);
        Assert.Equal(EvaluationBackend.Simd, report.Backend);
        Assert.Equal(2, report.VectorBlocks);
        Assert.Equal(3, report.TailCases);
        // Only node 2 (address 4) is active; nodes 0 and 1 must never execute.
        Assert.Equal(report.VectorBlocks, counters.KernelExecutions[2]);
        Assert.Equal(report.TailCases, counters.TailExecutions[2]);
        Assert.Equal(0, counters.KernelExecutions[0]);
        Assert.Equal(0, counters.KernelExecutions[1]);
        Assert.Equal(0, counters.TailExecutions[0]);
        Assert.Equal(0, counters.TailExecutions[1]);
    }

    [Fact]
    public void CompiledScalar_counts_each_active_node_once_per_call_and_never_reuses_values()
    {
        var functions = StandardDoubleFunctionSet.Instance;
        var counters = new CompiledExecutionCounters();
        var compiled = new CompiledGenomeEvaluator<double>(functions, DoubleValueSemantics.Instance, counters: counters);
        var genome = Genome<double>.FromGenes(
            new GenomeShape(InputCount: 1, NodeCount: 3, OutputCount: 1, MaxArity: 2, LevelsBack: 3),
            new[] { 0, 0, 0 },
            new[] { 0, 0, 1, 1, 0, 0 },
            new[] { 0.0, 0.0, 0.0 },
            new[] { 2 },
            DoubleValueSemantics.Instance,
            functions);
        var output = new double[1];
        counters.Reset();
        compiled.Evaluate(genome, new[] { 3.0 }, output);
        var firstScalar = (int[])counters.ScalarExecutions.Clone();
        compiled.Evaluate(genome, new[] { -5.0 }, output);
        Assert.Equal(firstScalar.Length, counters.ScalarExecutions.Length);
        for (var n = 0; n < firstScalar.Length; n++)
        {
            Assert.Equal(2 * firstScalar[n], counters.ScalarExecutions[n]);
        }
    }
}
