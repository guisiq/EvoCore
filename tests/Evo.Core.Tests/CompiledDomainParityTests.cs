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

    [Fact]
    public void CompiledDouble_shared_dag_counts_each_active_node_once_per_case()
    {
        AssertSharedDagCompiledCounts(StandardDoubleFunctionSet.Instance, DoubleValueSemantics.Instance, 0.0, new[] { 0.5 }, new[] { -3.0 });
    }

    [Fact]
    public void CompiledInt32_shared_dag_counts_each_active_node_once_per_case()
    {
        AssertSharedDagCompiledCounts(StandardInt32FunctionSet.Instance, Int32ValueSemantics.Instance, 0, new[] { 5 }, new[] { -7 });
    }

    [Fact]
    public void CompiledDoubleArray_shared_dag_counts_each_active_node_once_per_case()
    {
        AssertSharedDagCompiledCounts(
            StandardDoubleArrayFunctionSet.Instance,
            DoubleArrayValueSemantics.Instance,
            new[] { 1.0, 2.0 },
            new[] { new[] { 0.5, -1.0 } },
            new[] { new[] { -2.0, 3.0 } });
    }

    [Fact]
    public void CompiledInt32Array_shared_dag_counts_each_active_node_once_per_case()
    {
        AssertSharedDagCompiledCounts(
            StandardInt32ArrayFunctionSet.Instance,
            Int32ArrayValueSemantics.Instance,
            new[] { 1 },
            new[] { new[] { 5, -7 } },
            new[] { new[] { 0, 9 } });
    }

    [Fact]
    public void SimdDouble_shared_dag_counts_kernel_blocks_and_tails_for_all_case_shapes()
    {
        AssertSimdSharedDagCounts(StandardDoubleFunctionSet.Instance, DoubleValueSemantics.Instance, 0.0, i => (i % 7) * 0.5 - 1.0, System.Numerics.Vector<double>.Count);
    }

    [Fact]
    public void SimdInt32_shared_dag_counts_kernel_blocks_and_tails_for_all_case_shapes()
    {
        AssertSimdSharedDagCounts(StandardInt32FunctionSet.Instance, Int32ValueSemantics.Instance, 0, i => (i % 11) - 5, System.Numerics.Vector<int>.Count);
    }

    private static int IdOf<T>(IFunctionSet<T> functions, string name)
    {
        for (var i = 0; i < functions.Count; i++)
        {
            if (functions.Get(i).Id == name)
            {
                return i;
            }
        }

        throw new InvalidOperationException($"Função '{name}' ausente no conjunto '{functions.Id}'.");
    }

    // Nó 0 (endereço 1): Add(in0, in0). Nó 1 (endereço 2): Multiply(nó 0, nó 0), usado pelas saídas 0 e 2.
    // Nó 2 (endereço 3): Negate(in0) permanece inativo.
    private static void AssertSharedDagCompiledCounts<T>(IFunctionSet<T> functions, IValueSemantics<T> semantics, T parameter, T[] inputsFirst, T[] inputsSecond)
    {
        var shape = new GenomeShape(InputCount: 1, NodeCount: 3, OutputCount: 3, MaxArity: 2, LevelsBack: 3);
        var genome = Genome<T>.FromGenes(
            shape,
            new[] { IdOf(functions, "Add"), IdOf(functions, "Multiply"), IdOf(functions, "Negate") },
            new[] { 0, 0, 1, 1, 0, 0 },
            Enumerable.Repeat(parameter, 3).ToArray(),
            new[] { 2, 1, 2 },
            semantics,
            functions);
        var counters = new CompiledExecutionCounters();
        var compiled = new CompiledGenomeEvaluator<T>(functions, semantics, counters: counters);
        var reference = new ReferenceGenomeEvaluator<T>(functions);

        var first = new T[3];
        counters.Reset();
        compiled.Evaluate(genome, inputsFirst, first);
        Assert.Equal(1, counters.ScalarExecutions[0]);
        Assert.Equal(1, counters.ScalarExecutions[1]);
        Assert.Equal(0, counters.ScalarExecutions[2]);
        Assert.Equal(first[0], first[2]);
        var expectedFirst = new T[3];
        reference.Evaluate(genome, inputsFirst, expectedFirst);
        Assert.Equal(expectedFirst, first);

        // Segunda chamada com novos inputs: contagens acumulam exatamente 1 por chamada e saídas seguem os novos inputs.
        var second = new T[3];
        compiled.Evaluate(genome, inputsSecond, second);
        Assert.Equal(2, counters.ScalarExecutions[0]);
        Assert.Equal(2, counters.ScalarExecutions[1]);
        Assert.Equal(0, counters.ScalarExecutions[2]);
        var expectedSecond = new T[3];
        reference.Evaluate(genome, inputsSecond, expectedSecond);
        Assert.Equal(expectedSecond, second);
    }

    // Nó 0 (endereço 2): Add(in0, in1) compartilhado; nó 1 (endereço 3): Multiply(nó 0, nó 0);
    // nó 2 (endereço 4): Abs(in0) permanece inativo. Saídas: nó 1, nó 0, nó 1.
    private static void AssertSimdSharedDagCounts<T>(IFunctionSet<T> functions, IValueSemantics<T> semantics, T parameter, Func<int, T> inputAt, int width) where T : struct
    {
        var shape = new GenomeShape(InputCount: 2, NodeCount: 3, OutputCount: 3, MaxArity: 2, LevelsBack: 3);
        var genome = Genome<T>.FromGenes(
            shape,
            new[] { IdOf(functions, "Add"), IdOf(functions, "Multiply"), IdOf(functions, "Abs") },
            new[] { 0, 1, 2, 2, 0, 0 },
            Enumerable.Repeat(parameter, 3).ToArray(),
            new[] { 3, 2, 3 },
            semantics,
            functions);
        var reference = new ReferenceGenomeEvaluator<T>(functions);
        var counters = new CompiledExecutionCounters();
        var simd = new SimdGenomeEvaluator<T>(functions, semantics, counters: counters);

        foreach (var caseCount in new[] { width - 1, width, 2 * width, 2 * width + 3 })
        {
            var inputs = new T[caseCount * 2];
            for (var i = 0; i < inputs.Length; i++)
            {
                inputs[i] = inputAt(i);
            }

            var outputs = new T[caseCount * 3];
            counters.Reset();
            var report = simd.Evaluate(genome, inputs, caseCount, outputs);

            var expected = new T[caseCount * 3];
            for (var c = 0; c < caseCount; c++)
            {
                reference.Evaluate(genome, inputs.AsSpan(c * 2, 2), expected.AsSpan(c * 3, 3));
            }

            Assert.Equal(expected, outputs);
            Assert.Equal(caseCount, report.CaseCount);
            if (caseCount < width)
            {
                Assert.Equal(EvaluationBackend.CompiledScalar, report.Backend);
                Assert.Equal(0, report.VectorBlocks);
                Assert.Equal(caseCount, report.TailCases);
                Assert.Equal(caseCount, counters.ScalarExecutions[0]);
                Assert.Equal(caseCount, counters.ScalarExecutions[1]);
            }
            else
            {
                Assert.Equal(EvaluationBackend.Simd, report.Backend);
                Assert.Equal(caseCount / width, report.VectorBlocks);
                Assert.Equal(caseCount % width, report.TailCases);
                foreach (var node in new[] { 0, 1 })
                {
                    Assert.Equal(report.VectorBlocks, counters.KernelExecutions[node]);
                    Assert.Equal(report.TailCases, counters.TailExecutions[node]);
                }
            }

            Assert.Equal(0, counters.KernelExecutions[2]);
            Assert.Equal(0, counters.TailExecutions[2]);
        }
    }
}
