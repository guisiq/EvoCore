using Evo.Core;
using Evo.Core.Functions;
using Evo.Core.Values;
using Evo.Ilgpu;
using Xunit;

namespace Evo.Ilgpu.Tests;

public sealed class IlgpuArrayParityTests
{
    private static readonly GenomeShape RandomShape = new(InputCount: 2, NodeCount: 8, OutputCount: 2, MaxArity: 3, LevelsBack: 8);

    private static Genome<double[]> RandomDoubleArray(ulong seed, IFunctionSet<double[]> functions)
    {
        var rng = RandomStream.Create(seed, 0, 0, 0);
        var fg = new int[RandomShape.NodeCount];
        var cg = new int[RandomShape.NodeCount * RandomShape.MaxArity];
        var pg = new double[RandomShape.NodeCount][];
        for (var n = 0; n < RandomShape.NodeCount; n++)
        {
            fg[n] = rng.NextInt(0, functions.Count);
            for (var a = 0; a < RandomShape.MaxArity; a++)
            {
                cg[n * RandomShape.MaxArity + a] = rng.NextInt(0, RandomShape.InputCount + n);
            }

            pg[n] = [rng.NextDouble() * 4 - 2, rng.NextDouble() * 4 - 2, rng.NextDouble() * 4 - 2];
        }

        var og = new[]
        {
            rng.NextInt(0, RandomShape.InputCount + RandomShape.NodeCount),
            rng.NextInt(0, RandomShape.InputCount + RandomShape.NodeCount),
        };
        return Genome<double[]>.FromGenes(RandomShape, fg, cg, pg, og, DoubleArrayValueSemantics.Instance, functions);
    }

    private static Genome<int[]> RandomInt32Array(ulong seed, IFunctionSet<int[]> functions)
    {
        var rng = RandomStream.Create(seed, 0, 0, 0);
        var fg = new int[RandomShape.NodeCount];
        var cg = new int[RandomShape.NodeCount * RandomShape.MaxArity];
        var pg = new int[RandomShape.NodeCount][];
        for (var n = 0; n < RandomShape.NodeCount; n++)
        {
            fg[n] = rng.NextInt(0, functions.Count);
            for (var a = 0; a < RandomShape.MaxArity; a++)
            {
                cg[n * RandomShape.MaxArity + a] = rng.NextInt(0, RandomShape.InputCount + n);
            }

            pg[n] = [rng.NextInt(-20, 20), rng.NextInt(-20, 20), rng.NextInt(-20, 20)];
        }

        var og = new[]
        {
            rng.NextInt(0, RandomShape.InputCount + RandomShape.NodeCount),
            rng.NextInt(0, RandomShape.InputCount + RandomShape.NodeCount),
        };
        return Genome<int[]>.FromGenes(RandomShape, fg, cg, pg, og, Int32ArrayValueSemantics.Instance, functions);
    }

    private static Genome<double[]> DoubleArrayScenarioGenome(IFunctionSet<double[]> functions)
    {
        var shape = new GenomeShape(InputCount: 2, NodeCount: 8, OutputCount: 3, MaxArity: 3, LevelsBack: 8);
        var constantId = IlgpuTestHelpers.FindFunctionId(functions, "const");
        var sqrtId = IlgpuTestHelpers.FindFunctionId(functions, "sqrt");
        var logId = IlgpuTestHelpers.FindFunctionId(functions, "log");
        var addId = IlgpuTestHelpers.FindFunctionId(functions, "add");
        var divideId = IlgpuTestHelpers.FindFunctionId(functions, "divide");
        var greaterId = IlgpuTestHelpers.FindFunctionId(functions, "greater");
        var selectId = IlgpuTestHelpers.FindFunctionId(functions, "select");

        return Genome<double[]>.FromGenes(
            shape,
            [constantId, sqrtId, logId, addId, divideId, greaterId, selectId, addId],
            [
                0, 0, 0,
                2, 0, 0,
                2, 0, 0,
                3, 4, 0,
                0, 1, 0,
                0, 1, 0,
                7, 5, 6,
                3, 3, 0,
            ],
            [
                [-4.0, -4.0, -4.0],
                Array.Empty<double>(),
                Array.Empty<double>(),
                Array.Empty<double>(),
                Array.Empty<double>(),
                Array.Empty<double>(),
                Array.Empty<double>(),
                Array.Empty<double>(),
            ],
            [8, 5, 8],
            DoubleArrayValueSemantics.Instance,
            functions);
    }

    private static Genome<int[]> IntArrayScenarioGenome(IFunctionSet<int[]> functions)
    {
        var shape = new GenomeShape(InputCount: 2, NodeCount: 8, OutputCount: 3, MaxArity: 3, LevelsBack: 8);
        var constantId = IlgpuTestHelpers.FindFunctionId(functions, "const");
        var absId = IlgpuTestHelpers.FindFunctionId(functions, "abs");
        var divideId = IlgpuTestHelpers.FindFunctionId(functions, "divide");
        var addId = IlgpuTestHelpers.FindFunctionId(functions, "add");
        var greaterId = IlgpuTestHelpers.FindFunctionId(functions, "greater");
        var selectId = IlgpuTestHelpers.FindFunctionId(functions, "select");

        return Genome<int[]>.FromGenes(
            shape,
            [constantId, absId, constantId, divideId, addId, greaterId, selectId, addId],
            [
                0, 0, 0,
                2, 0, 0,
                0, 0, 0,
                2, 4, 0,
                3, 5, 0,
                0, 1, 0,
                7, 6, 3,
                3, 3, 0,
            ],
            [
                [int.MinValue, int.MinValue, int.MinValue],
                Array.Empty<int>(),
                [-1, -1, -1],
                Array.Empty<int>(),
                Array.Empty<int>(),
                Array.Empty<int>(),
                Array.Empty<int>(),
                Array.Empty<int>(),
            ],
            [8, 6, 8],
            Int32ArrayValueSemantics.Instance,
            functions);
    }

    [Fact]
    public void Ilgpu_double_array_matches_reference_elementwise_within_1e12()
    {
        var functions = StandardDoubleArrayFunctionSet.Instance;
        var reference = new ReferenceGenomeEvaluator<double[]>(functions);
        using var ilgpu = new IlgpuGenomeEvaluator<double[]>(functions);
        var inputs = new[] { new[] { 0.5, -1.25, 3.0 }, new[] { -2.0, 0.0, 1e3 } };
        for (var seed = 0UL; seed < 10; seed++)
        {
            var genome = RandomDoubleArray(seed, functions);
            var expected = new double[RandomShape.OutputCount][];
            var actual = new double[RandomShape.OutputCount][];
            reference.Evaluate(genome, inputs, expected);
            ilgpu.Evaluate(genome, inputs, actual);
            for (var o = 0; o < expected.Length; o++)
            {
                IlgpuTestHelpers.AssertDoubleArrayClose(expected[o], actual[o], $"seed {seed} output {o}");
            }
        }
    }

    [Fact]
    public void Ilgpu_int_array_matches_reference_exactly()
    {
        var functions = StandardInt32ArrayFunctionSet.Instance;
        var reference = new ReferenceGenomeEvaluator<int[]>(functions);
        using var ilgpu = new IlgpuGenomeEvaluator<int[]>(functions);
        var inputs = new[] { new[] { 3, -7, 11 }, new[] { 2, 5, -1 } };
        for (var seed = 0UL; seed < 10; seed++)
        {
            var genome = RandomInt32Array(seed, functions);
            var expected = new int[RandomShape.OutputCount][];
            var actual = new int[RandomShape.OutputCount][];
            reference.Evaluate(genome, inputs, expected);
            ilgpu.Evaluate(genome, inputs, actual);
            for (var o = 0; o < expected.Length; o++)
            {
                Assert.Equal(expected[o], actual[o]);
            }
        }
    }

    [Fact]
    public void Double_array_kernel_preserves_protected_functions_finiteness_order_counts_and_array_integrity()
    {
        var functions = StandardDoubleArrayFunctionSet.Instance;
        var genome = DoubleArrayScenarioGenome(functions);
        var reference = new ReferenceGenomeEvaluator<double[]>(functions);
        using var ilgpu = new IlgpuGenomeEvaluator<double[]>(functions);
        var inputs = new[]
        {
            new[] { 8.0, -4.0, double.NaN },
            new[] { 0.0, 2.0, 1.0 },
        };
        var originalInputs = inputs.Select(v => v.ToArray()).ToArray();
        var expected = new double[3][];
        var actual = new double[3][];
        reference.Evaluate(genome, inputs, expected);
        ilgpu.Evaluate(genome, inputs, actual);

        for (var o = 0; o < expected.Length; o++)
        {
            IlgpuTestHelpers.AssertDoubleArrayClose(expected[o], actual[o], $"output {o}");
        }

        Assert.Equal(actual[0], actual[2]);
        IlgpuTestHelpers.AssertRealKernelExecution(ilgpu, genome, functions);
        Assert.Equal(originalInputs[0], inputs[0]);
        Assert.Equal(originalInputs[1], inputs[1]);

        actual[0][0] = 12345.0;
        var rerun = new double[3][];
        ilgpu.Evaluate(genome, inputs, rerun);
        IlgpuTestHelpers.AssertDoubleArrayClose(expected[0], rerun[0], "rerun output 0");
    }

    [Fact]
    public void Int_array_kernel_preserves_extremes_order_counts_and_call_isolation()
    {
        var functions = StandardInt32ArrayFunctionSet.Instance;
        var genome = IntArrayScenarioGenome(functions);
        var reference = new ReferenceGenomeEvaluator<int[]>(functions);
        using var ilgpu = new IlgpuGenomeEvaluator<int[]>(functions);
        var inputs = new[]
        {
            new[] { 0, 2, 5 },
            new[] { 1, 1, 3 },
        };
        var expected = new int[3][];
        var actual = new int[3][];
        reference.Evaluate(genome, inputs, expected);
        ilgpu.Evaluate(genome, inputs, actual);

        for (var o = 0; o < expected.Length; o++)
        {
            Assert.Equal(expected[o], actual[o]);
        }

        Assert.Equal(actual[0], actual[2]);
        IlgpuTestHelpers.AssertRealKernelExecution(ilgpu, genome, functions);

        actual[1][0] = 777;
        var rerun = new int[3][];
        ilgpu.Evaluate(genome, inputs, rerun);
        Assert.Equal(expected[1], rerun[1]);
    }

    [Fact]
    public void Double_array_kernel_preserves_signed_zero_bits_elementwise_for_min_and_max()
    {
        var functions = StandardDoubleArrayFunctionSet.Instance;
        using var ilgpu = new IlgpuGenomeEvaluator<double[]>(functions);
        if (!ilgpu.Backend.StartsWith("ILGPU", StringComparison.Ordinal))
        {
            return;
        }

        var minId = IlgpuTestHelpers.FindFunctionId(functions, "min");
        var maxId = IlgpuTestHelpers.FindFunctionId(functions, "max");
        var shape = new GenomeShape(InputCount: 2, NodeCount: 2, OutputCount: 2, MaxArity: 2, LevelsBack: 2);
        var genome = Genome<double[]>.FromGenes(
            shape,
            [minId, maxId],
            [0, 1, 1, 0],
            [Array.Empty<double>(), Array.Empty<double>()],
            [2, 3],
            DoubleArrayValueSemantics.Instance,
            functions);
        var inputs = new[]
        {
            new[] { -0.0, 0.0 },
            new[] { 0.0, -0.0 },
        };
        var expected = new double[2][];
        var actual = new double[2][];
        new ReferenceGenomeEvaluator<double[]>(functions).Evaluate(genome, inputs, expected);
        ilgpu.Evaluate(genome, inputs, actual);

        IlgpuTestHelpers.AssertRealKernelExecution(ilgpu, genome, functions);
        for (var output = 0; output < expected.Length; output++)
        {
            for (var lane = 0; lane < expected[output].Length; lane++)
            {
                Assert.Equal(
                    BitConverter.DoubleToInt64Bits(expected[output][lane]),
                    BitConverter.DoubleToInt64Bits(actual[output][lane]));
            }
        }
    }

    [Fact]
    public void Double_array_kernel_preserves_nan_and_infinity_classification_elementwise()
    {
        var functions = StandardDoubleArrayFunctionSet.Instance;
        using var ilgpu = new IlgpuGenomeEvaluator<double[]>(functions);
        if (!ilgpu.Backend.StartsWith("ILGPU", StringComparison.Ordinal))
        {
            return;
        }

        var constId = IlgpuTestHelpers.FindFunctionId(functions, "const");
        var shape = new GenomeShape(InputCount: 1, NodeCount: 3, OutputCount: 3, MaxArity: 1, LevelsBack: 3);
        var genome = Genome<double[]>.FromGenes(
            shape,
            [constId, constId, constId],
            [0, 0, 0],
            [
                [double.NaN, 1.0],
                [double.PositiveInfinity, double.NegativeInfinity],
                [double.NegativeInfinity, double.PositiveInfinity],
            ],
            [1, 2, 3],
            DoubleArrayValueSemantics.Instance,
            functions);
        var expected = new double[3][];
        var actual = new double[3][];
        new ReferenceGenomeEvaluator<double[]>(functions).Evaluate(genome, [new[] { 7.0, 8.0 }], expected);
        ilgpu.Evaluate(genome, [new[] { 7.0, 8.0 }], actual);

        IlgpuTestHelpers.AssertRealKernelExecution(ilgpu, genome, functions);
        Assert.True(double.IsNaN(actual[0][0]));
        Assert.True(double.IsFinite(actual[0][1]));
        Assert.True(double.IsPositiveInfinity(actual[1][0]));
        Assert.True(double.IsNegativeInfinity(actual[1][1]));
        Assert.True(double.IsNegativeInfinity(actual[2][0]));
        Assert.True(double.IsPositiveInfinity(actual[2][1]));
        for (var i = 0; i < expected.Length; i++)
        {
            IlgpuTestHelpers.AssertDoubleArrayClose(expected[i], actual[i], $"output {i}");
        }
    }
}
