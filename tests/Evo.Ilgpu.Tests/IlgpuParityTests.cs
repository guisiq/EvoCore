using Evo.Core;
using Evo.Core.Functions;
using Evo.Core.Values;
using Evo.Ilgpu;
using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.OpenCL;
using Xunit;

namespace Evo.Ilgpu.Tests;

public sealed class IlgpuParityTests
{
    private static readonly GenomeShape RandomShape = new(InputCount: 2, NodeCount: 8, OutputCount: 2, MaxArity: 3, LevelsBack: 8);

    private static Genome<double> RandomDouble(ulong seed, IFunctionSet<double> functions)
    {
        var rng = RandomStream.Create(seed, 0, 0, 0);
        var fg = new int[RandomShape.NodeCount];
        var cg = new int[RandomShape.NodeCount * RandomShape.MaxArity];
        var pg = new double[RandomShape.NodeCount];
        for (var n = 0; n < RandomShape.NodeCount; n++)
        {
            fg[n] = rng.NextInt(0, functions.Count);
            for (var a = 0; a < RandomShape.MaxArity; a++)
            {
                cg[n * RandomShape.MaxArity + a] = rng.NextInt(0, RandomShape.InputCount + n);
            }

            pg[n] = rng.NextDouble() * 4 - 2;
        }

        var og = new[]
        {
            rng.NextInt(0, RandomShape.InputCount + RandomShape.NodeCount),
            rng.NextInt(0, RandomShape.InputCount + RandomShape.NodeCount),
        };
        return Genome<double>.FromGenes(RandomShape, fg, cg, pg, og, DoubleValueSemantics.Instance, functions);
    }

    private static Genome<int> RandomInt(ulong seed, IFunctionSet<int> functions)
    {
        var rng = RandomStream.Create(seed, 0, 0, 0);
        var fg = new int[RandomShape.NodeCount];
        var cg = new int[RandomShape.NodeCount * RandomShape.MaxArity];
        var pg = new int[RandomShape.NodeCount];
        for (var n = 0; n < RandomShape.NodeCount; n++)
        {
            fg[n] = rng.NextInt(0, functions.Count);
            for (var a = 0; a < RandomShape.MaxArity; a++)
            {
                cg[n * RandomShape.MaxArity + a] = rng.NextInt(0, RandomShape.InputCount + n);
            }

            pg[n] = rng.NextInt(-20, 20);
        }

        var og = new[]
        {
            rng.NextInt(0, RandomShape.InputCount + RandomShape.NodeCount),
            rng.NextInt(0, RandomShape.InputCount + RandomShape.NodeCount),
        };
        return Genome<int>.FromGenes(RandomShape, fg, cg, pg, og, Int32ValueSemantics.Instance, functions);
    }

    private static Genome<double> DoubleScenarioGenome(IFunctionSet<double> functions)
    {
        var shape = new GenomeShape(InputCount: 2, NodeCount: 8, OutputCount: 3, MaxArity: 3, LevelsBack: 8);
        var constantId = IlgpuTestHelpers.FindFunctionId(functions, "const");
        var sqrtId = IlgpuTestHelpers.FindFunctionId(functions, "sqrt");
        var logId = IlgpuTestHelpers.FindFunctionId(functions, "log");
        var addId = IlgpuTestHelpers.FindFunctionId(functions, "add");
        var divideId = IlgpuTestHelpers.FindFunctionId(functions, "divide");
        var greaterId = IlgpuTestHelpers.FindFunctionId(functions, "greater");
        var selectId = IlgpuTestHelpers.FindFunctionId(functions, "select");

        return Genome<double>.FromGenes(
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
            [-4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [8, 5, 8],
            DoubleValueSemantics.Instance,
            functions);
    }

    private static Genome<int> IntScenarioGenome(IFunctionSet<int> functions)
    {
        var shape = new GenomeShape(InputCount: 2, NodeCount: 8, OutputCount: 3, MaxArity: 3, LevelsBack: 8);
        var constantId = IlgpuTestHelpers.FindFunctionId(functions, "const");
        var absId = IlgpuTestHelpers.FindFunctionId(functions, "abs");
        var divideId = IlgpuTestHelpers.FindFunctionId(functions, "divide");
        var addId = IlgpuTestHelpers.FindFunctionId(functions, "add");
        var greaterId = IlgpuTestHelpers.FindFunctionId(functions, "greater");
        var selectId = IlgpuTestHelpers.FindFunctionId(functions, "select");

        return Genome<int>.FromGenes(
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
            [int.MinValue, 0, -1, 0, 0, 0, 0, 0],
            [8, 6, 8],
            Int32ValueSemantics.Instance,
            functions);
    }

    [Fact]
    public void Ilgpu_double_matches_reference_within_1e12()
    {
        var functions = StandardDoubleFunctionSet.Instance;
        var reference = new ReferenceGenomeEvaluator<double>(functions);
        using var ilgpu = new IlgpuGenomeEvaluator<double>(functions);
        var inputs = new[] { 0.5, -1.25 };
        for (var seed = 0UL; seed < 10; seed++)
        {
            var genome = RandomDouble(seed, functions);
            var expected = new double[RandomShape.OutputCount];
            var actual = new double[RandomShape.OutputCount];
            reference.Evaluate(genome, inputs, expected);
            ilgpu.Evaluate(genome, inputs, actual);
            for (var o = 0; o < expected.Length; o++)
            {
                IlgpuTestHelpers.AssertDoubleClose(expected[o], actual[o]);
            }
        }
    }

    [Fact]
    public void Ilgpu_int_matches_reference_exactly()
    {
        var functions = StandardInt32FunctionSet.Instance;
        var reference = new ReferenceGenomeEvaluator<int>(functions);
        using var ilgpu = new IlgpuGenomeEvaluator<int>(functions);
        var inputs = new[] { 3, -7 };
        for (var seed = 0UL; seed < 10; seed++)
        {
            var genome = RandomInt(seed, functions);
            var expected = new int[RandomShape.OutputCount];
            var actual = new int[RandomShape.OutputCount];
            reference.Evaluate(genome, inputs, expected);
            ilgpu.Evaluate(genome, inputs, actual);
            Assert.Equal(expected, actual);
        }
    }

    [Fact]
    public void Double_kernel_preserves_protected_functions_finiteness_output_order_and_inactive_counts()
    {
        var functions = StandardDoubleFunctionSet.Instance;
        var genome = DoubleScenarioGenome(functions);
        var reference = new ReferenceGenomeEvaluator<double>(functions);
        using var ilgpu = new IlgpuGenomeEvaluator<double>(functions);
        var cases = new[]
        {
            new[] { 7.0, 0.0 },
            new[] { double.NaN, 0.0 },
        };

        foreach (var inputs in cases)
        {
            var expected = new double[3];
            var actual = new double[3];
            reference.Evaluate(genome, inputs, expected);
            ilgpu.Evaluate(genome, inputs, actual);

            Assert.Equal(expected.Length, actual.Length);
            for (var i = 0; i < expected.Length; i++)
            {
                IlgpuTestHelpers.AssertDoubleClose(expected[i], actual[i], $"output {i}");
            }

            Assert.Equal(actual[0], actual[2]);
            IlgpuTestHelpers.AssertRealKernelExecution(ilgpu, genome, functions);
        }
    }

    [Fact]
    public void Int_kernel_preserves_extremes_reconvergence_output_order_and_inactive_counts()
    {
        var functions = StandardInt32FunctionSet.Instance;
        var genome = IntScenarioGenome(functions);
        var reference = new ReferenceGenomeEvaluator<int>(functions);
        using var ilgpu = new IlgpuGenomeEvaluator<int>(functions);
        var cases = new[]
        {
            new[] { 0, 1 },
            new[] { 2, 1 },
        };

        foreach (var inputs in cases)
        {
            var expected = new int[3];
            var actual = new int[3];
            reference.Evaluate(genome, inputs, expected);
            ilgpu.Evaluate(genome, inputs, actual);

            Assert.Equal(expected, actual);
            Assert.Equal(actual[0], actual[2]);
            IlgpuTestHelpers.AssertRealKernelExecution(ilgpu, genome, functions);
        }
    }

    [Fact]
    public void Backend_selection_is_stable_and_reports_name()
    {
        var first = IlgpuBackendSelector.Selection;
        var second = IlgpuBackendSelector.Selection;
        Assert.Same(first, second);
        Assert.False(string.IsNullOrEmpty(first.BackendName));
    }

    [Fact]
    public void Double_kernel_preserves_signed_zero_bits_for_min_and_max()
    {
        var functions = StandardDoubleFunctionSet.Instance;
        using var ilgpu = new IlgpuGenomeEvaluator<double>(functions);
        if (!ilgpu.Backend.StartsWith("ILGPU", StringComparison.Ordinal))
        {
            return;
        }

        var minId = IlgpuTestHelpers.FindFunctionId(functions, "min");
        var maxId = IlgpuTestHelpers.FindFunctionId(functions, "max");
        var shape = new GenomeShape(InputCount: 2, NodeCount: 2, OutputCount: 2, MaxArity: 2, LevelsBack: 2);
        var genome = Genome<double>.FromGenes(
            shape,
            [minId, maxId],
            [0, 1, 1, 0],
            [0.0, 0.0],
            [2, 3],
            DoubleValueSemantics.Instance,
            functions);
        var inputs = new[] { -0.0, 0.0 };
        var expected = new double[2];
        var actual = new double[2];
        new ReferenceGenomeEvaluator<double>(functions).Evaluate(genome, inputs, expected);
        ilgpu.Evaluate(genome, inputs, actual);

        IlgpuTestHelpers.AssertRealKernelExecution(ilgpu, genome, functions);
        Assert.Equal(BitConverter.DoubleToInt64Bits(expected[0]), BitConverter.DoubleToInt64Bits(actual[0]));
        Assert.Equal(BitConverter.DoubleToInt64Bits(expected[1]), BitConverter.DoubleToInt64Bits(actual[1]));
        Assert.Equal(unchecked((long)0x8000000000000000), BitConverter.DoubleToInt64Bits(actual[0]));
        Assert.Equal(0L, BitConverter.DoubleToInt64Bits(actual[1]));
    }

    [Fact]
    public void Double_kernel_preserves_nan_and_infinity_classification()
    {
        var functions = StandardDoubleFunctionSet.Instance;
        using var ilgpu = new IlgpuGenomeEvaluator<double>(functions);
        if (!ilgpu.Backend.StartsWith("ILGPU", StringComparison.Ordinal))
        {
            return;
        }

        var constId = IlgpuTestHelpers.FindFunctionId(functions, "const");
        var shape = new GenomeShape(InputCount: 1, NodeCount: 3, OutputCount: 3, MaxArity: 1, LevelsBack: 3);
        var genome = Genome<double>.FromGenes(
            shape,
            [constId, constId, constId],
            [0, 0, 0],
            [double.NaN, double.PositiveInfinity, double.NegativeInfinity],
            [1, 2, 3],
            DoubleValueSemantics.Instance,
            functions);
        var expected = new double[3];
        var actual = new double[3];
        new ReferenceGenomeEvaluator<double>(functions).Evaluate(genome, [1.0], expected);
        ilgpu.Evaluate(genome, [1.0], actual);

        IlgpuTestHelpers.AssertRealKernelExecution(ilgpu, genome, functions);
        Assert.True(double.IsNaN(actual[0]));
        Assert.True(double.IsPositiveInfinity(actual[1]));
        Assert.True(double.IsNegativeInfinity(actual[2]));
        for (var i = 0; i < expected.Length; i++)
        {
            IlgpuTestHelpers.AssertDoubleClose(expected[i], actual[i], $"output {i}");
        }
    }

    [Fact]
    public void OpenCl_device_loads_and_executes_min_max_signed_zero_and_non_finite_without_cpu_fallback()
    {
        using var probeContext = Context.Create(builder => builder.OpenCL());
        var clDevices = probeContext.Devices.OfType<CLDevice>().ToArray();
        if (clDevices.Length == 0)
        {
            return;
        }

        var device = clDevices[0];
        var selection = new IlgpuBackendSelection(
            IlgpuBackendKind.OpenClDevice,
            Array.IndexOf(probeContext.Devices.ToArray(), (ILGPU.Runtime.Device)device),
            device.Name,
            null,
            device.WarpSize,
            device.MaxNumThreads,
            device.NumMultiprocessors,
            device.MaxNumThreadsPerGroup,
            CpuNumThreads: 0,
            PlatformName: device.PlatformName,
            PlatformId: device.PlatformId.ToString(),
            OpenClDeviceId: device.DeviceId.ToString());

        foreach (var (functions, buildGenome, inputs, assertActual) in new (
            IFunctionSet<double> functions,
            Func<IFunctionSet<double>, Genome<double>> buildGenome,
            double[] inputs,
            Action<double[], double[]> assertActual)[]
        {
            (
                StandardDoubleFunctionSet.Instance,
                fs =>
                {
                    var minId = IlgpuTestHelpers.FindFunctionId(fs, "min");
                    var maxId = IlgpuTestHelpers.FindFunctionId(fs, "max");
                    var shape = new GenomeShape(InputCount: 2, NodeCount: 2, OutputCount: 2, MaxArity: 2, LevelsBack: 2);
                    return Genome<double>.FromGenes(shape, [minId, maxId], [0, 1, 1, 0], [0.0, 0.0], [2, 3], DoubleValueSemantics.Instance, fs);
                },
                [-0.0, 0.0],
                (expected, actual) =>
                {
                    Assert.Equal(unchecked((long)0x8000000000000000), BitConverter.DoubleToInt64Bits(actual[0]));
                    Assert.Equal(0L, BitConverter.DoubleToInt64Bits(actual[1]));
                }
            ),
            (
                StandardDoubleFunctionSet.Instance,
                fs =>
                {
                    var constId = IlgpuTestHelpers.FindFunctionId(fs, "const");
                    var shape = new GenomeShape(InputCount: 1, NodeCount: 3, OutputCount: 3, MaxArity: 1, LevelsBack: 3);
                    return Genome<double>.FromGenes(shape, [constId, constId, constId], [0, 0, 0],
                        [double.NaN, double.PositiveInfinity, double.NegativeInfinity], [1, 2, 3], DoubleValueSemantics.Instance, fs);
                },
                [1.0],
                (expected, actual) =>
                {
                    Assert.True(double.IsNaN(actual[0]));
                    Assert.True(double.IsPositiveInfinity(actual[1]));
                    Assert.True(double.IsNegativeInfinity(actual[2]));
                }
            ),
        })
        {
            var genome = buildGenome(functions);
            var expected = new double[genome.Shape.OutputCount];
            var actual = new double[genome.Shape.OutputCount];
            new ReferenceGenomeEvaluator<double>(functions).Evaluate(genome, inputs, expected);

            using var evaluator = new IlgpuGenomeEvaluator<double>(
                functions,
                fallback: null,
                selectionOverride: selection,
                contextFactory: _ => Context.Create(builder => builder.OpenCL()),
                acceleratorFactory: null,
                beforeKernelExecution: null);

            evaluator.Evaluate(genome, inputs, actual);

            Assert.Equal("ILGPU/OpenCL", evaluator.Backend);
            Assert.Null(evaluator.LastFallbackReason);
            assertActual(expected, actual);
        }

        // Paridade inteira no mesmo dispositivo OpenCL, sem fallback para CPU.
        var intFunctions = StandardInt32FunctionSet.Instance;
        var addId = IlgpuTestHelpers.FindFunctionId(intFunctions, "add");
        var intShape = new GenomeShape(InputCount: 2, NodeCount: 1, OutputCount: 1, MaxArity: 2, LevelsBack: 1);
        var intGenome = Genome<int>.FromGenes(intShape, [addId], [0, 1], [0], [2], Int32ValueSemantics.Instance, intFunctions);
        using var intEvaluator = new IlgpuGenomeEvaluator<int>(
            intFunctions,
            fallback: null,
            selectionOverride: selection,
            contextFactory: _ => Context.Create(builder => builder.OpenCL()),
            acceleratorFactory: null,
            beforeKernelExecution: null);
        var intActual = new int[1];
        intEvaluator.Evaluate(intGenome, [1, 2], intActual);
        Assert.Equal("ILGPU/OpenCL", intEvaluator.Backend);
        Assert.Equal(3, intActual[0]);
    }
}
