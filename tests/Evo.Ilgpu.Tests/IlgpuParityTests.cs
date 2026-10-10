using Evo.Core;
using Evo.Core.Functions;
using Evo.Core.Values;
using Evo.Ilgpu;
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
}
