using Evo.Core;
using Evo.Core.Functions;
using Evo.Core.Values;
using Evo.Ilgpu;
using Xunit;

namespace Evo.Ilgpu.Tests;

public sealed class IlgpuParityTests
{
    private static readonly GenomeShape Shape = new(InputCount: 2, NodeCount: 8, OutputCount: 2, MaxArity: 3, LevelsBack: 8);

    private static Genome<double> RandomDouble(ulong seed, IFunctionSet<double> functions)
    {
        var rng = RandomStream.Create(seed, 0, 0, 0);
        var fg = new int[Shape.NodeCount];
        var cg = new int[Shape.NodeCount * Shape.MaxArity];
        var pg = new double[Shape.NodeCount];
        for (var n = 0; n < Shape.NodeCount; n++)
        {
            fg[n] = rng.NextInt(0, functions.Count);
            for (var a = 0; a < Shape.MaxArity; a++)
            {
                cg[n * Shape.MaxArity + a] = rng.NextInt(0, Shape.InputCount + n);
            }

            pg[n] = rng.NextDouble() * 4 - 2;
        }

        var og = new[] { rng.NextInt(0, Shape.InputCount + Shape.NodeCount), rng.NextInt(0, Shape.InputCount + Shape.NodeCount) };
        return Genome<double>.FromGenes(Shape, fg, cg, pg, og, DoubleValueSemantics.Instance, functions);
    }

    private static Genome<int> RandomInt(ulong seed, IFunctionSet<int> functions)
    {
        var rng = RandomStream.Create(seed, 0, 0, 0);
        var fg = new int[Shape.NodeCount];
        var cg = new int[Shape.NodeCount * Shape.MaxArity];
        var pg = new int[Shape.NodeCount];
        for (var n = 0; n < Shape.NodeCount; n++)
        {
            fg[n] = rng.NextInt(0, functions.Count);
            for (var a = 0; a < Shape.MaxArity; a++)
            {
                cg[n * Shape.MaxArity + a] = rng.NextInt(0, Shape.InputCount + n);
            }

            pg[n] = rng.NextInt(-20, 20);
        }

        var og = new[] { rng.NextInt(0, Shape.InputCount + Shape.NodeCount), rng.NextInt(0, Shape.InputCount + Shape.NodeCount) };
        return Genome<int>.FromGenes(Shape, fg, cg, pg, og, Int32ValueSemantics.Instance, functions);
    }

    private static void AssertDoubleClose(double expected, double actual)
    {
        if (double.IsFinite(expected) != double.IsFinite(actual))
        {
            Assert.Fail($"finiteness mismatch: expected {expected}, actual {actual}");
        }

        if (!double.IsFinite(expected))
        {
            return;
        }

        var diff = Math.Abs(expected - actual);
        Assert.True(diff <= 1e-12 || diff <= 1e-12 * Math.Abs(expected), $"expected {expected}, actual {actual}");
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
            var expected = new double[Shape.OutputCount];
            var actual = new double[Shape.OutputCount];
            reference.Evaluate(genome, inputs, expected);
            ilgpu.Evaluate(genome, inputs, actual);
            for (var o = 0; o < expected.Length; o++)
            {
                AssertDoubleClose(expected[o], actual[o]);
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
            var expected = new int[Shape.OutputCount];
            var actual = new int[Shape.OutputCount];
            reference.Evaluate(genome, inputs, expected);
            ilgpu.Evaluate(genome, inputs, actual);
            Assert.Equal(expected, actual);
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

