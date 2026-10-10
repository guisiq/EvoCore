using Evo.Core;
using Evo.Core.Functions;
using Evo.Core.Values;
using Evo.Ilgpu;
using Xunit;

namespace Evo.Ilgpu.Tests;

public sealed class IlgpuArrayParityTests
{
    private static readonly GenomeShape Shape = new(InputCount: 2, NodeCount: 8, OutputCount: 2, MaxArity: 3, LevelsBack: 8);

    private static Genome<double[]> RandomDoubleArray(ulong seed, IFunctionSet<double[]> functions)
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

    private static Genome<int[]> RandomInt32Array(ulong seed, IFunctionSet<int[]> functions)
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
    public void Ilgpu_double_array_matches_reference_elementwise_within_1e12()
    {
        var functions = StandardDoubleArrayFunctionSet.Instance;
        var reference = new ReferenceGenomeEvaluator<double[]>(functions);
        using var ilgpu = new IlgpuGenomeEvaluator<double[]>(functions);
        var inputs = new[] { new[] { 0.5, -1.25, 3.0 }, new[] { -2.0, 0.0, 1e3 } };
        for (var seed = 0UL; seed < 10; seed++)
        {
            var genome = RandomDoubleArray(seed, functions);
            var expected = new double[Shape.OutputCount][];
            var actual = new double[Shape.OutputCount][];
            reference.Evaluate(genome, inputs, expected);
            ilgpu.Evaluate(genome, inputs, actual);
            for (var o = 0; o < expected.Length; o++)
            {
                Assert.Equal(expected[o].Length, actual[o].Length);
                for (var i = 0; i < expected[o].Length; i++)
                {
                    var e = expected[o][i];
                    var a = actual[o][i];
                    Assert.Equal(double.IsFinite(e), double.IsFinite(a));
                    if (!double.IsFinite(e))
                    {
                        continue;
                    }

                    var diff = Math.Abs(e - a);
                    Assert.True(diff <= 1e-12 || diff <= 1e-12 * Math.Abs(e), $"seed {seed} out {o} lane {i}: expected {e}, actual {a}");
                }
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
            var expected = new int[Shape.OutputCount][];
            var actual = new int[Shape.OutputCount][];
            reference.Evaluate(genome, inputs, expected);
            ilgpu.Evaluate(genome, inputs, actual);
            for (var o = 0; o < expected.Length; o++)
            {
                Assert.Equal(expected[o], actual[o]);
            }
        }
    }
}
