using Evo.Core;
using Evo.Core.Functions;
using Evo.Core.Values;
using Evo.Ilgpu;
using Xunit;

namespace Evo.Ilgpu.Tests;

public sealed class IlgpuCudaMatrixTests
{
    private static readonly GenomeShape Shape = new(InputCount: 2, NodeCount: 8, OutputCount: 2, MaxArity: 3, LevelsBack: 8);

    private static readonly double[][] DoubleEdgeInputs =
    {
        new[] { 0.0, -0.0 },
        new[] { -0.0, 0.0 },
        new[] { double.PositiveInfinity, 1.0 },
        new[] { double.NegativeInfinity, -1.0 },
        new[] { double.NaN, 2.0 },
        new[] { 1e-300, -1e300 },
    };

    private static readonly int[][] IntEdgeInputs =
    {
        new[] { 0, 0 },
        new[] { int.MinValue, -1 },
        new[] { int.MaxValue, 1 },
        new[] { 7, -3 },
        new[] { 2, 5 },
    };

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

        var og = new[]
        {
            rng.NextInt(0, Shape.InputCount + Shape.NodeCount),
            rng.NextInt(0, Shape.InputCount + Shape.NodeCount),
        };
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

            pg[n] = rng.NextInt(-5, 6);
        }

        var og = new[]
        {
            rng.NextInt(0, Shape.InputCount + Shape.NodeCount),
            rng.NextInt(0, Shape.InputCount + Shape.NodeCount),
        };
        return Genome<int>.FromGenes(Shape, fg, cg, pg, og, Int32ValueSemantics.Instance, functions);
    }

    private static bool SameDouble(double expected, double actual)
    {
        if (double.IsNaN(expected) || double.IsNaN(actual))
        {
            return double.IsNaN(expected) && double.IsNaN(actual);
        }

        if (double.IsInfinity(expected) || double.IsInfinity(actual))
        {
            return expected == actual;
        }

        // Zero com sinal: a referência e a GPU devem concordar também no bit de sinal.
        if (expected == 0.0 && actual == 0.0)
        {
            return BitConverter.DoubleToInt64Bits(expected) == BitConverter.DoubleToInt64Bits(actual);
        }

        var tolerance = 1e-12 * Math.Max(1, Math.Abs(expected));
        return Math.Abs(expected - actual) <= tolerance;
    }

    [Fact]
    public void Cuda_matches_reference_for_double_functions_edges_and_signed_zero()
    {
        if (!TryCudaRequired(out var selection))
        {
            return;
        }

        Assert.Equal(IlgpuBackendKind.CudaDevice, selection.Kind);
        var functions = StandardDoubleFunctionSet.Instance;
        var reference = new ReferenceGenomeEvaluator<double>(functions);
        using var cuda = new IlgpuGenomeEvaluator<double>(functions, IlgpuBackendPolicy.CudaRequired);

        for (var seed = 0UL; seed < 20; seed++)
        {
            var genome = RandomDouble(seed, functions);
            foreach (var inputs in DoubleEdgeInputs)
            {
                var expected = new double[2];
                var actual = new double[2];
                reference.Evaluate(genome, inputs, expected);
                cuda.Evaluate(genome, inputs, actual);
                Assert.True(cuda.LastEvaluationUsedDevice, $"seed={seed} não executou no dispositivo: {cuda.LastFallbackReason}");
                for (var i = 0; i < expected.Length; i++)
                {
                    Assert.True(SameDouble(expected[i], actual[i]), $"seed={seed} out={i} esperado={expected[i]} obtido={actual[i]}");
                }
            }
        }
    }

    [Fact]
    public void Cuda_matches_reference_for_int_functions_and_edges()
    {
        if (!TryCudaRequired(out var selection))
        {
            return;
        }

        Assert.Equal(IlgpuBackendKind.CudaDevice, selection.Kind);
        var functions = StandardInt32FunctionSet.Instance;
        var reference = new ReferenceGenomeEvaluator<int>(functions);
        using var cuda = new IlgpuGenomeEvaluator<int>(functions, IlgpuBackendPolicy.CudaRequired);

        for (var seed = 0UL; seed < 20; seed++)
        {
            var genome = RandomInt(seed, functions);
            foreach (var inputs in IntEdgeInputs)
            {
                var expected = new int[2];
                var actual = new int[2];
                reference.Evaluate(genome, inputs, expected);
                cuda.Evaluate(genome, inputs, actual);
                Assert.True(cuda.LastEvaluationUsedDevice, $"seed={seed} não executou no dispositivo: {cuda.LastFallbackReason}");
                Assert.Equal(expected, actual);
            }
        }
    }

    // Regressão #1615: joins inteiros (Select/Min/Max/GreaterThan) devem produzir valores fora de 0/1
    // e bater com a referência. A versão do ILGPU é registrada no nome do diagnóstico.
    [Fact]
    public void Regression_1615_integer_joins_with_values_outside_zero_one_match_reference()
    {
        if (!TryCudaRequired(out var selection))
        {
            return;
        }

        var ilgpuVersion = typeof(ILGPU.Context).Assembly.GetName().Version?.ToString() ?? "desconhecida";
        Assert.False(string.IsNullOrWhiteSpace(ilgpuVersion));
        Assert.Equal(IlgpuBackendKind.CudaDevice, selection.Kind);

        var functions = StandardInt32FunctionSet.Instance;
        var reference = new ReferenceGenomeEvaluator<int>(functions);
        using var cuda = new IlgpuGenomeEvaluator<int>(functions, IlgpuBackendPolicy.CudaRequired);
        var inputs = new[] { 41, -17 };

        for (var seed = 0UL; seed < 20; seed++)
        {
            var genome = RandomInt(seed, functions);
            var expected = new int[2];
            var actual = new int[2];
            reference.Evaluate(genome, inputs, expected);
            cuda.Evaluate(genome, inputs, actual);
            Assert.True(cuda.LastEvaluationUsedDevice, $"ilgpu={ilgpuVersion} seed={seed} não executou no dispositivo");
            Assert.Equal(expected, actual);
        }
    }

    private static bool TryCudaRequired(out IlgpuBackendSelection selection)
    {
        try
        {
            selection = IlgpuBackendSelector.SelectFor(IlgpuBackendPolicy.CudaRequired);
            return true;
        }
        catch (InvalidOperationException)
        {
            // Sem CUDA válido nesta máquina: a limitação é registrada em evidence; não é aprovação CUDA.
            selection = default!;
            return false;
        }
    }
}
