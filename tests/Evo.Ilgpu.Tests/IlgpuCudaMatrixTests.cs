using Evo.Core;
using Evo.Core.Functions;
using Evo.Core.Values;
using Evo.Ilgpu;
using Xunit;
using Xunit.Abstractions;

namespace Evo.Ilgpu.Tests;

public sealed class IlgpuCudaMatrixTests
{
    private readonly ITestOutputHelper _output;

    public IlgpuCudaMatrixTests(ITestOutputHelper output)
    {
        _output = output;
    }

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

    private static readonly double[][][] DoubleArrayEdgeInputs =
    {
        new[] { new[] { 0.0, -0.0, 1.0 }, new[] { -0.0, 0.0, -2.0 } },
        new[] { new[] { double.PositiveInfinity, 1.0, -1.0 }, new[] { double.NaN, 2.0, 0.5 } },
        new[] { new[] { 1e-300, -1e300, 3.0 }, new[] { -2.0, 0.0, 1e3 } },
    };

    private static readonly int[][][] IntArrayEdgeInputs =
    {
        new[] { new[] { 0, 0, 0 }, new[] { 0, 0, 0 } },
        new[] { new[] { int.MinValue, -1, 1 }, new[] { int.MaxValue, 1, 0 } },
        new[] { new[] { 7, -3, 2 }, new[] { 2, 5, -9 } },
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

        var og = new[]
        {
            rng.NextInt(0, Shape.InputCount + Shape.NodeCount),
            rng.NextInt(0, Shape.InputCount + Shape.NodeCount),
        };
        return Genome<double[]>.FromGenes(Shape, fg, cg, pg, og, DoubleArrayValueSemantics.Instance, functions);
    }

    private static Genome<int[]> RandomIntArray(ulong seed, IFunctionSet<int[]> functions)
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

            pg[n] = [rng.NextInt(-5, 6), rng.NextInt(-5, 6), rng.NextInt(-5, 6)];
        }

        var og = new[]
        {
            rng.NextInt(0, Shape.InputCount + Shape.NodeCount),
            rng.NextInt(0, Shape.InputCount + Shape.NodeCount),
        };
        return Genome<int[]>.FromGenes(Shape, fg, cg, pg, og, Int32ArrayValueSemantics.Instance, functions);
    }

    // DAG com subárvores compartilhadas: todos os nós a partir do 2º referenciam o mesmo par de nós anteriores.
    private static Genome<double> SharedSubtreeDouble(ulong seed, IFunctionSet<double> functions)
    {
        var rng = RandomStream.Create(seed, 1, 0, 0);
        var fg = new int[Shape.NodeCount];
        var cg = new int[Shape.NodeCount * Shape.MaxArity];
        var pg = new double[Shape.NodeCount];
        for (var n = 0; n < Shape.NodeCount; n++)
        {
            fg[n] = rng.NextInt(0, functions.Count);
            for (var a = 0; a < Shape.MaxArity; a++)
            {
                cg[n * Shape.MaxArity + a] = n < 2 ? rng.NextInt(0, Shape.InputCount + n) : (a == 0 ? 2 : Shape.InputCount + 1 + (n % 2));
            }

            pg[n] = rng.NextDouble() * 4 - 2;
        }

        var og = new[] { Shape.InputCount + Shape.NodeCount - 1, Shape.InputCount + 2 };
        return Genome<double>.FromGenes(Shape, fg, cg, pg, og, DoubleValueSemantics.Instance, functions);
    }

    private static Genome<int> SharedSubtreeInt(ulong seed, IFunctionSet<int> functions)
    {
        var rng = RandomStream.Create(seed, 1, 0, 0);
        var fg = new int[Shape.NodeCount];
        var cg = new int[Shape.NodeCount * Shape.MaxArity];
        var pg = new int[Shape.NodeCount];
        for (var n = 0; n < Shape.NodeCount; n++)
        {
            fg[n] = rng.NextInt(0, functions.Count);
            for (var a = 0; a < Shape.MaxArity; a++)
            {
                cg[n * Shape.MaxArity + a] = n < 2 ? rng.NextInt(0, Shape.InputCount + n) : (a == 0 ? 2 : Shape.InputCount + 1 + (n % 2));
            }

            pg[n] = rng.NextInt(-5, 6);
        }

        var og = new[] { Shape.InputCount + Shape.NodeCount - 1, Shape.InputCount + 2 };
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

    private static bool SameIntArray(int[][] expected, int[][] actual)
    {
        if (expected.Length != actual.Length)
        {
            return false;
        }

        for (var o = 0; o < expected.Length; o++)
        {
            if (!expected[o].AsSpan().SequenceEqual(actual[o]))
            {
                return false;
            }
        }

        return true;
    }

    private static void AssertDoubleArraysSame(double[][] expected, double[][] actual, string context)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (var o = 0; o < expected.Length; o++)
        {
            Assert.Equal(expected[o].Length, actual[o].Length);
            for (var i = 0; i < expected[o].Length; i++)
            {
                Assert.True(SameDouble(expected[o][i], actual[o][i]), $"{context} out={o}.{i} esperado={expected[o][i]} obtido={actual[o][i]}");
            }
        }
    }

    private static IlgpuBackendSelection RequireCuda()
    {
        try
        {
            var selection = IlgpuBackendSelector.SelectFor(IlgpuBackendPolicy.CudaRequired);
            Assert.Equal(IlgpuBackendKind.CudaDevice, selection.Kind);
            return selection;
        }
        catch (InvalidOperationException ex)
        {
            Assert.Fail($"CudaRequired sem dispositivo CUDA válido; a certificação exige execução real na GTX 1650: {ex.Message}");
            throw;
        }
    }

    [Fact]
    public void Cuda_matches_reference_for_double_functions_edges_and_signed_zero()
    {
        RequireCuda();
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
        RequireCuda();
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

    [Fact]
    public void Cuda_matches_reference_for_double_array_functions_and_edges()
    {
        RequireCuda();
        var functions = StandardDoubleArrayFunctionSet.Instance;
        var reference = new ReferenceGenomeEvaluator<double[]>(functions);
        using var cuda = new IlgpuGenomeEvaluator<double[]>(functions, IlgpuBackendPolicy.CudaRequired);

        for (var seed = 0UL; seed < 10; seed++)
        {
            var genome = RandomDoubleArray(seed, functions);
            foreach (var inputs in DoubleArrayEdgeInputs)
            {
                var expected = new double[2][];
                var actual = new double[2][];
                reference.Evaluate(genome, inputs, expected);
                cuda.Evaluate(genome, inputs, actual);
                Assert.True(cuda.LastEvaluationUsedDevice, $"seed={seed} não executou no dispositivo: {cuda.LastFallbackReason}");
                AssertDoubleArraysSame(expected, actual, $"seed={seed}");
            }
        }
    }

    [Fact]
    public void Cuda_matches_reference_for_int_array_functions_and_edges()
    {
        RequireCuda();
        var functions = StandardInt32ArrayFunctionSet.Instance;
        var reference = new ReferenceGenomeEvaluator<int[]>(functions);
        using var cuda = new IlgpuGenomeEvaluator<int[]>(functions, IlgpuBackendPolicy.CudaRequired);

        for (var seed = 0UL; seed < 10; seed++)
        {
            var genome = RandomIntArray(seed, functions);
            foreach (var inputs in IntArrayEdgeInputs)
            {
                var expected = new int[2][];
                var actual = new int[2][];
                reference.Evaluate(genome, inputs, expected);
                cuda.Evaluate(genome, inputs, actual);
                Assert.True(cuda.LastEvaluationUsedDevice, $"seed={seed} não executou no dispositivo: {cuda.LastFallbackReason}");
                Assert.True(SameIntArray(expected, actual), $"seed={seed} esperado e obtido divergem");
            }
        }
    }

    [Fact]
    public void Cuda_matches_reference_for_dag_with_shared_subtrees_double_and_int()
    {
        RequireCuda();
        var doubleFunctions = StandardDoubleFunctionSet.Instance;
        var doubleReference = new ReferenceGenomeEvaluator<double>(doubleFunctions);
        using var doubleCuda = new IlgpuGenomeEvaluator<double>(doubleFunctions, IlgpuBackendPolicy.CudaRequired);
        var intFunctions = StandardInt32FunctionSet.Instance;
        var intReference = new ReferenceGenomeEvaluator<int>(intFunctions);
        using var intCuda = new IlgpuGenomeEvaluator<int>(intFunctions, IlgpuBackendPolicy.CudaRequired);

        for (var seed = 0UL; seed < 10; seed++)
        {
            var dGenome = SharedSubtreeDouble(seed, doubleFunctions);
            var iGenome = SharedSubtreeInt(seed, intFunctions);
            foreach (var inputs in DoubleEdgeInputs)
            {
                var expected = new double[2];
                var actual = new double[2];
                doubleReference.Evaluate(dGenome, inputs, expected);
                doubleCuda.Evaluate(dGenome, inputs, actual);
                Assert.True(doubleCuda.LastEvaluationUsedDevice, $"dag seed={seed} não executou no dispositivo: {doubleCuda.LastFallbackReason}");
                for (var i = 0; i < expected.Length; i++)
                {
                    Assert.True(SameDouble(expected[i], actual[i]), $"dag double seed={seed} out={i} esperado={expected[i]} obtido={actual[i]}");
                }
            }

            foreach (var inputs in IntEdgeInputs)
            {
                var expected = new int[2];
                var actual = new int[2];
                intReference.Evaluate(iGenome, inputs, expected);
                intCuda.Evaluate(iGenome, inputs, actual);
                Assert.True(intCuda.LastEvaluationUsedDevice, $"dag seed={seed} não executou no dispositivo: {intCuda.LastFallbackReason}");
                Assert.Equal(expected, actual);
            }
        }
    }

    // Regressão #1615: joins inteiros (Select/Min/Max/GreaterThan) devem produzir valores fora de 0/1
    // e bater com a referência. A versão do ILGPU é registrada na saída do teste e em evidence.
    [Fact]
    public void Regression_1615_integer_joins_with_values_outside_zero_one_match_reference()
    {
        var ilgpuVersion = typeof(ILGPU.Context).Assembly.GetName().Version?.ToString() ?? "desconhecida";
        _output.WriteLine($"ILGPU versão={ilgpuVersion}");
        RequireCuda();

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
}
