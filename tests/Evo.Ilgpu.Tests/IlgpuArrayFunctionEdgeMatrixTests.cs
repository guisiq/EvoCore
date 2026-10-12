using Evo.Core;
using Evo.Core.Compiled;
using Evo.Core.Functions;
using Evo.Core.Values;
using Evo.Ilgpu;
using Xunit;

namespace Evo.Ilgpu.Tests;

/// <summary>
/// Matriz função × borda × dispositivo para <c>double[]</c> e <c>int[]</c> (EVO-102, R1-1, parte array):
/// cada função do conjunto padrão é avaliada elemento a elemento com entradas de borda nomeadas (comprimentos
/// válidos iguais) e comparada entre ReferenceGenomeEvaluator, CompiledGenomeEvaluator (instância distinta) e
/// IlgpuGenomeEvaluator em cada linha de <see cref="IlgpuDeviceMatrixTests.Devices"/>, exigindo execução real de
/// kernel. Arrays vazios e comprimentos incompatíveis ficam em testes separados que afirmam o fallback e a causa,
/// sem contar como cobertura de dispositivo.
/// </summary>
public sealed class IlgpuArrayFunctionEdgeMatrixTests
{
    public static IEnumerable<object[]> DoubleArrayEdges() =>
    [
        [new[] { 1e-12, -1e-12, 0.0 }, new[] { -1e-12, 1e-12, 0.0 }],
        [new[] { double.NaN, 1.0, -1.0 }, new[] { 1.0, double.NaN, 2.0 }],
        [new[] { double.NaN, double.NaN, 0.0 }, new[] { double.NaN, double.NaN, 1.0 }],
        [new[] { double.PositiveInfinity, 1.0, -2.0 }, new[] { 1.0, double.NegativeInfinity, 3.0 }],
        [new[] { double.PositiveInfinity, double.NegativeInfinity, 0.0 }, new[] { double.NegativeInfinity, double.PositiveInfinity, 0.0 }],
        [new[] { 0.0, -0.0, 2.5 }, new[] { -0.0, 0.0, -1.25 }],
        [new[] { 1e-300, 2.0, -3.5 }, new[] { 2.0, 1e-300, 0.5 }],
        [new[] { 7.0 }, new[] { -2.5 }],
        [new[] { 1.5, -2.0, 3.0, double.NaN, 0.0 }, new[] { -1.0, 4.0, double.PositiveInfinity, 2.0, -0.0 }],
    ];

    public static IEnumerable<object[]> IntArrayEdges() =>
    [
        [new[] { int.MaxValue, 1, 0 }, new[] { 1, int.MaxValue, 0 }],
        [new[] { int.MinValue, -1, 7 }, new[] { -1, int.MinValue, -4 }],
        [new[] { int.MinValue, 0, int.MaxValue }, new[] { 0, int.MinValue, int.MaxValue }],
        [new[] { 0, 0, 0 }, new[] { 0, 0, 0 }],
        [new[] { 7, -4, 1 }, new[] { -3, 7, -1 }],
        [new[] { 5 }, new[] { -8 }],
        [new[] { 1, -2, 3, int.MinValue, 0 }, new[] { -1, 4, int.MaxValue, 2, -7 }],
    ];

    [Theory]
    [MemberData(nameof(IlgpuDeviceMatrixTests.Devices), MemberType = typeof(IlgpuDeviceMatrixTests))]
    public void Double_array_function_edges_match_reference_and_compiled_on_device(string label, IlgpuBackendSelection selection)
    {
        var functions = StandardDoubleArrayFunctionSet.Instance;
        var exercised = 0;
        foreach (var row in DoubleArrayEdges())
        {
            var inputs = new[] { (double[])row[0], (double[])row[1] };
            for (var f = 0; f < functions.Count; f++)
            {
                var genome = SingleNodeArrayGenome(functions, f, (double[])row[0], DoubleArrayValueSemantics.Instance);
                var expected = new double[1][];
                var compiled = new double[1][];
                var device = new double[1][];
                new ReferenceGenomeEvaluator<double[]>(functions).Evaluate(genome, inputs, expected);
                new CompiledGenomeEvaluator<double[]>(functions, DoubleArrayValueSemantics.Instance).Evaluate(genome, inputs, compiled);
                using var evaluator = new IlgpuGenomeEvaluator<double[]>(functions, null, selection, null, null, null);
                evaluator.Evaluate(genome, inputs, device);
                var tag = $"{label} {functions.Get(f).Id} {Describe(inputs[0])}/{Describe(inputs[1])}";
                AssertDeviceKernel(evaluator, genome, functions, selection, tag);
                IlgpuTestHelpers.AssertDoubleArrayClose(expected[0], compiled[0], $"{tag} compiled");
                IlgpuTestHelpers.AssertDoubleArrayClose(expected[0], device[0], $"{tag} device");
                exercised++;
            }
        }

        Assert.Equal(functions.Count * DoubleArrayEdges().Count(), exercised);
    }

    [Theory]
    [MemberData(nameof(IlgpuDeviceMatrixTests.Devices), MemberType = typeof(IlgpuDeviceMatrixTests))]
    public void Int_array_function_edges_match_reference_and_compiled_on_device(string label, IlgpuBackendSelection selection)
    {
        var functions = StandardInt32ArrayFunctionSet.Instance;
        var exercised = 0;
        foreach (var row in IntArrayEdges())
        {
            var inputs = new[] { (int[])row[0], (int[])row[1] };
            for (var f = 0; f < functions.Count; f++)
            {
                var genome = SingleNodeArrayGenome(functions, f, (int[])row[0], Int32ArrayValueSemantics.Instance);
                var expected = new int[1][];
                var compiled = new int[1][];
                var device = new int[1][];
                new ReferenceGenomeEvaluator<int[]>(functions).Evaluate(genome, inputs, expected);
                new CompiledGenomeEvaluator<int[]>(functions, Int32ArrayValueSemantics.Instance).Evaluate(genome, inputs, compiled);
                using var evaluator = new IlgpuGenomeEvaluator<int[]>(functions, null, selection, null, null, null);
                evaluator.Evaluate(genome, inputs, device);
                var tag = $"{label} {functions.Get(f).Id} {Describe(inputs[0])}/{Describe(inputs[1])}";
                AssertDeviceKernel(evaluator, genome, functions, selection, tag);
                Assert.Equal(expected[0], compiled[0]);
                Assert.Equal(expected[0], device[0]);
                exercised++;
            }
        }

        Assert.Equal(functions.Count * IntArrayEdges().Count(), exercised);
    }

    [Theory]
    [MemberData(nameof(IlgpuDeviceMatrixTests.Devices), MemberType = typeof(IlgpuDeviceMatrixTests))]
    public void Empty_arrays_fall_back_to_cpu_without_counting_as_device_coverage(string label, IlgpuBackendSelection selection)
    {
        _ = label;
        var functions = StandardDoubleArrayFunctionSet.Instance;
        var addId = IlgpuTestHelpers.FindFunctionId(functions, "add");
        var genome = SingleNodeArrayGenome(functions, addId, Array.Empty<double>(), DoubleArrayValueSemantics.Instance);
        var inputs = new[] { Array.Empty<double>(), Array.Empty<double>() };
        var expected = new double[1][];
        var compiled = new double[1][];
        var device = new double[1][];
        new ReferenceGenomeEvaluator<double[]>(functions).Evaluate(genome, inputs, expected);
        new CompiledGenomeEvaluator<double[]>(functions, DoubleArrayValueSemantics.Instance).Evaluate(genome, inputs, compiled);
        using var evaluator = new IlgpuGenomeEvaluator<double[]>(functions, null, selection, null, null, null);
        evaluator.Evaluate(genome, inputs, device);

        Assert.Empty(expected[0]);
        Assert.Empty(compiled[0]);
        Assert.Empty(device[0]);
        Assert.False(evaluator.LastEvaluationUsedDevice);
        Assert.Equal("CPU fallback", evaluator.Backend);
        Assert.Equal("nenhum", evaluator.DeviceName);
        Assert.Equal("compiled-default", evaluator.FallbackEvaluatorKind);
        Assert.Contains("vazi", evaluator.LastFallbackReason ?? string.Empty, StringComparison.OrdinalIgnoreCase);
    }

    [Theory]
    [MemberData(nameof(IlgpuDeviceMatrixTests.Devices), MemberType = typeof(IlgpuDeviceMatrixTests))]
    public void Empty_int_arrays_fall_back_to_cpu_without_counting_as_device_coverage(string label, IlgpuBackendSelection selection)
    {
        // Variação int[] (R13-2) do caso "vazio válido" acima, que só cobria double[].
        _ = label;
        var functions = StandardInt32ArrayFunctionSet.Instance;
        var addId = IlgpuTestHelpers.FindFunctionId(functions, "add");
        var genome = SingleNodeArrayGenome(functions, addId, Array.Empty<int>(), Int32ArrayValueSemantics.Instance);
        var inputs = new[] { Array.Empty<int>(), Array.Empty<int>() };
        var expected = new int[1][];
        var compiled = new int[1][];
        var device = new int[1][];
        new ReferenceGenomeEvaluator<int[]>(functions).Evaluate(genome, inputs, expected);
        new CompiledGenomeEvaluator<int[]>(functions, Int32ArrayValueSemantics.Instance).Evaluate(genome, inputs, compiled);
        using var evaluator = new IlgpuGenomeEvaluator<int[]>(functions, null, selection, null, null, null);
        evaluator.Evaluate(genome, inputs, device);

        Assert.Empty(expected[0]);
        Assert.Empty(compiled[0]);
        Assert.Empty(device[0]);
        Assert.False(evaluator.LastEvaluationUsedDevice);
        Assert.Equal("CPU fallback", evaluator.Backend);
        Assert.Equal("nenhum", evaluator.DeviceName);
        Assert.Equal("compiled-default", evaluator.FallbackEvaluatorKind);
        Assert.Contains("vazi", evaluator.LastFallbackReason ?? string.Empty, StringComparison.OrdinalIgnoreCase);
    }

    [Theory]
    [MemberData(nameof(IlgpuDeviceMatrixTests.Devices), MemberType = typeof(IlgpuDeviceMatrixTests))]
    public void Mismatched_array_lengths_fall_back_and_match_reference_failure_without_counting_as_device_coverage(string label, IlgpuBackendSelection selection)
    {
        _ = label;
        var functions = StandardDoubleArrayFunctionSet.Instance;
        var addId = IlgpuTestHelpers.FindFunctionId(functions, "add");
        var genome = SingleNodeArrayGenome(functions, addId, [1.0, 2.0], DoubleArrayValueSemantics.Instance);
        var inputs = new[] { new[] { 1.0, 2.0 }, new[] { 1.0, 2.0, 3.0 } };
        var output = new double[1][];
        Assert.Throws<ArgumentException>(() => new ReferenceGenomeEvaluator<double[]>(functions).Evaluate(genome, inputs, output));
        Assert.Throws<ArgumentException>(() => new CompiledGenomeEvaluator<double[]>(functions, DoubleArrayValueSemantics.Instance).Evaluate(genome, inputs, output));
        using var evaluator = new IlgpuGenomeEvaluator<double[]>(functions, null, selection, null, null, null);
        Assert.Throws<ArgumentException>(() => evaluator.Evaluate(genome, inputs, output));
        Assert.False(evaluator.LastEvaluationUsedDevice);
        Assert.Equal("CPU fallback", evaluator.Backend);
        Assert.Equal("nenhum", evaluator.DeviceName);
        Assert.Equal("compiled-default", evaluator.FallbackEvaluatorKind);
        Assert.False(string.IsNullOrWhiteSpace(evaluator.LastFallbackReason), "causa de fallback ausente");
    }

    /// <summary>Genoma mínimo: um nó com a função <paramref name="functionId"/> sobre Input0/Input1 e uma saída.</summary>
    private static Genome<T> SingleNodeArrayGenome<T>(IFunctionSet<T> functions, int functionId, T constant, IValueSemantics<T> semantics)
    {
        var shape = new GenomeShape(InputCount: 2, NodeCount: 1, OutputCount: 1, MaxArity: 3, LevelsBack: 1);
        return Genome<T>.FromGenes(shape, [functionId], [0, 1, 1], [constant], [2], semantics, functions);
    }

    /// <summary>Genoma mínimo com conexões e número de inputs explícitos (R7-1/R7-2): casos discriminantes de
    /// Select (três ramos distintos), ProtectedDivide perto do EPS e comprimentos incompatíveis em Select/Constant.</summary>
    private static Genome<T> SingleNodeArrayGenomeWithInputs<T>(IFunctionSet<T> functions, int functionId, int inputCount, int[] connections, T constant, IValueSemantics<T> semantics)
    {
        var shape = new GenomeShape(InputCount: inputCount, NodeCount: 1, OutputCount: 1, MaxArity: connections.Length, LevelsBack: 1);
        return Genome<T>.FromGenes(shape, [functionId], connections, [constant], [inputCount], semantics, functions);
    }

    [Theory]
    [MemberData(nameof(IlgpuDeviceMatrixTests.Devices), MemberType = typeof(IlgpuDeviceMatrixTests))]
    public void Select_double_array_discriminates_three_distinct_branches_on_device(string label, IlgpuBackendSelection selection)
    {
        var functions = StandardDoubleArrayFunctionSet.Instance;
        var selectId = IlgpuTestHelpers.FindFunctionId(functions, "select");
        var genome = SingleNodeArrayGenomeWithInputs(functions, selectId, 3, [0, 1, 2], Array.Empty<double>(), DoubleArrayValueSemantics.Instance);
        var aBranch = new[] { 10.0, 20.0, 30.0 };
        var bBranch = new[] { -10.0, -20.0, -30.0 };
        var cases = new (double[] c, double[] expectedBranch)[]
        {
            ([5.0, 5.0, 5.0], aBranch),
            ([0.0, 0.0, 0.0], bBranch),
            ([-3.0, -3.0, -3.0], bBranch),
            ([double.NaN, double.NaN, double.NaN], bBranch),
        };
        foreach (var (c, expectedBranch) in cases)
        {
            var inputs = new[] { c, aBranch, bBranch };
            var expected = new double[1][];
            var compiled = new double[1][];
            var device = new double[1][];
            new ReferenceGenomeEvaluator<double[]>(functions).Evaluate(genome, inputs, expected);
            new CompiledGenomeEvaluator<double[]>(functions, DoubleArrayValueSemantics.Instance).Evaluate(genome, inputs, compiled);
            using var evaluator = new IlgpuGenomeEvaluator<double[]>(functions, null, selection, null, null, null);
            evaluator.Evaluate(genome, inputs, device);
            var tag = $"{label} select array c={Describe(c)}";
            AssertDeviceKernel(evaluator, genome, functions, selection, tag);
            Assert.Equal(expectedBranch, expected[0]);
            IlgpuTestHelpers.AssertDoubleArrayClose(expected[0], compiled[0], $"{tag} compiled");
            IlgpuTestHelpers.AssertDoubleArrayClose(expected[0], device[0], $"{tag} device");
        }
    }

    [Theory]
    [MemberData(nameof(IlgpuDeviceMatrixTests.Devices), MemberType = typeof(IlgpuDeviceMatrixTests))]
    public void Select_int_array_discriminates_three_distinct_branches_on_device(string label, IlgpuBackendSelection selection)
    {
        var functions = StandardInt32ArrayFunctionSet.Instance;
        var selectId = IlgpuTestHelpers.FindFunctionId(functions, "select");
        var genome = SingleNodeArrayGenomeWithInputs(functions, selectId, 3, [0, 1, 2], Array.Empty<int>(), Int32ArrayValueSemantics.Instance);
        var aBranch = new[] { 100, 200, 300 };
        var bBranch = new[] { -100, -200, -300 };
        var cases = new (int[] c, int[] expectedBranch)[]
        {
            ([5, 5, 5], aBranch),
            ([0, 0, 0], bBranch),
            ([-3, -3, -3], bBranch),
        };
        foreach (var (c, expectedBranch) in cases)
        {
            var inputs = new[] { c, aBranch, bBranch };
            var expected = new int[1][];
            var compiled = new int[1][];
            var device = new int[1][];
            new ReferenceGenomeEvaluator<int[]>(functions).Evaluate(genome, inputs, expected);
            new CompiledGenomeEvaluator<int[]>(functions, Int32ArrayValueSemantics.Instance).Evaluate(genome, inputs, compiled);
            using var evaluator = new IlgpuGenomeEvaluator<int[]>(functions, null, selection, null, null, null);
            evaluator.Evaluate(genome, inputs, device);
            var tag = $"{label} select array c={Describe(c)}";
            AssertDeviceKernel(evaluator, genome, functions, selection, tag);
            Assert.Equal(expectedBranch, expected[0]);
            Assert.Equal(expected[0], compiled[0]);
            Assert.Equal(expected[0], device[0]);
        }
    }

    [Theory]
    [MemberData(nameof(IlgpuDeviceMatrixTests.Devices), MemberType = typeof(IlgpuDeviceMatrixTests))]
    public void ProtectedDivide_double_array_near_eps_boundary_on_device(string label, IlgpuBackendSelection selection)
    {
        const double Eps = 1e-12;
        var functions = StandardDoubleArrayFunctionSet.Instance;
        var divId = IlgpuTestHelpers.FindFunctionId(functions, "protecteddivide");
        var genome = SingleNodeArrayGenomeWithInputs(functions, divId, 2, [0, 1], Array.Empty<double>(), DoubleArrayValueSemantics.Instance);
        var numerator = new[] { 3.0, 3.0, 3.0 };
        var denominators = new[]
        {
            new[] { Eps * 0.5, -Eps * 0.5, Eps },
            new[] { -Eps, Eps * 1.5, -Eps * 1.5 },
            new[] { Math.BitDecrement(Eps), Math.BitIncrement(Eps), Math.BitDecrement(-Eps) },
            new[] { Math.BitIncrement(-Eps), Eps, -Eps },
        };
        foreach (var denom in denominators)
        {
            var inputs = new[] { numerator, denom };
            var expected = new double[1][];
            var compiled = new double[1][];
            var device = new double[1][];
            new ReferenceGenomeEvaluator<double[]>(functions).Evaluate(genome, inputs, expected);
            new CompiledGenomeEvaluator<double[]>(functions, DoubleArrayValueSemantics.Instance).Evaluate(genome, inputs, compiled);
            using var evaluator = new IlgpuGenomeEvaluator<double[]>(functions, null, selection, null, null, null);
            evaluator.Evaluate(genome, inputs, device);
            var tag = $"{label} protecteddivide array denom={Describe(denom)}";
            AssertDeviceKernel(evaluator, genome, functions, selection, tag);
            IlgpuTestHelpers.AssertDoubleArrayClose(expected[0], compiled[0], $"{tag} compiled");
            IlgpuTestHelpers.AssertDoubleArrayClose(expected[0], device[0], $"{tag} device");
        }
    }

    [Theory]
    [MemberData(nameof(IlgpuDeviceMatrixTests.Devices), MemberType = typeof(IlgpuDeviceMatrixTests))]
    public void Mismatched_array_lengths_int_fall_back_and_match_reference_failure_without_counting_as_device_coverage(string label, IlgpuBackendSelection selection)
    {
        _ = label;
        var functions = StandardInt32ArrayFunctionSet.Instance;
        var addId = IlgpuTestHelpers.FindFunctionId(functions, "add");
        var genome = SingleNodeArrayGenome(functions, addId, [1, 2], Int32ArrayValueSemantics.Instance);
        var inputs = new[] { new[] { 1, 2 }, new[] { 1, 2, 3 } };
        var output = new int[1][];
        Assert.Throws<ArgumentException>(() => new ReferenceGenomeEvaluator<int[]>(functions).Evaluate(genome, inputs, output));
        Assert.Throws<ArgumentException>(() => new CompiledGenomeEvaluator<int[]>(functions, Int32ArrayValueSemantics.Instance).Evaluate(genome, inputs, output));
        using var evaluator = new IlgpuGenomeEvaluator<int[]>(functions, null, selection, null, null, null);
        Assert.Throws<ArgumentException>(() => evaluator.Evaluate(genome, inputs, output));
        Assert.False(evaluator.LastEvaluationUsedDevice);
        Assert.Equal("CPU fallback", evaluator.Backend);
        Assert.Equal("nenhum", evaluator.DeviceName);
        Assert.Equal("compiled-default", evaluator.FallbackEvaluatorKind);
        Assert.False(string.IsNullOrWhiteSpace(evaluator.LastFallbackReason), "causa de fallback ausente");
    }

    [Theory]
    [MemberData(nameof(IlgpuDeviceMatrixTests.Devices), MemberType = typeof(IlgpuDeviceMatrixTests))]
    public void Select_array_mismatched_argument_lengths_fall_back_and_match_reference_failure_without_counting_as_device_coverage(string label, IlgpuBackendSelection selection)
    {
        _ = label;
        var functions = StandardDoubleArrayFunctionSet.Instance;
        var selectId = IlgpuTestHelpers.FindFunctionId(functions, "select");
        var genome = SingleNodeArrayGenomeWithInputs(functions, selectId, 3, [0, 1, 2], Array.Empty<double>(), DoubleArrayValueSemantics.Instance);
        // Condição e ramo "a" com comprimento 3, ramo "b" com comprimento 2 (incompatível).
        var inputs = new[] { new[] { 1.0, 1.0, 1.0 }, new[] { 10.0, 20.0, 30.0 }, new[] { -10.0, -20.0 } };
        var output = new double[1][];
        Assert.Throws<ArgumentException>(() => new ReferenceGenomeEvaluator<double[]>(functions).Evaluate(genome, inputs, output));
        Assert.Throws<ArgumentException>(() => new CompiledGenomeEvaluator<double[]>(functions, DoubleArrayValueSemantics.Instance).Evaluate(genome, inputs, output));
        using var evaluator = new IlgpuGenomeEvaluator<double[]>(functions, null, selection, null, null, null);
        Assert.Throws<ArgumentException>(() => evaluator.Evaluate(genome, inputs, output));
        Assert.False(evaluator.LastEvaluationUsedDevice);
        Assert.Equal("CPU fallback", evaluator.Backend);
        Assert.Equal("nenhum", evaluator.DeviceName);
        Assert.Equal("compiled-default", evaluator.FallbackEvaluatorKind);
        Assert.False(string.IsNullOrWhiteSpace(evaluator.LastFallbackReason), "causa de fallback ausente");
    }

    [Theory]
    [MemberData(nameof(IlgpuDeviceMatrixTests.Devices), MemberType = typeof(IlgpuDeviceMatrixTests))]
    public void Select_int_array_mismatched_argument_lengths_fall_back_and_match_reference_failure_without_counting_as_device_coverage(string label, IlgpuBackendSelection selection)
    {
        // Variação int[] (R13-2) do ramo "b" divergente acima, que só cobria double[].
        _ = label;
        var functions = StandardInt32ArrayFunctionSet.Instance;
        var selectId = IlgpuTestHelpers.FindFunctionId(functions, "select");
        var genome = SingleNodeArrayGenomeWithInputs(functions, selectId, 3, [0, 1, 2], Array.Empty<int>(), Int32ArrayValueSemantics.Instance);
        var inputs = new[] { new[] { 1, 1, 1 }, new[] { 10, 20, 30 }, new[] { -10, -20 } };
        var output = new int[1][];
        Assert.Throws<ArgumentException>(() => new ReferenceGenomeEvaluator<int[]>(functions).Evaluate(genome, inputs, output));
        Assert.Throws<ArgumentException>(() => new CompiledGenomeEvaluator<int[]>(functions, Int32ArrayValueSemantics.Instance).Evaluate(genome, inputs, output));
        using var evaluator = new IlgpuGenomeEvaluator<int[]>(functions, null, selection, null, null, null);
        Assert.Throws<ArgumentException>(() => evaluator.Evaluate(genome, inputs, output));
        Assert.False(evaluator.LastEvaluationUsedDevice);
        Assert.Equal("CPU fallback", evaluator.Backend);
        Assert.Equal("nenhum", evaluator.DeviceName);
        Assert.Equal("compiled-default", evaluator.FallbackEvaluatorKind);
        Assert.False(string.IsNullOrWhiteSpace(evaluator.LastFallbackReason), "causa de fallback ausente");
    }

    [Theory]
    [MemberData(nameof(IlgpuDeviceMatrixTests.Devices), MemberType = typeof(IlgpuDeviceMatrixTests))]
    public void Select_array_mismatched_condition_length_falls_back_and_matches_reference_failure(string label, IlgpuBackendSelection selection)
    {
        // Variação complementar (R9-3): aqui é a condição "c" que diverge (comprimento 2), enquanto os dois ramos
        // "a"/"b" têm comprimento compatível (3) entre si — argumento diferente do caso a/b já coberto acima.
        _ = label;
        var functions = StandardDoubleArrayFunctionSet.Instance;
        var selectId = IlgpuTestHelpers.FindFunctionId(functions, "select");
        var genome = SingleNodeArrayGenomeWithInputs(functions, selectId, 3, [0, 1, 2], Array.Empty<double>(), DoubleArrayValueSemantics.Instance);
        var inputs = new[] { new[] { 1.0, 1.0 }, new[] { 10.0, 20.0, 30.0 }, new[] { -10.0, -20.0, -30.0 } };
        var output = new double[1][];
        Assert.Throws<ArgumentException>(() => new ReferenceGenomeEvaluator<double[]>(functions).Evaluate(genome, inputs, output));
        Assert.Throws<ArgumentException>(() => new CompiledGenomeEvaluator<double[]>(functions, DoubleArrayValueSemantics.Instance).Evaluate(genome, inputs, output));
        using var evaluator = new IlgpuGenomeEvaluator<double[]>(functions, null, selection, null, null, null);
        Assert.Throws<ArgumentException>(() => evaluator.Evaluate(genome, inputs, output));
        Assert.False(evaluator.LastEvaluationUsedDevice);
        Assert.Equal("CPU fallback", evaluator.Backend);
        Assert.Equal("nenhum", evaluator.DeviceName);
        Assert.Equal("compiled-default", evaluator.FallbackEvaluatorKind);
        Assert.False(string.IsNullOrWhiteSpace(evaluator.LastFallbackReason), "causa de fallback ausente");
    }

    [Theory]
    [MemberData(nameof(IlgpuDeviceMatrixTests.Devices), MemberType = typeof(IlgpuDeviceMatrixTests))]
    public void Select_int_array_mismatched_condition_length_falls_back_and_matches_reference_failure(string label, IlgpuBackendSelection selection)
    {
        // Variação int[] (R13-2) da condição divergente acima, que só cobria double[].
        _ = label;
        var functions = StandardInt32ArrayFunctionSet.Instance;
        var selectId = IlgpuTestHelpers.FindFunctionId(functions, "select");
        var genome = SingleNodeArrayGenomeWithInputs(functions, selectId, 3, [0, 1, 2], Array.Empty<int>(), Int32ArrayValueSemantics.Instance);
        var inputs = new[] { new[] { 1, 1 }, new[] { 10, 20, 30 }, new[] { -10, -20, -30 } };
        var output = new int[1][];
        Assert.Throws<ArgumentException>(() => new ReferenceGenomeEvaluator<int[]>(functions).Evaluate(genome, inputs, output));
        Assert.Throws<ArgumentException>(() => new CompiledGenomeEvaluator<int[]>(functions, Int32ArrayValueSemantics.Instance).Evaluate(genome, inputs, output));
        using var evaluator = new IlgpuGenomeEvaluator<int[]>(functions, null, selection, null, null, null);
        Assert.Throws<ArgumentException>(() => evaluator.Evaluate(genome, inputs, output));
        Assert.False(evaluator.LastEvaluationUsedDevice);
        Assert.Equal("CPU fallback", evaluator.Backend);
        Assert.Equal("nenhum", evaluator.DeviceName);
        Assert.Equal("compiled-default", evaluator.FallbackEvaluatorKind);
        Assert.False(string.IsNullOrWhiteSpace(evaluator.LastFallbackReason), "causa de fallback ausente");
    }

    [Theory]
    [MemberData(nameof(IlgpuDeviceMatrixTests.Devices), MemberType = typeof(IlgpuDeviceMatrixTests))]
    public void Select_array_mismatched_branch_a_length_falls_back_and_matches_reference_failure(string label, IlgpuBackendSelection selection)
    {
        // Variação discriminante (R15-1): aqui é apenas o ramo "a" (comprimento 2) que diverge da condição e do
        // ramo "b" (comprimento 3 ambos), diferente dos casos já cobertos de "b" e de condição divergentes.
        _ = label;
        var functions = StandardDoubleArrayFunctionSet.Instance;
        var selectId = IlgpuTestHelpers.FindFunctionId(functions, "select");
        var genome = SingleNodeArrayGenomeWithInputs(functions, selectId, 3, [0, 1, 2], Array.Empty<double>(), DoubleArrayValueSemantics.Instance);
        var inputs = new[] { new[] { 1.0, 1.0, 1.0 }, new[] { 10.0, 20.0 }, new[] { -10.0, -20.0, -30.0 } };
        var output = new double[1][];
        Assert.Throws<ArgumentException>(() => new ReferenceGenomeEvaluator<double[]>(functions).Evaluate(genome, inputs, output));
        Assert.Throws<ArgumentException>(() => new CompiledGenomeEvaluator<double[]>(functions, DoubleArrayValueSemantics.Instance).Evaluate(genome, inputs, output));
        using var evaluator = new IlgpuGenomeEvaluator<double[]>(functions, null, selection, null, null, null);
        Assert.Throws<ArgumentException>(() => evaluator.Evaluate(genome, inputs, output));
        Assert.False(evaluator.LastEvaluationUsedDevice);
        Assert.Equal("CPU fallback", evaluator.Backend);
        Assert.Equal("nenhum", evaluator.DeviceName);
        Assert.Equal("compiled-default", evaluator.FallbackEvaluatorKind);
        Assert.False(string.IsNullOrWhiteSpace(evaluator.LastFallbackReason), "causa de fallback ausente");
    }

    [Theory]
    [MemberData(nameof(IlgpuDeviceMatrixTests.Devices), MemberType = typeof(IlgpuDeviceMatrixTests))]
    public void Select_int_array_mismatched_branch_a_length_falls_back_and_matches_reference_failure(string label, IlgpuBackendSelection selection)
    {
        // Variação int[] (R15-1) do ramo "a" divergente acima, que só cobria double[].
        _ = label;
        var functions = StandardInt32ArrayFunctionSet.Instance;
        var selectId = IlgpuTestHelpers.FindFunctionId(functions, "select");
        var genome = SingleNodeArrayGenomeWithInputs(functions, selectId, 3, [0, 1, 2], Array.Empty<int>(), Int32ArrayValueSemantics.Instance);
        var inputs = new[] { new[] { 1, 1, 1 }, new[] { 10, 20 }, new[] { -10, -20, -30 } };
        var output = new int[1][];
        Assert.Throws<ArgumentException>(() => new ReferenceGenomeEvaluator<int[]>(functions).Evaluate(genome, inputs, output));
        Assert.Throws<ArgumentException>(() => new CompiledGenomeEvaluator<int[]>(functions, Int32ArrayValueSemantics.Instance).Evaluate(genome, inputs, output));
        using var evaluator = new IlgpuGenomeEvaluator<int[]>(functions, null, selection, null, null, null);
        Assert.Throws<ArgumentException>(() => evaluator.Evaluate(genome, inputs, output));
        Assert.False(evaluator.LastEvaluationUsedDevice);
        Assert.Equal("CPU fallback", evaluator.Backend);
        Assert.Equal("nenhum", evaluator.DeviceName);
        Assert.Equal("compiled-default", evaluator.FallbackEvaluatorKind);
        Assert.False(string.IsNullOrWhiteSpace(evaluator.LastFallbackReason), "causa de fallback ausente");
    }

    [Theory]
    [MemberData(nameof(IlgpuDeviceMatrixTests.Devices), MemberType = typeof(IlgpuDeviceMatrixTests))]
    public void Constant_array_length_differs_from_sibling_input_is_valid_as_direct_output_on_device(string label, IlgpuBackendSelection selection)
    {
        // Genoma de um nó Constant (arity 0, R9-3): o comprimento do parâmetro (5) difere do comprimento das
        // entradas (3), mas como a saída é o próprio nó Constant (sem combiná-lo com a entrada), é uma saída
        // válida, não uma incompatibilidade — diferente do caso de operando incompatível acima.
        _ = label;
        var functions = StandardDoubleArrayFunctionSet.Instance;
        var constId = IlgpuTestHelpers.FindFunctionId(functions, "constant");
        var parameter = new[] { 1.0, -2.0, double.NaN, double.PositiveInfinity, 0.0 };
        var genome = SingleNodeArrayGenome(functions, constId, parameter, DoubleArrayValueSemantics.Instance);
        var inputs = new[] { new[] { 1.0, 2.0, 3.0 }, new[] { 4.0, 5.0, 6.0 } };
        var expected = new double[1][];
        var compiled = new double[1][];
        var device = new double[1][];
        new ReferenceGenomeEvaluator<double[]>(functions).Evaluate(genome, inputs, expected);
        new CompiledGenomeEvaluator<double[]>(functions, DoubleArrayValueSemantics.Instance).Evaluate(genome, inputs, compiled);
        using var evaluator = new IlgpuGenomeEvaluator<double[]>(functions, null, selection, null, null, null);
        evaluator.Evaluate(genome, inputs, device);
        // Comprimento divergente de arrays no mesmo genoma não é suportado pelo kernel ILGPU mesmo quando o nó
        // Constant em si ignora a entrada (arity 0): cai no fallback compiled-default legítimo, sem contar como
        // cobertura de dispositivo, mas a saída continua válida e igual à referência.
        IlgpuTestHelpers.AssertDoubleArrayClose(expected[0], compiled[0], "compiled");
        IlgpuTestHelpers.AssertDoubleArrayClose(expected[0], device[0], "device");
        Assert.False(evaluator.LastEvaluationUsedDevice);
        Assert.Equal("nenhum", evaluator.DeviceName);
        Assert.Equal("compiled-default", evaluator.FallbackEvaluatorKind);
        Assert.False(string.IsNullOrWhiteSpace(evaluator.LastFallbackReason), "causa de fallback ausente");
    }

    [Theory]
    [MemberData(nameof(IlgpuDeviceMatrixTests.Devices), MemberType = typeof(IlgpuDeviceMatrixTests))]
    public void Constant_int_array_length_differs_from_sibling_input_is_valid_as_direct_output_on_device(string label, IlgpuBackendSelection selection)
    {
        _ = label;
        var functions = StandardInt32ArrayFunctionSet.Instance;
        var constId = IlgpuTestHelpers.FindFunctionId(functions, "constant");
        var parameter = new[] { 1, -2, int.MinValue, int.MaxValue, 0 };
        var genome = SingleNodeArrayGenome(functions, constId, parameter, Int32ArrayValueSemantics.Instance);
        var inputs = new[] { new[] { 1, 2, 3 }, new[] { 4, 5, 6 } };
        var expected = new int[1][];
        var compiled = new int[1][];
        var device = new int[1][];
        new ReferenceGenomeEvaluator<int[]>(functions).Evaluate(genome, inputs, expected);
        new CompiledGenomeEvaluator<int[]>(functions, Int32ArrayValueSemantics.Instance).Evaluate(genome, inputs, compiled);
        using var evaluator = new IlgpuGenomeEvaluator<int[]>(functions, null, selection, null, null, null);
        evaluator.Evaluate(genome, inputs, device);
        Assert.Equal(expected[0], compiled[0]);
        Assert.Equal(expected[0], device[0]);
        Assert.False(evaluator.LastEvaluationUsedDevice);
        Assert.Equal("nenhum", evaluator.DeviceName);
        Assert.Equal("compiled-default", evaluator.FallbackEvaluatorKind);
        Assert.False(string.IsNullOrWhiteSpace(evaluator.LastFallbackReason), "causa de fallback ausente");
    }

    [Theory]
    [MemberData(nameof(IlgpuDeviceMatrixTests.Devices), MemberType = typeof(IlgpuDeviceMatrixTests))]
    public void Constant_array_length_mismatch_with_sibling_input_falls_back_and_matches_reference_failure(string label, IlgpuBackendSelection selection)
    {
        // Genoma de dois nós: node0=Constant(comprimento 2), node1=Add(node0, input0) com input0 de comprimento 3:
        // o comprimento do parâmetro Constant difere do comprimento da entrada usada no mesmo cálculo.
        _ = label;
        var functions = StandardDoubleArrayFunctionSet.Instance;
        var constId = IlgpuTestHelpers.FindFunctionId(functions, "constant");
        var addId = IlgpuTestHelpers.FindFunctionId(functions, "add");
        var shape = new GenomeShape(InputCount: 1, NodeCount: 2, OutputCount: 1, MaxArity: 2, LevelsBack: 3);
        var genome = Genome<double[]>.FromGenes(
            shape,
            [constId, addId],
            [0, 0, 0, 1],
            [new[] { 1.0, 2.0 }, Array.Empty<double>()],
            [2],
            DoubleArrayValueSemantics.Instance,
            functions);
        var inputs = new[] { new[] { 1.0, 2.0, 3.0 } };
        var output = new double[1][];
        Assert.Throws<ArgumentException>(() => new ReferenceGenomeEvaluator<double[]>(functions).Evaluate(genome, inputs, output));
        Assert.Throws<ArgumentException>(() => new CompiledGenomeEvaluator<double[]>(functions, DoubleArrayValueSemantics.Instance).Evaluate(genome, inputs, output));
        using var evaluator = new IlgpuGenomeEvaluator<double[]>(functions, null, selection, null, null, null);
        Assert.Throws<ArgumentException>(() => evaluator.Evaluate(genome, inputs, output));
        Assert.False(evaluator.LastEvaluationUsedDevice);
        Assert.Equal("CPU fallback", evaluator.Backend);
        Assert.Equal("nenhum", evaluator.DeviceName);
        Assert.Equal("compiled-default", evaluator.FallbackEvaluatorKind);
        Assert.False(string.IsNullOrWhiteSpace(evaluator.LastFallbackReason), "causa de fallback ausente");
    }

    [Theory]
    [MemberData(nameof(IlgpuDeviceMatrixTests.Devices), MemberType = typeof(IlgpuDeviceMatrixTests))]
    public void Constant_int_array_length_mismatch_with_sibling_input_falls_back_and_matches_reference_failure(string label, IlgpuBackendSelection selection)
    {
        // Variação int[] (R13-2) do Constant usado como operando incompatível acima, que só cobria double[].
        _ = label;
        var functions = StandardInt32ArrayFunctionSet.Instance;
        var constId = IlgpuTestHelpers.FindFunctionId(functions, "constant");
        var addId = IlgpuTestHelpers.FindFunctionId(functions, "add");
        var shape = new GenomeShape(InputCount: 1, NodeCount: 2, OutputCount: 1, MaxArity: 2, LevelsBack: 3);
        var genome = Genome<int[]>.FromGenes(
            shape,
            [constId, addId],
            [0, 0, 0, 1],
            [new[] { 1, 2 }, Array.Empty<int>()],
            [2],
            Int32ArrayValueSemantics.Instance,
            functions);
        var inputs = new[] { new[] { 1, 2, 3 } };
        var output = new int[1][];
        Assert.Throws<ArgumentException>(() => new ReferenceGenomeEvaluator<int[]>(functions).Evaluate(genome, inputs, output));
        Assert.Throws<ArgumentException>(() => new CompiledGenomeEvaluator<int[]>(functions, Int32ArrayValueSemantics.Instance).Evaluate(genome, inputs, output));
        using var evaluator = new IlgpuGenomeEvaluator<int[]>(functions, null, selection, null, null, null);
        Assert.Throws<ArgumentException>(() => evaluator.Evaluate(genome, inputs, output));
        Assert.False(evaluator.LastEvaluationUsedDevice);
        Assert.Equal("CPU fallback", evaluator.Backend);
        Assert.Equal("nenhum", evaluator.DeviceName);
        Assert.Equal("compiled-default", evaluator.FallbackEvaluatorKind);
        Assert.False(string.IsNullOrWhiteSpace(evaluator.LastFallbackReason), "causa de fallback ausente");
    }

    private static void AssertDeviceKernel<T>(IlgpuGenomeEvaluator<T> evaluator, Genome<T> genome, IFunctionSet<T> functions, IlgpuBackendSelection selection, string tag)
    {
        IlgpuTestHelpers.AssertRealKernelExecution(evaluator, genome, functions);
        Assert.True(evaluator.Backend == selection.BackendName, $"{tag}: backend");
        Assert.True(evaluator.DeviceName == selection.DiagnosticDeviceName, $"{tag}: dispositivo");
    }

    private static string Describe(double[] values) => "[" + string.Join(",", values.Select(v => v.ToString("R"))) + "]";

    private static string Describe(int[] values) => "[" + string.Join(",", values) + "]";
}
