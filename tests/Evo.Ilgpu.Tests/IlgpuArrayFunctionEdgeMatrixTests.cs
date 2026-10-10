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
    ];

    public static IEnumerable<object[]> IntArrayEdges() =>
    [
        [new[] { int.MaxValue, 1, 0 }, new[] { 1, int.MaxValue, 0 }],
        [new[] { int.MinValue, -1, 7 }, new[] { -1, int.MinValue, -4 }],
        [new[] { int.MinValue, 0, int.MaxValue }, new[] { 0, int.MinValue, int.MaxValue }],
        [new[] { 0, 0, 0 }, new[] { 0, 0, 0 }],
        [new[] { 7, -4, 1 }, new[] { -3, 7, -1 }],
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
        var device = new double[1][];
        new ReferenceGenomeEvaluator<double[]>(functions).Evaluate(genome, inputs, expected);
        using var evaluator = new IlgpuGenomeEvaluator<double[]>(functions, null, selection, null, null, null);
        evaluator.Evaluate(genome, inputs, device);

        Assert.Empty(expected[0]);
        Assert.Empty(device[0]);
        Assert.False(evaluator.LastEvaluationUsedDevice);
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
        using var evaluator = new IlgpuGenomeEvaluator<double[]>(functions, null, selection, null, null, null);
        Assert.Throws<ArgumentException>(() => evaluator.Evaluate(genome, inputs, output));
        Assert.False(evaluator.LastEvaluationUsedDevice);
    }

    /// <summary>Genoma mínimo: um nó com a função <paramref name="functionId"/> sobre Input0/Input1 e uma saída.</summary>
    private static Genome<T> SingleNodeArrayGenome<T>(IFunctionSet<T> functions, int functionId, T constant, IValueSemantics<T> semantics)
    {
        var shape = new GenomeShape(InputCount: 2, NodeCount: 1, OutputCount: 1, MaxArity: 3, LevelsBack: 1);
        return Genome<T>.FromGenes(shape, [functionId], [0, 1, 1], [constant], [2], semantics, functions);
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
