using ILGPU;
using Evo.Core;
using Evo.Core.Compiled;
using Evo.Core.Functions;
using Evo.Core.Values;
using Evo.Ilgpu;
using Xunit;

namespace Evo.Ilgpu.Tests;

/// <summary>
/// Matriz função × borda × tipo escalar × dispositivo (EVO-102, R1-1, parcial): cada função do conjunto padrão de
/// double e int é avaliada com entradas de borda nomeadas e comparada entre ReferenceGenomeEvaluator,
/// CompiledGenomeEvaluator (instância distinta) e IlgpuGenomeEvaluator em cada linha de
/// <see cref="IlgpuDeviceMatrixTests.Devices"/>. Os tipos array ficam pendentes (ver evidence/010b-matrix-correctness).
/// </summary>
public sealed class IlgpuFunctionEdgeMatrixTests
{
    private const double ConstantDouble = 0.5;
    private const int ConstantInt = 3;

    public static IEnumerable<object[]> DoubleEdges() =>
    [
        [1e-12, -1e-12],
        [1e-300, 2.0],
        [double.NaN, 1.0],
        [1.0, double.NaN],
        [double.NaN, double.NaN],
        [double.PositiveInfinity, 1.0],
        [double.NegativeInfinity, 1.0],
        [1.0, double.PositiveInfinity],
        [1.0, double.NegativeInfinity],
        [double.PositiveInfinity, double.NegativeInfinity],
        [0.0, 0.0],
        [-0.0, 0.0],
        [2.5, -1.25],
        [3.0, -0.5],
    ];

    public static IEnumerable<object[]> IntEdges() =>
    [
        [int.MaxValue, 1],
        [int.MinValue, -1],
        [int.MinValue, 0],
        [0, 0],
        [7, 0],
        [-4, 7],
        [int.MaxValue, int.MaxValue],
        [1, -1],
    ];

    [Theory]
    [MemberData(nameof(IlgpuDeviceMatrixTests.Devices), MemberType = typeof(IlgpuDeviceMatrixTests))]
    public void Double_function_edges_match_reference_and_compiled_on_device(string label, IlgpuBackendSelection selection)
    {
        var functions = StandardDoubleFunctionSet.Instance;
        var exercised = 0;
        foreach (var row in DoubleEdges())
        {
            var inputs = new[] { (double)row[0], (double)row[1] };
            for (var f = 0; f < functions.Count; f++)
            {
                var genome = SingleNodeGenome(functions, f, ConstantDouble, DoubleValueSemantics.Instance);
                var expected = new double[1];
                var compiled = new double[1];
                var device = new double[1];
                new ReferenceGenomeEvaluator<double>(functions).Evaluate(genome, inputs, expected);
                new CompiledGenomeEvaluator<double>(functions, DoubleValueSemantics.Instance).Evaluate(genome, inputs, compiled);
                using var evaluator = new IlgpuGenomeEvaluator<double>(functions, null, selection, null, null, null);
                evaluator.Evaluate(genome, inputs, device);
                var tag = $"{label} {functions.Get(f).Id} {Describe(inputs)}";
                AssertDeviceKernel(evaluator, genome, functions, selection, tag);
                IlgpuTestHelpers.AssertDoubleClose(expected[0], compiled[0], $"{tag} compiled");
                IlgpuTestHelpers.AssertDoubleClose(expected[0], device[0], $"{tag} device");
                exercised++;
            }
        }

        Assert.Equal(functions.Count * DoubleEdges().Count(), exercised);
    }

    [Theory]
    [MemberData(nameof(IlgpuDeviceMatrixTests.Devices), MemberType = typeof(IlgpuDeviceMatrixTests))]
    public void Int_function_edges_match_reference_and_compiled_on_device(string label, IlgpuBackendSelection selection)
    {
        var functions = StandardInt32FunctionSet.Instance;
        var exercised = 0;
        foreach (var row in IntEdges())
        {
            var inputs = new[] { (int)row[0], (int)row[1] };
            for (var f = 0; f < functions.Count; f++)
            {
                var genome = SingleNodeGenome(functions, f, ConstantInt, Int32ValueSemantics.Instance);
                var expected = new int[1];
                var compiled = new int[1];
                var device = new int[1];
                new ReferenceGenomeEvaluator<int>(functions).Evaluate(genome, inputs, expected);
                new CompiledGenomeEvaluator<int>(functions, Int32ValueSemantics.Instance).Evaluate(genome, inputs, compiled);
                using var evaluator = new IlgpuGenomeEvaluator<int>(functions, null, selection, null, null, null);
                evaluator.Evaluate(genome, inputs, device);
                var tag = $"{label} {functions.Get(f).Id} {Describe(inputs)}";
                AssertDeviceKernel(evaluator, genome, functions, selection, tag);
                Assert.Equal(expected[0], compiled[0]);
                Assert.Equal(expected[0], device[0]);
                exercised++;
            }
        }

        Assert.Equal(functions.Count * IntEdges().Count(), exercised);
    }

    /// <summary>Genoma mínimo: um nó com a função <paramref name="functionId"/> sobre Input0/Input1 e uma saída.</summary>
    private static Genome<T> SingleNodeGenome<T>(IFunctionSet<T> functions, int functionId, T constant, IValueSemantics<T> semantics)
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
