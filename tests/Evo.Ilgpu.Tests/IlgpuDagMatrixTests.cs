using System.Linq;
using System.Threading;
using Evo.Core;
using Evo.Core.Compiled;
using Evo.Core.Functions;
using Evo.Core.Values;
using Evo.Ilgpu;
using Xunit;

namespace Evo.Ilgpu.Tests;

/// <summary>
/// Matriz de DAG (EVO-102, R1-2): um único genoma por tipo combina fan-out, diamante/reconvergência,
/// argumento repetido, outputs múltiplos/repetidos/diretos, Constant e um nó inativo. Compara
/// ReferenceGenomeEvaluator, CompiledGenomeEvaluator (instância distinta) e ILGPU em cada linha de
/// <see cref="IlgpuDeviceMatrixTests.Devices"/>, afirma contagens reais por endereço contra uma
/// expectativa calculada à mão (independente de <see cref="Genome{T}.Analyze"/>) e repete a avaliação
/// com um segundo lote de entradas para comprovar isolamento e integridade profunda dos arrays
/// anteriores (inputs, parâmetros Constant de array e outputs).
/// </summary>
public sealed class IlgpuDagMatrixTests
{
    // Endereços: input0=0, input1=1; node0(Constant)=2, node1(Add)=3, node2(Add fan-out)=4,
    // node3(Multiply, argumento repetido addr4,addr4; consumidor 1 do fan-out)=5,
    // node4(Subtract, addr4-input0; consumidor 2 do fan-out)=6,
    // node5(Add, addr5+addr6 — DIAMANTE REAL: reconverge os dois consumidores do fan-out addr4)=7,
    // node6(Add)=8 — nunca referenciado, portanto INATIVO.
    private static readonly int[] ExpectedActiveAddresses = [2, 3, 4, 5, 6, 7];
    private static readonly int[] ExpectedInactiveAddresses = [8];
    private const int InputCount = 2;
    private const int NodeCount = 7;

    [Theory]
    [MemberData(nameof(IlgpuDeviceMatrixTests.Devices), MemberType = typeof(IlgpuDeviceMatrixTests))]
    public void Dag_matrix_matches_reference_with_independent_address_counts_and_isolation(string label, IlgpuBackendSelection selection)
    {
        AssertDouble(label, selection);
        AssertInt(label, selection);
        AssertDoubleArray(label, selection);
        AssertIntArray(label, selection);
    }

    private static GenomeShape Shape(int outputCount) =>
        new(InputCount: InputCount, NodeCount: NodeCount, OutputCount: outputCount, MaxArity: 2, LevelsBack: InputCount + NodeCount);

    private static void AssertExpectedAddresses(int[]? counts, string tag)
    {
        Assert.NotNull(counts);
        for (var input = 0; input < InputCount; input++)
        {
            Assert.Equal(0, counts![input]);
        }

        foreach (var addr in ExpectedActiveAddresses)
        {
            Assert.True(counts![addr] == 1, $"{tag}: endereço ativo {addr} deveria valer 1, valeu {counts[addr]}");
        }

        foreach (var addr in ExpectedInactiveAddresses)
        {
            Assert.True(counts![addr] == 0, $"{tag}: endereço inativo {addr} deveria valer 0, valeu {counts[addr]}");
        }
    }

    private static void AssertDouble(string label, IlgpuBackendSelection selection)
    {
        var functions = StandardDoubleFunctionSet.Instance;
        var constId = IlgpuTestHelpers.FindFunctionId(functions, "constant");
        var addId = IlgpuTestHelpers.FindFunctionId(functions, "add");
        var mulId = IlgpuTestHelpers.FindFunctionId(functions, "multiply");
        var subId = IlgpuTestHelpers.FindFunctionId(functions, "subtract");
        var genome = Genome<double>.FromGenes(
            Shape(5),
            [constId, addId, addId, mulId, subId, addId, addId],
            [0, 0, 0, 2, 0, 1, 4, 4, 4, 0, 5, 6, 0, 1],
            [5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [3, 7, 6, 7, 1],
            DoubleValueSemantics.Instance,
            functions);

        var inputs1 = new[] { 3.0, -0.5 };
        var inputs1Copy = (double[])inputs1.Clone();
        RunDagCase($"{label} double", genome, inputs1, functions,
            new ReferenceGenomeEvaluator<double>(functions),
            new CompiledGenomeEvaluator<double>(functions, DoubleValueSemantics.Instance),
            new IlgpuGenomeEvaluator<double>(functions, null, selection, null, null, null),
            selection,
            IlgpuTestHelpers.AssertDoubleArrayClose);
        Assert.Equal(inputs1Copy, inputs1);

        var inputs2 = new[] { 1.25, 9.0 };
        RunDagCase($"{label} double (segundo lote)", genome, inputs2, functions,
            new ReferenceGenomeEvaluator<double>(functions),
            new CompiledGenomeEvaluator<double>(functions, DoubleValueSemantics.Instance),
            new IlgpuGenomeEvaluator<double>(functions, null, selection, null, null, null),
            selection,
            IlgpuTestHelpers.AssertDoubleArrayClose);
        Assert.Equal(inputs1Copy, inputs1);
    }

    private static void AssertInt(string label, IlgpuBackendSelection selection)
    {
        var functions = StandardInt32FunctionSet.Instance;
        var constId = IlgpuTestHelpers.FindFunctionId(functions, "constant");
        var addId = IlgpuTestHelpers.FindFunctionId(functions, "add");
        var mulId = IlgpuTestHelpers.FindFunctionId(functions, "multiply");
        var subId = IlgpuTestHelpers.FindFunctionId(functions, "subtract");
        var genome = Genome<int>.FromGenes(
            Shape(5),
            [constId, addId, addId, mulId, subId, addId, addId],
            [0, 0, 0, 2, 0, 1, 4, 4, 4, 0, 5, 6, 0, 1],
            [5, 0, 0, 0, 0, 0, 0],
            [3, 7, 6, 7, 1],
            Int32ValueSemantics.Instance,
            functions);

        var inputs1 = new[] { -4, 7 };
        var inputs1Copy = (int[])inputs1.Clone();
        RunDagCase($"{label} int", genome, inputs1, functions,
            new ReferenceGenomeEvaluator<int>(functions),
            new CompiledGenomeEvaluator<int>(functions, Int32ValueSemantics.Instance),
            new IlgpuGenomeEvaluator<int>(functions, null, selection, null, null, null),
            selection,
            (expected, actual, prefix) => Assert.Equal(expected, actual));
        Assert.Equal(inputs1Copy, inputs1);

        var inputs2 = new[] { int.MaxValue, 2 };
        RunDagCase($"{label} int (segundo lote)", genome, inputs2, functions,
            new ReferenceGenomeEvaluator<int>(functions),
            new CompiledGenomeEvaluator<int>(functions, Int32ValueSemantics.Instance),
            new IlgpuGenomeEvaluator<int>(functions, null, selection, null, null, null),
            selection,
            (expected, actual, prefix) => Assert.Equal(expected, actual));
        Assert.Equal(inputs1Copy, inputs1);
    }

    private static void AssertDoubleArray(string label, IlgpuBackendSelection selection)
    {
        var functions = StandardDoubleArrayFunctionSet.Instance;
        var constId = IlgpuTestHelpers.FindFunctionId(functions, "constant");
        var addId = IlgpuTestHelpers.FindFunctionId(functions, "add");
        var mulId = IlgpuTestHelpers.FindFunctionId(functions, "multiply");
        var subId = IlgpuTestHelpers.FindFunctionId(functions, "subtract");
        var constantParam = new[] { 9.0, -1.0 };
        var genome = Genome<double[]>.FromGenes(
            Shape(5),
            [constId, addId, addId, mulId, subId, addId, addId],
            [0, 0, 0, 2, 0, 1, 4, 4, 4, 0, 5, 6, 0, 1],
            [constantParam, [], [], [], [], [], []],
            [3, 7, 6, 7, 1],
            DoubleArrayValueSemantics.Instance,
            functions);
        var constantParamCopy = (double[])constantParam.Clone();

        var inputs1 = new[] { new[] { 1.0, 2.0 }, new[] { 3.0, 4.0 } };
        var inputs1Copy = inputs1.Select(a => (double[])a.Clone()).ToArray();
        RunDagArrayCase($"{label} double[]", genome, inputs1, functions,
            new ReferenceGenomeEvaluator<double[]>(functions),
            new CompiledGenomeEvaluator<double[]>(functions, DoubleArrayValueSemantics.Instance),
            new IlgpuGenomeEvaluator<double[]>(functions, null, selection, null, null, null),
            selection);
        for (var i = 0; i < inputs1.Length; i++)
        {
            Assert.Equal(inputs1Copy[i], inputs1[i]);
        }

        Assert.Equal(constantParamCopy, constantParam);

        var inputs2 = new[] { new[] { -5.0, 0.5 }, new[] { 2.0, -2.0 } };
        RunDagArrayCase($"{label} double[] (segundo lote)", genome, inputs2, functions,
            new ReferenceGenomeEvaluator<double[]>(functions),
            new CompiledGenomeEvaluator<double[]>(functions, DoubleArrayValueSemantics.Instance),
            new IlgpuGenomeEvaluator<double[]>(functions, null, selection, null, null, null),
            selection);
        for (var i = 0; i < inputs1.Length; i++)
        {
            Assert.Equal(inputs1Copy[i], inputs1[i]);
        }

        Assert.Equal(constantParamCopy, constantParam);
    }

    private static void AssertIntArray(string label, IlgpuBackendSelection selection)
    {
        var functions = StandardInt32ArrayFunctionSet.Instance;
        var constId = IlgpuTestHelpers.FindFunctionId(functions, "constant");
        var addId = IlgpuTestHelpers.FindFunctionId(functions, "add");
        var mulId = IlgpuTestHelpers.FindFunctionId(functions, "multiply");
        var subId = IlgpuTestHelpers.FindFunctionId(functions, "subtract");
        var constantParam = new[] { 9, -1 };
        var genome = Genome<int[]>.FromGenes(
            Shape(5),
            [constId, addId, addId, mulId, subId, addId, addId],
            [0, 0, 0, 2, 0, 1, 4, 4, 4, 0, 5, 6, 0, 1],
            [constantParam, [], [], [], [], [], []],
            [3, 7, 6, 7, 1],
            Int32ArrayValueSemantics.Instance,
            functions);
        var constantParamCopy = (int[])constantParam.Clone();

        var inputs1 = new[] { new[] { 1, 2 }, new[] { 3, 4 } };
        var inputs1Copy = inputs1.Select(a => (int[])a.Clone()).ToArray();
        RunDagArrayCase($"{label} int[]", genome, inputs1, functions,
            new ReferenceGenomeEvaluator<int[]>(functions),
            new CompiledGenomeEvaluator<int[]>(functions, Int32ArrayValueSemantics.Instance),
            new IlgpuGenomeEvaluator<int[]>(functions, null, selection, null, null, null),
            selection);
        for (var i = 0; i < inputs1.Length; i++)
        {
            Assert.Equal(inputs1Copy[i], inputs1[i]);
        }

        Assert.Equal(constantParamCopy, constantParam);

        var inputs2 = new[] { new[] { -5, 10 }, new[] { 2, -2 } };
        RunDagArrayCase($"{label} int[] (segundo lote)", genome, inputs2, functions,
            new ReferenceGenomeEvaluator<int[]>(functions),
            new CompiledGenomeEvaluator<int[]>(functions, Int32ArrayValueSemantics.Instance),
            new IlgpuGenomeEvaluator<int[]>(functions, null, selection, null, null, null),
            selection);
        for (var i = 0; i < inputs1.Length; i++)
        {
            Assert.Equal(inputs1Copy[i], inputs1[i]);
        }

        Assert.Equal(constantParamCopy, constantParam);
    }

    private static void RunDagCase<T>(
        string tag,
        Genome<T> genome,
        T[] inputs,
        IFunctionSet<T> functions,
        IGenomeEvaluator<T> reference,
        IGenomeEvaluator<T> compiled,
        IlgpuGenomeEvaluator<T> device,
        IlgpuBackendSelection selection,
        Action<T[], T[], string> assertClose)
    {
        using var deviceEvaluator = device;
        var expected = new T[genome.Shape.OutputCount];
        var compiledActual = new T[genome.Shape.OutputCount];
        var deviceActual = new T[genome.Shape.OutputCount];
        reference.Evaluate(genome, inputs, expected);
        compiled.Evaluate(genome, inputs, compiledActual);
        deviceEvaluator.Evaluate(genome, inputs, deviceActual);

        assertClose(expected, compiledActual, $"{tag} compiled");
        assertClose(expected, deviceActual, $"{tag} device");

        IlgpuTestHelpers.AssertRealKernelExecution(deviceEvaluator, genome, functions);
        Assert.Equal(selection.BackendName, deviceEvaluator.Backend);
        Assert.Equal(selection.DiagnosticDeviceName, deviceEvaluator.DeviceName);
        AssertExpectedAddresses(deviceEvaluator.LastDeviceExecutionCounts, tag);
    }

    private static void RunDagArrayCase<T>(
        string tag,
        Genome<T[]> genome,
        T[][] inputs,
        IFunctionSet<T[]> functions,
        IGenomeEvaluator<T[]> reference,
        IGenomeEvaluator<T[]> compiled,
        IlgpuGenomeEvaluator<T[]> device,
        IlgpuBackendSelection selection)
        where T : notnull
    {
        using var deviceEvaluator = device;
        var expected = new T[genome.Shape.OutputCount][];
        var compiledActual = new T[genome.Shape.OutputCount][];
        var deviceActual = new T[genome.Shape.OutputCount][];
        reference.Evaluate(genome, inputs, expected);
        compiled.Evaluate(genome, inputs, compiledActual);
        deviceEvaluator.Evaluate(genome, inputs, deviceActual);

        for (var i = 0; i < expected.Length; i++)
        {
            Assert.Equal(expected[i], compiledActual[i]);
            Assert.Equal(expected[i], deviceActual[i]);
        }

        IlgpuTestHelpers.AssertRealKernelExecution(deviceEvaluator, genome, functions);
        Assert.Equal(selection.BackendName, deviceEvaluator.Backend);
        Assert.Equal(selection.DiagnosticDeviceName, deviceEvaluator.DeviceName);
        AssertExpectedAddresses(deviceEvaluator.LastDeviceExecutionCounts, tag);
    }
}
