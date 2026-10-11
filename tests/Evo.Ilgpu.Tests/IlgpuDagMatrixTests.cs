using System.Linq;
using Evo.Core;
using Evo.Core.Compiled;
using Evo.Core.Functions;
using Evo.Core.Values;
using Evo.Ilgpu;
using Xunit;

namespace Evo.Ilgpu.Tests;

/// <summary>
/// Matriz de DAG (EVO-102, R1-2, R7-3, R7-4): um único genoma "A" por tipo combina fan-out, diamante/reconvergência,
/// argumento repetido, outputs múltiplos/repetidos/diretos, Constant e um nó inativo. Um segundo genoma "B" de
/// mesmo shape (mesmas funções/conexões/parâmetros, outputs diferentes) ativa um fenótipo bem menor para provar
/// isolamento real: as MESMAS instâncias de ReferenceGenomeEvaluator, CompiledGenomeEvaluator e
/// IlgpuGenomeEvaluator são reutilizadas entre dois lotes de entradas do genoma A, depois para o genoma B, e por
/// fim de volta ao genoma A — comparando contagens reais por endereço, saídas e integridade profunda de inputs e
/// parâmetros Constant de array entre todas as chamadas.
/// </summary>
public sealed class IlgpuDagMatrixTests
{
    // Endereços: input0=0, input1=1; node0(Constant)=2, node1(Add)=3, node2(Add fan-out)=4,
    // node3(Multiply, argumento repetido addr4,addr4; consumidor 1 do fan-out)=5,
    // node4(Subtract, addr4-input0; consumidor 2 do fan-out)=6,
    // node5(Add, addr5+addr6 — DIAMANTE REAL: reconverge os dois consumidores do fan-out addr4)=7,
    // node6(Add)=8.
    // Genoma A: outputs=[3,7,6,7,1] ativa {2,3,4,5,6,7}, deixa addr8 inativo.
    // Genoma B: mesmas funções/conexões/parâmetros, outputs=[3,2,3,2,0] ativa apenas {2,3} (fenótipo bem menor).
    private static readonly int[] ActiveA = [2, 3, 4, 5, 6, 7];
    private static readonly int[] InactiveA = [8];
    private static readonly int[] ActiveB = [2, 3];
    private static readonly int[] InactiveB = [4, 5, 6, 7, 8];
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

    private static void AssertExpectedAddresses(int[]? counts, int[] active, int[] inactive, string tag)
    {
        Assert.NotNull(counts);
        for (var input = 0; input < InputCount; input++)
        {
            Assert.Equal(0, counts![input]);
        }

        foreach (var addr in active)
        {
            Assert.True(counts![addr] == 1, $"{tag}: endereço ativo {addr} deveria valer 1, valeu {counts[addr]}");
        }

        foreach (var addr in inactive)
        {
            Assert.True(counts![addr] == 0, $"{tag}: endereço inativo {addr} deveria valer 0, valeu {counts[addr]}");
        }
    }

    /// <summary>
    /// R9-1: afirma contagens reais por nó (endereço - InputCount) de ReferenceExecutionCounters/
    /// CompiledExecutionCounters: ativo=1, inativo=0. Os contadores são reiniciados antes da chamada
    /// para que a asserção seja um valor exato por chamada, não um delta acumulado.
    /// </summary>
    private static void AssertExpectedNodeCounts(int[] nodeCounts, int[] active, int[] inactive, string tag)
    {
        foreach (var addr in active)
        {
            var node = addr - InputCount;
            Assert.True(nodeCounts[node] == 1, $"{tag}: nó do endereço ativo {addr} deveria valer 1, valeu {nodeCounts[node]}");
        }

        foreach (var addr in inactive)
        {
            var node = addr - InputCount;
            Assert.True(nodeCounts[node] == 0, $"{tag}: nó do endereço inativo {addr} deveria valer 0, valeu {nodeCounts[node]}");
        }
    }

    /// <summary>
    /// R9-2: retém snapshots profundos independentes dos três caminhos (reference/compiled/device),
    /// não apenas de "expected", para que o chamador possa conferir ausência de mutação entre chamadas
    /// sucessivas reaproveitando as mesmas instâncias de evaluator.
    /// </summary>
    private readonly record struct ScalarSnapshot<T>(T[] Reference, T[] Compiled, T[] Device);

    private static ScalarSnapshot<T> EvaluateAndAssert<T>(
        string tag,
        Genome<T> genome,
        T[] inputs,
        IFunctionSet<T> functions,
        IGenomeEvaluator<T> reference,
        IGenomeEvaluator<T> compiled,
        IlgpuGenomeEvaluator<T> device,
        IlgpuBackendSelection selection,
        int[] active,
        int[] inactive,
        Action<T[], T[], string> assertClose,
        ReferenceExecutionCounters refCounters,
        CompiledExecutionCounters compiledCounters)
    {
        refCounters.Reset();
        compiledCounters.Reset();

        var expected = new T[genome.Shape.OutputCount];
        var compiledActual = new T[genome.Shape.OutputCount];
        var deviceActual = new T[genome.Shape.OutputCount];
        reference.Evaluate(genome, inputs, expected);
        compiled.Evaluate(genome, inputs, compiledActual);
        device.Evaluate(genome, inputs, deviceActual);

        assertClose(expected, compiledActual, $"{tag} compiled");
        assertClose(expected, deviceActual, $"{tag} device");

        IlgpuTestHelpers.AssertRealKernelExecution(device, genome, functions);
        Assert.Equal(selection.BackendName, device.Backend);
        Assert.Equal(selection.DiagnosticDeviceName, device.DeviceName);
        AssertExpectedAddresses(device.LastDeviceExecutionCounts, active, inactive, tag);
        AssertExpectedNodeCounts(refCounters.NodeExecutions, active, inactive, $"{tag} reference");
        AssertExpectedNodeCounts(compiledCounters.ScalarExecutions, active, inactive, $"{tag} compiled");
        return new ScalarSnapshot<T>((T[])expected.Clone(), (T[])compiledActual.Clone(), (T[])deviceActual.Clone());
    }

    /// <summary>
    /// R9-2: confere que um snapshot retido de uma chamada anterior (reference/compiled/device)
    /// permanece intacto depois de reaproveitar as mesmas instâncias de evaluator em chamadas
    /// subsequentes (outro lote, genoma B, retorno a A). Ausência de mutação double é bit a bit.
    /// </summary>
    private static void AssertSnapshotUnchanged(string tag, ScalarSnapshot<double> previous, ScalarSnapshot<double> previousClone)
    {
        IlgpuTestHelpers.AssertDoubleArrayBitExact(previousClone.Reference, previous.Reference, $"{tag} reference retido");
        IlgpuTestHelpers.AssertDoubleArrayBitExact(previousClone.Compiled, previous.Compiled, $"{tag} compiled retido");
        IlgpuTestHelpers.AssertDoubleArrayBitExact(previousClone.Device, previous.Device, $"{tag} device retido");
    }

    private static void AssertSnapshotUnchanged(string tag, ScalarSnapshot<int> previous, ScalarSnapshot<int> previousClone)
    {
        Assert.Equal(previousClone.Reference, previous.Reference);
        Assert.Equal(previousClone.Compiled, previous.Compiled);
        Assert.Equal(previousClone.Device, previous.Device);
    }

    private readonly record struct ArraySnapshot<T>(T[][] Reference, T[][] Compiled, T[][] Device) where T : notnull;

    private static T[][] CloneJagged<T>(T[][] source) where T : notnull
    {
        var clone = new T[source.Length][];
        for (var i = 0; i < source.Length; i++)
        {
            clone[i] = source[i] switch
            {
                double[] d => (T[])(object)(double[])d.Clone(),
                int[] n => (T[])(object)(int[])n.Clone(),
                _ => (T[])source[i].Clone()
            };
        }

        return clone;
    }

    private static void AssertJaggedEqual<T>(T[][] expected, T[][] actual, string prefix) where T : notnull
    {
        Assert.Equal(expected.Length, actual.Length);
        for (var i = 0; i < expected.Length; i++)
        {
            if (expected[i] is double[] expectedDoubles && actual[i] is double[] actualDoubles)
            {
                IlgpuTestHelpers.AssertDoubleArrayBitExact(expectedDoubles, actualDoubles, $"{prefix}[{i}]");
            }
            else
            {
                Assert.Equal(expected[i], actual[i]);
            }
        }
    }

    /// <summary>
    /// R9-2: confere que um snapshot de array retido de uma chamada anterior permanece intacto
    /// após reaproveitar as mesmas instâncias de evaluator em chamadas subsequentes.
    /// </summary>
    private static void AssertArraySnapshotUnchanged<T>(string tag, ArraySnapshot<T> previous, ArraySnapshot<T> previousClone) where T : notnull
    {
        AssertJaggedEqual(previousClone.Reference, previous.Reference, $"{tag} reference retido");
        AssertJaggedEqual(previousClone.Compiled, previous.Compiled, $"{tag} compiled retido");
        AssertJaggedEqual(previousClone.Device, previous.Device, $"{tag} device retido");
    }

    private static ArraySnapshot<T> EvaluateArrayAndAssert<T>(
        string tag,
        Genome<T[]> genome,
        T[][] inputs,
        IFunctionSet<T[]> functions,
        IGenomeEvaluator<T[]> reference,
        IGenomeEvaluator<T[]> compiled,
        IlgpuGenomeEvaluator<T[]> device,
        IlgpuBackendSelection selection,
        int[] active,
        int[] inactive,
        ReferenceExecutionCounters refCounters,
        CompiledExecutionCounters compiledCounters)
        where T : notnull
    {
        refCounters.Reset();
        compiledCounters.Reset();

        var expected = new T[genome.Shape.OutputCount][];
        var compiledActual = new T[genome.Shape.OutputCount][];
        var deviceActual = new T[genome.Shape.OutputCount][];
        reference.Evaluate(genome, inputs, expected);
        compiled.Evaluate(genome, inputs, compiledActual);
        device.Evaluate(genome, inputs, deviceActual);

        for (var i = 0; i < expected.Length; i++)
        {
            Assert.Equal(expected[i], compiledActual[i]);
            Assert.Equal(expected[i], deviceActual[i]);
        }

        IlgpuTestHelpers.AssertRealKernelExecution(device, genome, functions);
        Assert.Equal(selection.BackendName, device.Backend);
        Assert.Equal(selection.DiagnosticDeviceName, device.DeviceName);
        AssertExpectedAddresses(device.LastDeviceExecutionCounts, active, inactive, tag);
        AssertExpectedNodeCounts(refCounters.NodeExecutions, active, inactive, $"{tag} reference");
        AssertExpectedNodeCounts(compiledCounters.ScalarExecutions, active, inactive, $"{tag} compiled");
        return new ArraySnapshot<T>(CloneJagged(expected), CloneJagged(compiledActual), CloneJagged(deviceActual));
    }

    private static void AssertDouble(string label, IlgpuBackendSelection selection)
    {
        var functions = StandardDoubleFunctionSet.Instance;
        var constId = IlgpuTestHelpers.FindFunctionId(functions, "constant");
        var addId = IlgpuTestHelpers.FindFunctionId(functions, "add");
        var mulId = IlgpuTestHelpers.FindFunctionId(functions, "multiply");
        var subId = IlgpuTestHelpers.FindFunctionId(functions, "subtract");
        var genes = new[] { constId, addId, addId, mulId, subId, addId, addId };
        var connections = new[] { 0, 0, 0, 2, 0, 1, 4, 4, 4, 0, 5, 6, 0, 1 };
        var parameters = new[] { 5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
        var genomeA = Genome<double>.FromGenes(Shape(5), genes, connections, parameters, [3, 7, 6, 7, 1], DoubleValueSemantics.Instance, functions);
        var genomeB = Genome<double>.FromGenes(Shape(5), genes, connections, parameters, [3, 2, 3, 2, 0], DoubleValueSemantics.Instance, functions);

        var refCounters = new ReferenceExecutionCounters();
        var compiledCounters = new CompiledExecutionCounters();
        var reference = new ReferenceGenomeEvaluator<double>(functions, refCounters);
        var compiled = new CompiledGenomeEvaluator<double>(functions, DoubleValueSemantics.Instance, counters: compiledCounters);
        using var device = new IlgpuGenomeEvaluator<double>(functions, null, selection, null, null, null);

        var paramsACopy = genomeA.ParameterGenes.ToArray();
        var paramsBCopy = genomeB.ParameterGenes.ToArray();

        var inputs1 = new[] { 3.0, -0.5 };
        var inputs1Copy = (double[])inputs1.Clone();
        var out1 = EvaluateAndAssert($"{label} double A lote1", genomeA, inputs1, functions, reference, compiled, device, selection, ActiveA, InactiveA, IlgpuTestHelpers.AssertDoubleArrayClose, refCounters, compiledCounters);
        Assert.Equal(inputs1Copy, inputs1);
        IlgpuTestHelpers.AssertDoubleArrayBitExact(paramsACopy, genomeA.ParameterGenes.ToArray(), $"{label} double A lote1 ParameterGenes");

        var inputs2 = new[] { 1.25, 9.0 };
        var out2 = EvaluateAndAssert($"{label} double A lote2", genomeA, inputs2, functions, reference, compiled, device, selection, ActiveA, InactiveA, IlgpuTestHelpers.AssertDoubleArrayClose, refCounters, compiledCounters);
        Assert.Equal(inputs1Copy, inputs1);
        IlgpuTestHelpers.AssertDoubleArrayBitExact(paramsACopy, genomeA.ParameterGenes.ToArray(), $"{label} double A lote2 ParameterGenes");
        AssertSnapshotUnchanged($"{label} double A lote1 pós-lote2", out1, out1);

        var inputsB = new[] { -2.0, 4.0 };
        var outB = EvaluateAndAssert($"{label} double B", genomeB, inputsB, functions, reference, compiled, device, selection, ActiveB, InactiveB, IlgpuTestHelpers.AssertDoubleArrayClose, refCounters, compiledCounters);
        IlgpuTestHelpers.AssertDoubleArrayBitExact(paramsBCopy, genomeB.ParameterGenes.ToArray(), $"{label} double B ParameterGenes");
        AssertSnapshotUnchanged($"{label} double A lote1 pós-B", out1, out1);
        AssertSnapshotUnchanged($"{label} double A lote2 pós-B", out2, out2);

        var out1Again = EvaluateAndAssert($"{label} double A retorno", genomeA, inputs1, functions, reference, compiled, device, selection, ActiveA, InactiveA, IlgpuTestHelpers.AssertDoubleArrayClose, refCounters, compiledCounters);
        IlgpuTestHelpers.AssertDoubleArrayBitExact(out1.Reference, out1Again.Reference, $"{label} double A retorno deve repetir saída reference do lote1");
        IlgpuTestHelpers.AssertDoubleArrayBitExact(out1.Compiled, out1Again.Compiled, $"{label} double A retorno deve repetir saída compiled do lote1");
        IlgpuTestHelpers.AssertDoubleArrayBitExact(out1.Device, out1Again.Device, $"{label} double A retorno deve repetir saída device do lote1");
        Assert.Equal(inputs1Copy, inputs1);
        IlgpuTestHelpers.AssertDoubleArrayBitExact(paramsACopy, genomeA.ParameterGenes.ToArray(), $"{label} double A retorno ParameterGenes");
        IlgpuTestHelpers.AssertDoubleArrayBitExact(paramsBCopy, genomeB.ParameterGenes.ToArray(), $"{label} double B ParameterGenes pós-retorno");
    }

    private static void AssertInt(string label, IlgpuBackendSelection selection)
    {
        var functions = StandardInt32FunctionSet.Instance;
        var constId = IlgpuTestHelpers.FindFunctionId(functions, "constant");
        var addId = IlgpuTestHelpers.FindFunctionId(functions, "add");
        var mulId = IlgpuTestHelpers.FindFunctionId(functions, "multiply");
        var subId = IlgpuTestHelpers.FindFunctionId(functions, "subtract");
        var genes = new[] { constId, addId, addId, mulId, subId, addId, addId };
        var connections = new[] { 0, 0, 0, 2, 0, 1, 4, 4, 4, 0, 5, 6, 0, 1 };
        var parameters = new[] { 5, 0, 0, 0, 0, 0, 0 };
        var genomeA = Genome<int>.FromGenes(Shape(5), genes, connections, parameters, [3, 7, 6, 7, 1], Int32ValueSemantics.Instance, functions);
        var genomeB = Genome<int>.FromGenes(Shape(5), genes, connections, parameters, [3, 2, 3, 2, 0], Int32ValueSemantics.Instance, functions);

        var refCounters = new ReferenceExecutionCounters();
        var compiledCounters = new CompiledExecutionCounters();
        var reference = new ReferenceGenomeEvaluator<int>(functions, refCounters);
        var compiled = new CompiledGenomeEvaluator<int>(functions, Int32ValueSemantics.Instance, counters: compiledCounters);
        using var device = new IlgpuGenomeEvaluator<int>(functions, null, selection, null, null, null);
        Action<int[], int[], string> assertEqual = (expected, actual, _) => Assert.Equal(expected, actual);
        var paramsACopy = genomeA.ParameterGenes.ToArray();
        var paramsBCopy = genomeB.ParameterGenes.ToArray();

        var inputs1 = new[] { -4, 7 };
        var inputs1Copy = (int[])inputs1.Clone();
        var out1 = EvaluateAndAssert($"{label} int A lote1", genomeA, inputs1, functions, reference, compiled, device, selection, ActiveA, InactiveA, assertEqual, refCounters, compiledCounters);
        Assert.Equal(inputs1Copy, inputs1);
        Assert.Equal(paramsACopy, genomeA.ParameterGenes.ToArray());

        var inputs2 = new[] { int.MaxValue, 2 };
        var out2 = EvaluateAndAssert($"{label} int A lote2", genomeA, inputs2, functions, reference, compiled, device, selection, ActiveA, InactiveA, assertEqual, refCounters, compiledCounters);
        Assert.Equal(inputs1Copy, inputs1);
        Assert.Equal(paramsACopy, genomeA.ParameterGenes.ToArray());
        AssertSnapshotUnchanged($"{label} int A lote1 pós-lote2", out1, out1);

        var inputsB = new[] { -9, 11 };
        var outB = EvaluateAndAssert($"{label} int B", genomeB, inputsB, functions, reference, compiled, device, selection, ActiveB, InactiveB, assertEqual, refCounters, compiledCounters);
        Assert.Equal(paramsBCopy, genomeB.ParameterGenes.ToArray());
        AssertSnapshotUnchanged($"{label} int A lote1 pós-B", out1, out1);
        AssertSnapshotUnchanged($"{label} int A lote2 pós-B", out2, out2);

        var out1Again = EvaluateAndAssert($"{label} int A retorno", genomeA, inputs1, functions, reference, compiled, device, selection, ActiveA, InactiveA, assertEqual, refCounters, compiledCounters);
        Assert.Equal(out1.Reference, out1Again.Reference);
        Assert.Equal(out1.Compiled, out1Again.Compiled);
        Assert.Equal(out1.Device, out1Again.Device);
        Assert.Equal(inputs1Copy, inputs1);
        Assert.Equal(paramsACopy, genomeA.ParameterGenes.ToArray());
        Assert.Equal(paramsBCopy, genomeB.ParameterGenes.ToArray());
    }

    private static void AssertDoubleArray(string label, IlgpuBackendSelection selection)
    {
        var functions = StandardDoubleArrayFunctionSet.Instance;
        var constId = IlgpuTestHelpers.FindFunctionId(functions, "constant");
        var addId = IlgpuTestHelpers.FindFunctionId(functions, "add");
        var mulId = IlgpuTestHelpers.FindFunctionId(functions, "multiply");
        var subId = IlgpuTestHelpers.FindFunctionId(functions, "subtract");
        var genes = new[] { constId, addId, addId, mulId, subId, addId, addId };
        var connections = new[] { 0, 0, 0, 2, 0, 1, 4, 4, 4, 0, 5, 6, 0, 1 };
        var constantParam = new[] { 9.0, -1.0 };
        var constantParamCopy = (double[])constantParam.Clone();
        var parameters = new[] { constantParam, [], [], [], [], [], (double[])[] };
        var genomeA = Genome<double[]>.FromGenes(Shape(5), genes, connections, parameters, [3, 7, 6, 7, 1], DoubleArrayValueSemantics.Instance, functions);
        var genomeB = Genome<double[]>.FromGenes(Shape(5), genes, connections, parameters, [3, 2, 3, 2, 0], DoubleArrayValueSemantics.Instance, functions);

        var refCounters = new ReferenceExecutionCounters();
        var compiledCounters = new CompiledExecutionCounters();
        var reference = new ReferenceGenomeEvaluator<double[]>(functions, refCounters);
        var compiled = new CompiledGenomeEvaluator<double[]>(functions, DoubleArrayValueSemantics.Instance, counters: compiledCounters);
        using var device = new IlgpuGenomeEvaluator<double[]>(functions, null, selection, null, null, null);
        var paramsACopy = CloneJagged(genomeA.ParameterGenes.ToArray());
        var paramsBCopy = CloneJagged(genomeB.ParameterGenes.ToArray());

        var inputs1 = new[] { new[] { 1.0, 2.0 }, new[] { 3.0, 4.0 } };
        var inputs1Copy = inputs1.Select(a => (double[])a.Clone()).ToArray();
        var out1 = EvaluateArrayAndAssert($"{label} double[] A lote1", genomeA, inputs1, functions, reference, compiled, device, selection, ActiveA, InactiveA, refCounters, compiledCounters);
        for (var i = 0; i < inputs1.Length; i++) Assert.Equal(inputs1Copy[i], inputs1[i]);
        Assert.Equal(constantParamCopy, constantParam);
        AssertJaggedEqual(paramsACopy, genomeA.ParameterGenes.ToArray(), $"{label} double[] A lote1 ParameterGenes");

        var inputs2 = new[] { new[] { -5.0, 0.5 }, new[] { 2.0, -2.0 } };
        var out2 = EvaluateArrayAndAssert($"{label} double[] A lote2", genomeA, inputs2, functions, reference, compiled, device, selection, ActiveA, InactiveA, refCounters, compiledCounters);
        for (var i = 0; i < inputs1.Length; i++) Assert.Equal(inputs1Copy[i], inputs1[i]);
        Assert.Equal(constantParamCopy, constantParam);
        AssertJaggedEqual(paramsACopy, genomeA.ParameterGenes.ToArray(), $"{label} double[] A lote2 ParameterGenes");
        AssertArraySnapshotUnchanged($"{label} double[] A lote1 pós-lote2", out1, out1);

        var inputsB = new[] { new[] { 7.0, -3.0 }, new[] { 0.5, 6.0 } };
        var outB = EvaluateArrayAndAssert($"{label} double[] B", genomeB, inputsB, functions, reference, compiled, device, selection, ActiveB, InactiveB, refCounters, compiledCounters);
        Assert.Equal(constantParamCopy, constantParam);
        AssertJaggedEqual(paramsBCopy, genomeB.ParameterGenes.ToArray(), $"{label} double[] B ParameterGenes");
        AssertArraySnapshotUnchanged($"{label} double[] A lote1 pós-B", out1, out1);
        AssertArraySnapshotUnchanged($"{label} double[] A lote2 pós-B", out2, out2);

        var out1Again = EvaluateArrayAndAssert($"{label} double[] A retorno", genomeA, inputs1, functions, reference, compiled, device, selection, ActiveA, InactiveA, refCounters, compiledCounters);
        AssertJaggedEqual(out1.Reference, out1Again.Reference, $"{label} double[] A retorno reference");
        AssertJaggedEqual(out1.Compiled, out1Again.Compiled, $"{label} double[] A retorno compiled");
        AssertJaggedEqual(out1.Device, out1Again.Device, $"{label} double[] A retorno device");
        for (var i = 0; i < inputs1.Length; i++) Assert.Equal(inputs1Copy[i], inputs1[i]);
        Assert.Equal(constantParamCopy, constantParam);
        AssertJaggedEqual(paramsACopy, genomeA.ParameterGenes.ToArray(), $"{label} double[] A retorno ParameterGenes");
        AssertJaggedEqual(paramsBCopy, genomeB.ParameterGenes.ToArray(), $"{label} double[] B ParameterGenes pós-retorno");
    }

    private static void AssertIntArray(string label, IlgpuBackendSelection selection)
    {
        var functions = StandardInt32ArrayFunctionSet.Instance;
        var constId = IlgpuTestHelpers.FindFunctionId(functions, "constant");
        var addId = IlgpuTestHelpers.FindFunctionId(functions, "add");
        var mulId = IlgpuTestHelpers.FindFunctionId(functions, "multiply");
        var subId = IlgpuTestHelpers.FindFunctionId(functions, "subtract");
        var genes = new[] { constId, addId, addId, mulId, subId, addId, addId };
        var connections = new[] { 0, 0, 0, 2, 0, 1, 4, 4, 4, 0, 5, 6, 0, 1 };
        var constantParam = new[] { 9, -1 };
        var constantParamCopy = (int[])constantParam.Clone();
        var parameters = new[] { constantParam, [], [], [], [], [], (int[])[] };
        var genomeA = Genome<int[]>.FromGenes(Shape(5), genes, connections, parameters, [3, 7, 6, 7, 1], Int32ArrayValueSemantics.Instance, functions);
        var genomeB = Genome<int[]>.FromGenes(Shape(5), genes, connections, parameters, [3, 2, 3, 2, 0], Int32ArrayValueSemantics.Instance, functions);

        var refCounters = new ReferenceExecutionCounters();
        var compiledCounters = new CompiledExecutionCounters();
        var reference = new ReferenceGenomeEvaluator<int[]>(functions, refCounters);
        var compiled = new CompiledGenomeEvaluator<int[]>(functions, Int32ArrayValueSemantics.Instance, counters: compiledCounters);
        using var device = new IlgpuGenomeEvaluator<int[]>(functions, null, selection, null, null, null);
        var paramsACopy = CloneJagged(genomeA.ParameterGenes.ToArray());
        var paramsBCopy = CloneJagged(genomeB.ParameterGenes.ToArray());

        var inputs1 = new[] { new[] { 1, 2 }, new[] { 3, 4 } };
        var inputs1Copy = inputs1.Select(a => (int[])a.Clone()).ToArray();
        var out1 = EvaluateArrayAndAssert($"{label} int[] A lote1", genomeA, inputs1, functions, reference, compiled, device, selection, ActiveA, InactiveA, refCounters, compiledCounters);
        for (var i = 0; i < inputs1.Length; i++) Assert.Equal(inputs1Copy[i], inputs1[i]);
        Assert.Equal(constantParamCopy, constantParam);
        AssertJaggedEqual(paramsACopy, genomeA.ParameterGenes.ToArray(), $"{label} int[] A lote1 ParameterGenes");

        var inputs2 = new[] { new[] { -5, 10 }, new[] { 2, -2 } };
        var out2 = EvaluateArrayAndAssert($"{label} int[] A lote2", genomeA, inputs2, functions, reference, compiled, device, selection, ActiveA, InactiveA, refCounters, compiledCounters);
        for (var i = 0; i < inputs1.Length; i++) Assert.Equal(inputs1Copy[i], inputs1[i]);
        Assert.Equal(constantParamCopy, constantParam);
        AssertJaggedEqual(paramsACopy, genomeA.ParameterGenes.ToArray(), $"{label} int[] A lote2 ParameterGenes");
        AssertArraySnapshotUnchanged($"{label} int[] A lote1 pós-lote2", out1, out1);

        var inputsB = new[] { new[] { 13, -6 }, new[] { 1, 8 } };
        var outB = EvaluateArrayAndAssert($"{label} int[] B", genomeB, inputsB, functions, reference, compiled, device, selection, ActiveB, InactiveB, refCounters, compiledCounters);
        Assert.Equal(constantParamCopy, constantParam);
        AssertJaggedEqual(paramsBCopy, genomeB.ParameterGenes.ToArray(), $"{label} int[] B ParameterGenes");
        AssertArraySnapshotUnchanged($"{label} int[] A lote1 pós-B", out1, out1);
        AssertArraySnapshotUnchanged($"{label} int[] A lote2 pós-B", out2, out2);

        var out1Again = EvaluateArrayAndAssert($"{label} int[] A retorno", genomeA, inputs1, functions, reference, compiled, device, selection, ActiveA, InactiveA, refCounters, compiledCounters);
        AssertJaggedEqual(out1.Reference, out1Again.Reference, $"{label} int[] A retorno reference");
        AssertJaggedEqual(out1.Compiled, out1Again.Compiled, $"{label} int[] A retorno compiled");
        AssertJaggedEqual(out1.Device, out1Again.Device, $"{label} int[] A retorno device");
        for (var i = 0; i < inputs1.Length; i++) Assert.Equal(inputs1Copy[i], inputs1[i]);
        Assert.Equal(constantParamCopy, constantParam);
        AssertJaggedEqual(paramsACopy, genomeA.ParameterGenes.ToArray(), $"{label} int[] A retorno ParameterGenes");
        AssertJaggedEqual(paramsBCopy, genomeB.ParameterGenes.ToArray(), $"{label} int[] B ParameterGenes pós-retorno");
    }
}
