using System.Collections.Concurrent;
using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.CPU;
using ILGPU.Runtime.OpenCL;
using Evo.Core;
using Evo.Core.Compiled;
using Evo.Core.Functions;
using Evo.Core.Values;
using Evo.Ilgpu;
using Xunit;
using Xunit.Abstractions;

namespace Evo.Ilgpu.Tests;

/// <summary>
/// Matriz de validação parametrizada por dispositivo (EVO-102): roda o mesmo conjunto de provas (paridade de
/// 4 tipos, zero com sinal em Min/Max, determinismo 1/2/N workers, cancelamento antes/durante e identidade de
/// dispositivo entre contextos) forçando explicitamente cada dispositivo enumerado pelo runtime ILGPU neste PC
/// (acelerador CPU do ILGPU e o dispositivo OpenCL real selecionado pelo processo, quando houver um).
/// </summary>
public sealed class IlgpuDeviceMatrixTests
{
    private readonly ITestOutputHelper _output;

    public IlgpuDeviceMatrixTests(ITestOutputHelper output) => _output = output;

    /// <summary>
    /// MemberData compartilhado com <see cref="OpenClDeviceCatalogTests"/>. Linhas indisponíveis carregam a causa
    /// e são reprovadas pela matriz, nunca aprovadas silenciosamente.
    /// </summary>
    public static IEnumerable<object[]> Devices() =>
        IlgpuDeviceCatalog.Entries()
            .Where(entry => entry[1] is not null && entry[2] is null)
            .Select(entry => new object[] { entry[0]!, entry[1]! });

    /// <summary>Dispositivos listados pelo catálogo sem seleção utilizável: reprovados com a causa, nunca aprovados.</summary>
    public static IEnumerable<object[]> UnavailableDevices() =>
        IlgpuDeviceCatalog.Entries()
            .Where(entry => entry[2] is not null)
            .Select(entry => new object[] { entry[0]!, entry[2]! });

    /// <summary>
    /// Indisponibilidade é resultado observável, não falha: cada linha indisponível carrega rótulo (identidade) e
    /// causa, e nunca entra no conjunto aprovado da matriz. A ausência de OpenCL neste PC não reprova o gate.
    /// </summary>
    [Fact]
    public void Unavailable_devices_are_reported_with_cause_and_never_approved()
    {
        foreach (var row in UnavailableDevices())
        {
            var label = Assert.IsType<string>(row[0]);
            var cause = Assert.IsType<string>(row[1]);
            Assert.False(string.IsNullOrWhiteSpace(label), "linha indisponível deve ter identidade");
            Assert.False(string.IsNullOrWhiteSpace(cause), $"{label} deve registrar a causa da indisponibilidade");
            Assert.DoesNotContain(Devices(), approved => (string)approved[0] == label);
            _output.WriteLine($"DISPOSITIVO INDISPONÍVEL | identidade={label} | status=INDISPONÍVEL (não aprovado) | causa={cause}");
        }
    }

    /// <summary>Somente-CPU/sem OpenCL: a linha OpenCL fica indisponível com causa e a CPU continua aprovada.</summary>
    [Fact]
    public void Cpu_only_host_reports_opencl_unavailable_with_cause_and_keeps_cpu_approved()
    {
        var cpuEntry = new IlgpuDeviceCatalog.CatalogEntry("ILGPU/CPU", CpuSelection(), null);
        var entries = new List<IlgpuDeviceCatalog.CatalogEntry>
        {
            cpuEntry,
        };
        entries.AddRange(IlgpuDeviceCatalog.BuildOpenClEntries([], _ => null));

        var approved = entries.Where(IlgpuDeviceCatalog.IsApproved).Select(e => e.Label).ToArray();
        var unavailable = entries.Where(e => !IlgpuDeviceCatalog.IsApproved(e)).ToArray();

        Assert.Equal(new[] { "ILGPU/CPU" }, approved);
        var openClRow = Assert.Single(unavailable);
        Assert.Contains("OpenCL", openClRow.Label);
        Assert.Equal("nenhum dispositivo OpenCL encontrado pelo runtime ILGPU", openClRow.UnavailableCause);
        Assert.Null(openClRow.Selection);
    }

    /// <summary>Candidato OpenCL inutilizável ao lado da CPU: causa visível com identidade, sem aprovação, CPU segue aprovada.</summary>
    [Fact]
    public void Unusable_opencl_candidate_reports_cause_identity_and_keeps_cpu_tests_running()
    {
        var unusable = new IlgpuBackendSelection(
            IlgpuBackendKind.OpenClDevice, 0, "GPU Falha", null, 0, 256, 4, 256, 0,
            PlatformName: "Plataforma Y", PlatformId: "p-1", OpenClDeviceId: "d-9");
        var entries = new List<IlgpuDeviceCatalog.CatalogEntry>
        {
            new("ILGPU/CPU", CpuSelection(), null),
        };
        entries.AddRange(IlgpuDeviceCatalog.BuildOpenClEntries([unusable], _ => new InvalidOperationException()));

        var approved = entries.Where(IlgpuDeviceCatalog.IsApproved).ToArray();
        var unavailable = entries.Where(e => !IlgpuDeviceCatalog.IsApproved(e)).ToArray();

        Assert.Equal(new[] { "ILGPU/CPU" }, approved.Select(e => e.Label));
        var row = Assert.Single(unavailable);
        Assert.Contains("Plataforma Y :: GPU Falha", row.Label);
        Assert.Contains("deviceId=d-9", row.Label);
        Assert.Contains("InvalidOperationException", row.UnavailableCause);
    }

    private static void AssertDeviceEvaluation<T>(IlgpuGenomeEvaluator<T> evaluator, Genome<T> genome, IFunctionSet<T> functions, IlgpuBackendSelection selection)
    {
        IlgpuTestHelpers.AssertRealKernelExecution(evaluator, genome, functions);
        Assert.Equal(selection.BackendName, evaluator.Backend);
        Assert.Equal(selection.DiagnosticDeviceName, evaluator.DeviceName);
    }

    private static IlgpuBackendSelection CpuSelection() =>
        new(IlgpuBackendKind.IlgpuCpu, 0, "CPU ILGPU", null, 8, 8, 1, 8, 4);

    /// <summary>
    /// Falha controlada (lançamento ou sincronização) depois da seleção do dispositivo: a exceção original é
    /// preservada, o backend/dispositivo continuam sendo os selecionados (sem fallback de sucesso), os buffers
    /// são descartados e Accelerator/Context são liberados pelo Dispose do avaliador.
    /// </summary>
    private static void AssertPostSelectionFailure<T>(
        IlgpuBackendSelection selection,
        IFunctionSet<T> functions,
        bool launchFailure,
        string label,
        Action<IlgpuGenomeEvaluator<T>> evaluate)
    {
        MemoryBuffer[]? observedBuffers = null;
        var forced = new InvalidOperationException($"falha-{(launchFailure ? "lancamento" : "sincronizacao")}-{label}");
        var evaluator = launchFailure
            ? new IlgpuGenomeEvaluator<T>(
                functions,
                fallback: null,
                selectionOverride: selection,
                contextFactory: null,
                acceleratorFactory: null,
                beforeKernelExecution: null,
                launchInvoker: _ => throw forced,
                synchronizeInvoker: null,
                bufferObserver: buffers => observedBuffers = buffers.ToArray())
            : new IlgpuGenomeEvaluator<T>(
                functions,
                fallback: null,
                selectionOverride: selection,
                contextFactory: null,
                acceleratorFactory: null,
                beforeKernelExecution: null,
                launchInvoker: null,
                synchronizeInvoker: _ => throw forced,
                bufferObserver: buffers => observedBuffers = buffers.ToArray());
        try
        {
            var error = Assert.Throws<InvalidOperationException>(() => evaluate(evaluator));
            Assert.Same(forced, error);
            Assert.NotNull(observedBuffers);
            Assert.All(observedBuffers!, buffer => Assert.True(buffer.IsDisposed));
            Assert.Equal(selection.BackendName, evaluator.Backend);
            Assert.Equal(selection.DiagnosticDeviceName, evaluator.DeviceName);
            Assert.NotEqual("CPU fallback", evaluator.Backend);

            var accelerator = evaluator.InternalAccelerator;
            var context = evaluator.InternalContext;
            Assert.NotNull(accelerator);
            Assert.NotNull(context);
            Assert.False(accelerator!.IsDisposed);
            Assert.False(context!.IsDisposed);

            evaluator.Dispose();
            Assert.True(accelerator.IsDisposed);
            Assert.True(context.IsDisposed);
        }
        finally
        {
            evaluator.Dispose();
        }
    }

    private static GenomeShape AddShape() =>
        new(InputCount: 2, NodeCount: 1, OutputCount: 1, MaxArity: 2, LevelsBack: 1);

    [Theory]
    [MemberData(nameof(Devices))]
    public void Post_selection_launch_and_sync_failures_preserve_identity_for_double(string label, IlgpuBackendSelection selection)
    {
        var functions = StandardDoubleFunctionSet.Instance;
        var genome = Genome<double>.FromGenes(AddShape(), [IlgpuTestHelpers.FindFunctionId(functions, "add")], [0, 1],
            [0.0], [2], DoubleValueSemantics.Instance, functions);
        AssertPostSelectionFailure(selection, functions, true, label, ev => ev.Evaluate(genome, [1.0, 2.0], new double[1]));
        AssertPostSelectionFailure(selection, functions, false, label, ev => ev.Evaluate(genome, [1.0, 2.0], new double[1]));
    }

    [Theory]
    [MemberData(nameof(Devices))]
    public void Post_selection_launch_and_sync_failures_preserve_identity_for_int(string label, IlgpuBackendSelection selection)
    {
        var functions = StandardInt32FunctionSet.Instance;
        var genome = Genome<int>.FromGenes(AddShape(), [IlgpuTestHelpers.FindFunctionId(functions, "add")], [0, 1],
            [0], [2], Int32ValueSemantics.Instance, functions);
        AssertPostSelectionFailure(selection, functions, true, label, ev => ev.Evaluate(genome, [1, 2], new int[1]));
        AssertPostSelectionFailure(selection, functions, false, label, ev => ev.Evaluate(genome, [1, 2], new int[1]));
    }

    [Theory]
    [MemberData(nameof(Devices))]
    public void Post_selection_launch_and_sync_failures_preserve_identity_for_double_array(string label, IlgpuBackendSelection selection)
    {
        var functions = StandardDoubleArrayFunctionSet.Instance;
        var genome = Genome<double[]>.FromGenes(AddShape(), [IlgpuTestHelpers.FindFunctionId(functions, "add")], [0, 1],
            [Array.Empty<double>()], [2], DoubleArrayValueSemantics.Instance, functions);
        double[][] inputs = [[1.0, 2.0], [3.0, 4.0]];
        AssertPostSelectionFailure(selection, functions, true, label, ev => ev.Evaluate(genome, inputs, new double[1][]));
        AssertPostSelectionFailure(selection, functions, false, label, ev => ev.Evaluate(genome, inputs, new double[1][]));
    }

    [Theory]
    [MemberData(nameof(Devices))]
    public void Post_selection_launch_and_sync_failures_preserve_identity_for_int_array(string label, IlgpuBackendSelection selection)
    {
        var functions = StandardInt32ArrayFunctionSet.Instance;
        var genome = Genome<int[]>.FromGenes(AddShape(), [IlgpuTestHelpers.FindFunctionId(functions, "add")], [0, 1],
            [Array.Empty<int>()], [2], Int32ArrayValueSemantics.Instance, functions);
        int[][] inputs = [[1, 2], [3, 4]];
        AssertPostSelectionFailure(selection, functions, true, label, ev => ev.Evaluate(genome, inputs, new int[1][]));
        AssertPostSelectionFailure(selection, functions, false, label, ev => ev.Evaluate(genome, inputs, new int[1][]));
    }

    private static Genome<double> DoubleScenario(IFunctionSet<double> functions)
    {
        var shape = new GenomeShape(InputCount: 2, NodeCount: 3, OutputCount: 2, MaxArity: 2, LevelsBack: 3);
        var addId = IlgpuTestHelpers.FindFunctionId(functions, "add");
        var divideId = IlgpuTestHelpers.FindFunctionId(functions, "divide");
        var greaterId = IlgpuTestHelpers.FindFunctionId(functions, "greater");
        return Genome<double>.FromGenes(
            shape,
            [addId, divideId, greaterId],
            [0, 1, 0, 1, 2, 3],
            [0.0, 0.0, 0.0],
            [2, 4],
            DoubleValueSemantics.Instance,
            functions);
    }

    private static Genome<int> IntScenario(IFunctionSet<int> functions)
    {
        var shape = new GenomeShape(InputCount: 2, NodeCount: 2, OutputCount: 1, MaxArity: 2, LevelsBack: 2);
        var addId = IlgpuTestHelpers.FindFunctionId(functions, "add");
        var absId = IlgpuTestHelpers.FindFunctionId(functions, "abs");
        return Genome<int>.FromGenes(shape, [addId, absId], [0, 1, 2, 0], [0, 0], [3], Int32ValueSemantics.Instance, functions);
    }

    private static Genome<double[]> DoubleArrayScenario(IFunctionSet<double[]> functions)
    {
        var shape = new GenomeShape(InputCount: 2, NodeCount: 1, OutputCount: 1, MaxArity: 2, LevelsBack: 1);
        var addId = IlgpuTestHelpers.FindFunctionId(functions, "add");
        return Genome<double[]>.FromGenes(shape, [addId], [0, 1], [Array.Empty<double>()], [2], DoubleArrayValueSemantics.Instance, functions);
    }

    private static Genome<int[]> IntArrayScenario(IFunctionSet<int[]> functions)
    {
        var shape = new GenomeShape(InputCount: 2, NodeCount: 1, OutputCount: 1, MaxArity: 2, LevelsBack: 1);
        var addId = IlgpuTestHelpers.FindFunctionId(functions, "add");
        return Genome<int[]>.FromGenes(shape, [addId], [0, 1], [Array.Empty<int>()], [2], Int32ArrayValueSemantics.Instance, functions);
    }

    [Theory]
    [MemberData(nameof(Devices))]
    public void Four_types_match_reference_on_device(string label, IlgpuBackendSelection selection)
    {
        Assert.NotNull(label);

        {
            var functions = StandardDoubleFunctionSet.Instance;
            var genome = DoubleScenario(functions);
            var inputs = new[] { 3.0, -0.5 };
            var expected = new double[2];
            var actual = new double[2];
            new ReferenceGenomeEvaluator<double>(functions).Evaluate(genome, inputs, expected);
            using var evaluator = new IlgpuGenomeEvaluator<double>(functions, null, selection, null, null, null);
            evaluator.Evaluate(genome, inputs, actual);
            IlgpuTestHelpers.AssertRealKernelExecution(evaluator, genome, functions);
            Assert.Equal(selection.BackendName, evaluator.Backend);
            Assert.Equal(selection.DiagnosticDeviceName, evaluator.DeviceName);
            IlgpuTestHelpers.AssertDoubleArrayClose(expected, actual, $"{label} double");
        }

        {
            var functions = StandardInt32FunctionSet.Instance;
            var genome = IntScenario(functions);
            var inputs = new[] { -4, 7 };
            var expected = new int[1];
            var actual = new int[1];
            new ReferenceGenomeEvaluator<int>(functions).Evaluate(genome, inputs, expected);
            using var evaluator = new IlgpuGenomeEvaluator<int>(functions, null, selection, null, null, null);
            evaluator.Evaluate(genome, inputs, actual);
            AssertDeviceEvaluation(evaluator, genome, functions, selection);
            Assert.Equal(expected, actual);
        }

        {
            var functions = StandardDoubleArrayFunctionSet.Instance;
            var genome = DoubleArrayScenario(functions);
            var inputs = new[] { new[] { 1.0, 2.0 }, new[] { 3.0, 4.0 } };
            var expected = new double[1][];
            var actual = new double[1][];
            new ReferenceGenomeEvaluator<double[]>(functions).Evaluate(genome, inputs, expected);
            using var evaluator = new IlgpuGenomeEvaluator<double[]>(functions, null, selection, null, null, null);
            evaluator.Evaluate(genome, inputs, actual);
            AssertDeviceEvaluation(evaluator, genome, functions, selection);
            Assert.Equal(expected[0], actual[0]);
        }

        {
            var functions = StandardInt32ArrayFunctionSet.Instance;
            var genome = IntArrayScenario(functions);
            var inputs = new[] { new[] { 1, 2 }, new[] { 3, 4 } };
            var expected = new int[1][];
            var actual = new int[1][];
            new ReferenceGenomeEvaluator<int[]>(functions).Evaluate(genome, inputs, expected);
            using var evaluator = new IlgpuGenomeEvaluator<int[]>(functions, null, selection, null, null, null);
            evaluator.Evaluate(genome, inputs, actual);
            AssertDeviceEvaluation(evaluator, genome, functions, selection);
            Assert.Equal(expected[0], actual[0]);
        }
    }

    /// <summary>
    /// Compiled CPU é um caminho distinto da referência: ambos devem produzir a mesma saída para os 4 tipos,
    /// e o ILGPU do dispositivo deve igualar as duas.
    /// </summary>
    [Theory]
    [MemberData(nameof(Devices))]
    public void Compiled_cpu_and_device_match_reference_for_four_types(string label, IlgpuBackendSelection selection)
    {
        {
            var functions = StandardDoubleFunctionSet.Instance;
            var genome = DoubleScenario(functions);
            var inputs = new[] { 3.0, -0.5 };
            var expected = new double[2];
            var compiled = new double[2];
            var device = new double[2];
            new ReferenceGenomeEvaluator<double>(functions).Evaluate(genome, inputs, expected);
            new CompiledGenomeEvaluator<double>(functions, DoubleValueSemantics.Instance).Evaluate(genome, inputs, compiled);
            using var evaluator = new IlgpuGenomeEvaluator<double>(functions, null, selection, null, null, null);
            evaluator.Evaluate(genome, inputs, device);
            AssertDeviceEvaluation(evaluator, genome, functions, selection);
            IlgpuTestHelpers.AssertDoubleArrayClose(expected, compiled, $"{label} compiled double");
            IlgpuTestHelpers.AssertDoubleArrayClose(expected, device, $"{label} device double");
        }

        {
            var functions = StandardInt32FunctionSet.Instance;
            var genome = IntScenario(functions);
            var inputs = new[] { -4, 7 };
            var expected = new int[1];
            var compiled = new int[1];
            var device = new int[1];
            new ReferenceGenomeEvaluator<int>(functions).Evaluate(genome, inputs, expected);
            new CompiledGenomeEvaluator<int>(functions, Int32ValueSemantics.Instance).Evaluate(genome, inputs, compiled);
            using var evaluator = new IlgpuGenomeEvaluator<int>(functions, null, selection, null, null, null);
            evaluator.Evaluate(genome, inputs, device);
            AssertDeviceEvaluation(evaluator, genome, functions, selection);
            Assert.Equal(expected, compiled);
            Assert.Equal(expected, device);
        }

        {
            var functions = StandardDoubleArrayFunctionSet.Instance;
            var genome = DoubleArrayScenario(functions);
            var inputs = new[] { new[] { 1.0, 2.0 }, new[] { 3.0, 4.0 } };
            var expected = new double[1][];
            var compiled = new double[1][];
            var device = new double[1][];
            new ReferenceGenomeEvaluator<double[]>(functions).Evaluate(genome, inputs, expected);
            new CompiledGenomeEvaluator<double[]>(functions, DoubleArrayValueSemantics.Instance).Evaluate(genome, inputs, compiled);
            using var evaluator = new IlgpuGenomeEvaluator<double[]>(functions, null, selection, null, null, null);
            evaluator.Evaluate(genome, inputs, device);
            AssertDeviceEvaluation(evaluator, genome, functions, selection);
            Assert.Equal(expected[0], compiled[0]);
            Assert.Equal(expected[0], device[0]);
        }

        {
            var functions = StandardInt32ArrayFunctionSet.Instance;
            var genome = IntArrayScenario(functions);
            var inputs = new[] { new[] { 1, 2 }, new[] { 3, 4 } };
            var expected = new int[1][];
            var compiled = new int[1][];
            var device = new int[1][];
            new ReferenceGenomeEvaluator<int[]>(functions).Evaluate(genome, inputs, expected);
            new CompiledGenomeEvaluator<int[]>(functions, Int32ArrayValueSemantics.Instance).Evaluate(genome, inputs, compiled);
            using var evaluator = new IlgpuGenomeEvaluator<int[]>(functions, null, selection, null, null, null);
            evaluator.Evaluate(genome, inputs, device);
            AssertDeviceEvaluation(evaluator, genome, functions, selection);
            Assert.Equal(expected[0], compiled[0]);
            Assert.Equal(expected[0], device[0]);
        }
    }

    [Theory]
    [MemberData(nameof(Devices))]
    public void Signed_zero_min_max_matches_bit_for_bit_on_device(string label, IlgpuBackendSelection selection)
    {
        var functions = StandardDoubleFunctionSet.Instance;
        var minId = IlgpuTestHelpers.FindFunctionId(functions, "min");
        var maxId = IlgpuTestHelpers.FindFunctionId(functions, "max");
        var shape = new GenomeShape(InputCount: 2, NodeCount: 4, OutputCount: 4, MaxArity: 2, LevelsBack: 4);
        var genome = Genome<double>.FromGenes(
            shape,
            [minId, maxId, minId, maxId],
            [0, 1, 1, 0, 1, 0, 0, 1],
            [0.0, 0.0, 0.0, 0.0],
            [2, 3, 4, 5],
            DoubleValueSemantics.Instance,
            functions);
        var signedZeroPairs = new[]
        {
            (0.0, 0.0),
            (0.0, -0.0),
            (-0.0, 0.0),
            (-0.0, -0.0),
        };
        using var evaluator = new IlgpuGenomeEvaluator<double>(functions, null, selection, null, null, null);
        foreach (var (a, b) in signedZeroPairs)
        {
            var inputs = new[] { a, b };
            var expected = new double[4];
            var compiled = new double[4];
            var actual = new double[4];
            new ReferenceGenomeEvaluator<double>(functions).Evaluate(genome, inputs, expected);
            new CompiledGenomeEvaluator<double>(functions, DoubleValueSemantics.Instance).Evaluate(genome, inputs, compiled);
            evaluator.Evaluate(genome, inputs, actual);
            IlgpuTestHelpers.AssertRealKernelExecution(evaluator, genome, functions);
            Assert.Equal(selection.BackendName, evaluator.Backend);
            Assert.Equal(selection.DiagnosticDeviceName, evaluator.DeviceName);

            var pairTag = $"{label} ({Sign(a)},{Sign(b)})";
            for (var i = 0; i < 4; i++)
            {
                var e = BitConverter.DoubleToInt64Bits(expected[i]);
                Assert.True(
                    e == BitConverter.DoubleToInt64Bits(compiled[i]),
                    $"{pairTag} compiled saída {i}: esperado bits {e:x}, atual {BitConverter.DoubleToInt64Bits(compiled[i]):x}");
                Assert.True(
                    e == BitConverter.DoubleToInt64Bits(actual[i]),
                    $"{pairTag} dispositivo saída {i}: esperado bits {e:x}, atual {BitConverter.DoubleToInt64Bits(actual[i]):x}");
            }
        }
    }

    private static string Sign(double value) => double.IsNegative(value) ? "-0" : "+0";

    /// <summary>
    /// Zero com sinal em Min/Max para double[] (elemento a elemento), nos três caminhos: referência, compiled CPU
    /// distinto e dispositivo. Comparação bit a bit, sem tolerância.
    /// </summary>
    [Theory]
    [MemberData(nameof(Devices))]
    public void Signed_zero_min_max_arrays_match_bit_for_bit_on_device(string label, IlgpuBackendSelection selection)
    {
        var functions = StandardDoubleArrayFunctionSet.Instance;
        var minId = IlgpuTestHelpers.FindFunctionId(functions, "min");
        var maxId = IlgpuTestHelpers.FindFunctionId(functions, "max");
        var shape = new GenomeShape(InputCount: 2, NodeCount: 4, OutputCount: 4, MaxArity: 2, LevelsBack: 4);
        var genome = Genome<double[]>.FromGenes(
            shape,
            [minId, maxId, minId, maxId],
            [0, 1, 1, 0, 1, 0, 0, 1],
            [Array.Empty<double>(), Array.Empty<double>(), Array.Empty<double>(), Array.Empty<double>()],
            [2, 3, 4, 5],
            DoubleArrayValueSemantics.Instance,
            functions);
        var signedZeroPairs = new[]
        {
            (0.0, 0.0),
            (0.0, -0.0),
            (-0.0, 0.0),
            (-0.0, -0.0),
        };
        using var evaluator = new IlgpuGenomeEvaluator<double[]>(functions, null, selection, null, null, null);
        foreach (var (a, b) in signedZeroPairs)
        {
            var inputs = new[] { new[] { a, a }, new[] { b, b } };
            var expected = new double[4][];
            var compiled = new double[4][];
            var actual = new double[4][];
            new ReferenceGenomeEvaluator<double[]>(functions).Evaluate(genome, inputs, expected);
            new CompiledGenomeEvaluator<double[]>(functions, DoubleArrayValueSemantics.Instance).Evaluate(genome, inputs, compiled);
            evaluator.Evaluate(genome, inputs, actual);
            AssertDeviceEvaluation(evaluator, genome, functions, selection);

            var pairTag = $"{label} ({Sign(a)},{Sign(b)})";
        for (var i = 0; i < 4; i++)
        {
            Assert.Equal(expected[i].Length, compiled[i].Length);
            Assert.Equal(expected[i].Length, actual[i].Length);
            for (var j = 0; j < expected[i].Length; j++)
            {
                var e = BitConverter.DoubleToInt64Bits(expected[i][j]);
                Assert.True(e == BitConverter.DoubleToInt64Bits(compiled[i][j]), $"{pairTag} compiled saída {i}[{j}]");
                Assert.True(e == BitConverter.DoubleToInt64Bits(actual[i][j]), $"{pairTag} dispositivo saída {i}[{j}]");
            }
        }
        }
    }

    /// <summary>
    /// Carga fixa e heterogênea (genomas e entradas distintos) usada para provar ordem real: cada caso só é
    /// considerado correto se o resultado no índice correspondente bater com a referência daquele caso
    /// específico, não apenas se a contagem de itens processados bater.
    /// </summary>
    private static (Genome<double> Genome, double[] Inputs)[] HeterogeneousDoubleCases(IFunctionSet<double> functions)
    {
        var genome = DoubleScenario(functions);
        return
        [
            (genome, [1.25, -3.5]),
            (genome, [-0.0, 0.0]),
            (genome, [7.5, 2.0]),
            (genome, [-9.0, -1.5]),
            (genome, [0.001, 1000.0]),
            (genome, [-4.75, 6.25]),
        ];
    }

    private static void ShuffleDeterministic<T>(T[] items, int seed)
    {
        var rng = new Random(seed);
        for (var i = items.Length - 1; i > 0; i--)
        {
            var j = rng.Next(i + 1);
            (items[i], items[j]) = (items[j], items[i]);
        }
    }

    [Theory]
    [MemberData(nameof(Devices))]
    public void Worker_counts_1_2_n_produce_bitwise_identical_results_and_order_on_device(string label, IlgpuBackendSelection selection)
    {
        Assert.False(string.IsNullOrEmpty(label), "rótulo do dispositivo deve identificar o backend sob teste");
        var functions = StandardDoubleFunctionSet.Instance;
        var cases = HeterogeneousDoubleCases(functions);
        var reference = new double[cases.Length][];
        for (var i = 0; i < cases.Length; i++)
        {
            reference[i] = new double[2];
            new ReferenceGenomeEvaluator<double>(functions).Evaluate(cases[i].Genome, cases[i].Inputs, reference[i]);
        }

        double[][]? baseline = null;
        foreach (var workerCount in new[] { 1, 2, 4 })
        {
            var results = new double[cases.Length][];
            var callsPerCase = new int[cases.Length];
            var threadIds = new ConcurrentBag<int>();

            // Varia o agendamento/particionamento entre rodadas embaralhando a ordem de processamento dos
            // índices, mantendo a carga (genomas/casos) fixa e heterogênea.
            var indices = Enumerable.Range(0, cases.Length).ToArray();
            ShuffleDeterministic(indices, seed: workerCount);

            Parallel.ForEach(
                indices,
                new ParallelOptions { MaxDegreeOfParallelism = workerCount },
                idx =>
                {
                    using var evaluator = new IlgpuGenomeEvaluator<double>(functions, null, selection, null, null, null);
                    var output = new double[2];
                    evaluator.Evaluate(cases[idx].Genome, cases[idx].Inputs, output);
                    results[idx] = output;
                    Interlocked.Increment(ref callsPerCase[idx]);
                    threadIds.Add(Environment.CurrentManagedThreadId);
                });

            // Prova de ordem real por índice: cada caso foi avaliado exatamente uma vez e o resultado no slot
            // correto corresponde à referência daquele caso especifico (tolerância normativa, não bitwise,
            // contra a referência independente).
            for (var i = 0; i < cases.Length; i++)
            {
                Assert.Equal(1, callsPerCase[i]);
                Assert.NotNull(results[i]);
                IlgpuTestHelpers.AssertDoubleArrayClose(reference[i], results[i]!, $"{label} workers={workerCount} caso={i} vs referência");
            }

            // Identidade efetiva: com mais de um worker configurado, threads foram de fato utilizadas para o
            // processamento (não apenas um único caminho serial disfarçado).
            Assert.True(threadIds.Count >= 1 && threadIds.Count <= cases.Length);

            // Comparação bit a bit entre contagens de workers do mesmo backend/dispositivo: variar o número de
            // workers e o agendamento não pode alterar o resultado numérico exato produzido pelo mesmo backend.
            if (baseline is null)
            {
                baseline = results;
            }
            else
            {
                for (var i = 0; i < cases.Length; i++)
                {
                    Assert.Equal(BitConverter.DoubleToInt64Bits(baseline[i][0]), BitConverter.DoubleToInt64Bits(results[i][0]));
                    Assert.Equal(BitConverter.DoubleToInt64Bits(baseline[i][1]), BitConverter.DoubleToInt64Bits(results[i][1]));
                }
            }
        }
    }

    [Theory]
    [MemberData(nameof(Devices))]
    public void Cancellation_before_and_during_evaluation_releases_resources_on_device(string label, IlgpuBackendSelection selection)
    {
        var functions = StandardDoubleFunctionSet.Instance;
        var genome = DoubleScenario(functions);

        // Cancelamento antes da avaliação: nenhuma chamada ao avaliador pode ocorrer e os recursos são descartados.
        ILGPU.Runtime.Accelerator? beforeAccelerator;
        Context? beforeContext;
        using (var evaluator = new IlgpuGenomeEvaluator<double>(functions, null, selection, null, null, null))
        {
            using var beforeCts = new CancellationTokenSource();
            beforeCts.Cancel();
            var fitness = new CancellingFitness(beforeCts.Token, cancelOnCall: 0);
            beforeAccelerator = evaluator.InternalAccelerator;
            beforeContext = evaluator.InternalContext;
            Assert.Throws<OperationCanceledException>(() => fitness.Evaluate(genome, evaluator, beforeCts.Token));
            Assert.Equal(0, fitness.Calls);
        }

        Assert.True(beforeAccelerator is null || beforeAccelerator.IsDisposed, $"{label}: accelerator não descartado após cancelamento antes");
        Assert.True(beforeContext is null || beforeContext.IsDisposed, $"{label}: context não descartado após cancelamento antes");

        // Cancelamento durante a avaliação: exatamente uma chamada real ao avaliador ocorre antes do cancelamento.
        ILGPU.Runtime.Accelerator? duringAccelerator;
        Context? duringContext;
        using (var evaluator = new IlgpuGenomeEvaluator<double>(functions, null, selection, null, null, null))
        {
            using var duringCts = new CancellationTokenSource();
            var fitness = new CancellingFitness(duringCts.Token, cancelOnCall: 1, cancelBeforeCall: duringCts.Cancel);
            duringAccelerator = evaluator.InternalAccelerator;
            duringContext = evaluator.InternalContext;
            Assert.Throws<OperationCanceledException>(() => fitness.Evaluate(genome, evaluator, duringCts.Token));
            Assert.Equal(1, fitness.Calls);
        }

        Assert.True(duringAccelerator is null || duringAccelerator.IsDisposed, $"{label}: accelerator não descartado após cancelamento durante");
        Assert.True(duringContext is null || duringContext.IsDisposed, $"{label}: context não descartado após cancelamento durante");

        // Um worker novo com a mesma seleção precisa continuar funcional depois do cancelamento: os recursos
        // do worker cancelado foram liberados (Dispose) sem deixar o acelerador/contexto compartilhado inválido.
        using var healthy = new IlgpuGenomeEvaluator<double>(functions, null, selection, null, null, null);
        var inputs = new[] { 2.0, 1.0 };
        var expected = new double[2];
        var actual = new double[2];
        new ReferenceGenomeEvaluator<double>(functions).Evaluate(genome, inputs, expected);
        healthy.Evaluate(genome, inputs, actual);
        IlgpuTestHelpers.AssertDoubleArrayClose(expected, actual, $"{label} pós-cancelamento");
    }

    private sealed class CancellingFitness : IFitnessEvaluator<double>
    {
        private readonly CancellationToken _token;
        private readonly int _cancelOnCall;
        private readonly Action? _cancelBeforeCall;
        private int _calls;

        public CancellingFitness(CancellationToken token, int cancelOnCall, Action? cancelBeforeCall = null)
        {
            _token = token;
            _cancelOnCall = cancelOnCall;
            _cancelBeforeCall = cancelBeforeCall;
        }

        public int Calls => _calls;

        public FitnessResult Evaluate(Genome<double> genome, IGenomeEvaluator<double> evaluator, CancellationToken cancellationToken)
        {
            cancellationToken.ThrowIfCancellationRequested();
            var output = new double[2];
            for (var i = 0; i < 3; i++)
            {
                if (_calls == _cancelOnCall)
                {
                    _cancelBeforeCall?.Invoke();
                }

                cancellationToken.ThrowIfCancellationRequested();
                evaluator.Evaluate(genome, [1.0, 2.0], output);
                _calls++;
            }

            return FitnessResult.Valid(output[0]);
        }
    }

    [Theory]
    [MemberData(nameof(Devices))]
    public void Cancellation_before_and_during_releases_resources_for_int_on_device(string label, IlgpuBackendSelection selection)
    {
        var functions = StandardInt32FunctionSet.Instance;
        AssertCancellationReleasesResources(label, selection, functions, IntScenario(functions), [-4, 7], 1,
            (expected, actual, message) => Assert.Equal(expected, actual));
    }

    [Theory]
    [MemberData(nameof(Devices))]
    public void Cancellation_before_and_during_releases_resources_for_double_array_on_device(string label, IlgpuBackendSelection selection)
    {
        var functions = StandardDoubleArrayFunctionSet.Instance;
        var inputs = new[] { new[] { 1.0, 2.0 }, new[] { 3.0, 4.0 } };
        AssertCancellationReleasesResources(label, selection, functions, DoubleArrayScenario(functions), inputs, 1,
            (expected, actual, message) => Assert.Equal(expected[0], actual[0]));
    }

    [Theory]
    [MemberData(nameof(Devices))]
    public void Cancellation_before_and_during_releases_resources_for_int_array_on_device(string label, IlgpuBackendSelection selection)
    {
        var functions = StandardInt32ArrayFunctionSet.Instance;
        var inputs = new[] { new[] { 1, 2 }, new[] { 3, 4 } };
        AssertCancellationReleasesResources(label, selection, functions, IntArrayScenario(functions), inputs, 1,
            (expected, actual, message) => Assert.Equal(expected[0], actual[0]));
    }

    /// <summary>
    /// Cancelamento antes e durante a avaliação para um tipo genérico: zero chamadas antes; exatamente uma chamada
    /// real de kernel antes do cancelamento durante; descarte de Accelerator/Context após limpeza do chamador;
    /// worker novo continua funcional com identidade efetiva do backend.
    /// </summary>
    private static void AssertCancellationReleasesResources<T>(
        string label,
        IlgpuBackendSelection selection,
        IFunctionSet<T> functions,
        Genome<T> genome,
        T[] inputs,
        int outputCount,
        Action<T[], T[], string> compare)
    {
        ILGPU.Runtime.Accelerator? beforeAccelerator;
        Context? beforeContext;
        using (var evaluator = new IlgpuGenomeEvaluator<T>(functions, null, selection, null, null, null))
        {
            using var beforeCts = new CancellationTokenSource();
            beforeCts.Cancel();
            var fitness = new CancellingFitness<T>(beforeCts.Token, cancelOnCall: 0, inputs, outputCount);
            beforeAccelerator = evaluator.InternalAccelerator;
            beforeContext = evaluator.InternalContext;
            Assert.Throws<OperationCanceledException>(() => fitness.Evaluate(genome, evaluator, beforeCts.Token));
            Assert.Equal(0, fitness.Calls);
        }

        Assert.True(beforeAccelerator is null || beforeAccelerator.IsDisposed, $"{label}: accelerator não descartado após cancelamento antes");
        Assert.True(beforeContext is null || beforeContext.IsDisposed, $"{label}: context não descartado após cancelamento antes");

        ILGPU.Runtime.Accelerator? duringAccelerator;
        Context? duringContext;
        using (var evaluator = new IlgpuGenomeEvaluator<T>(functions, null, selection, null, null, null))
        {
            using var duringCts = new CancellationTokenSource();
            var fitness = new CancellingFitness<T>(duringCts.Token, cancelOnCall: 1, inputs, outputCount, duringCts.Cancel);
            duringAccelerator = evaluator.InternalAccelerator;
            duringContext = evaluator.InternalContext;
            Assert.Throws<OperationCanceledException>(() => fitness.Evaluate(genome, evaluator, duringCts.Token));
            Assert.Equal(1, fitness.Calls);
            IlgpuTestHelpers.AssertRealKernelExecution(evaluator, genome, functions);
            Assert.Equal(selection.BackendName, evaluator.Backend);
            Assert.Equal(selection.DiagnosticDeviceName, evaluator.DeviceName);
        }

        Assert.True(duringAccelerator is null || duringAccelerator.IsDisposed, $"{label}: accelerator não descartado após cancelamento durante");
        Assert.True(duringContext is null || duringContext.IsDisposed, $"{label}: context não descartado após cancelamento durante");

        using var healthy = new IlgpuGenomeEvaluator<T>(functions, null, selection, null, null, null);
        var expected = new T[outputCount];
        var actual = new T[outputCount];
        new ReferenceGenomeEvaluator<T>(functions).Evaluate(genome, inputs, expected);
        healthy.Evaluate(genome, inputs, actual);
        AssertDeviceEvaluation(healthy, genome, functions, selection);
        compare(expected, actual, $"{label} pós-cancelamento");
    }

    private sealed class CancellingFitness<T> : IFitnessEvaluator<T>
    {
        private readonly CancellationToken _token;
        private readonly int _cancelOnCall;
        private readonly T[] _inputs;
        private readonly int _outputCount;
        private readonly Action? _cancelBeforeCall;
        private int _calls;

        public CancellingFitness(CancellationToken token, int cancelOnCall, T[] inputs, int outputCount, Action? cancelBeforeCall = null)
        {
            _token = token;
            _cancelOnCall = cancelOnCall;
            _inputs = inputs;
            _outputCount = outputCount;
            _cancelBeforeCall = cancelBeforeCall;
        }

        public int Calls => _calls;

        public FitnessResult Evaluate(Genome<T> genome, IGenomeEvaluator<T> evaluator, CancellationToken cancellationToken)
        {
            cancellationToken.ThrowIfCancellationRequested();
            var output = new T[_outputCount];
            for (var i = 0; i < 3; i++)
            {
                if (_calls == _cancelOnCall)
                {
                    _cancelBeforeCall?.Invoke();
                }

                cancellationToken.ThrowIfCancellationRequested();
                evaluator.Evaluate(genome, _inputs, output);
                _calls++;
            }

            return FitnessResult.Valid(0.0);
        }
    }

    [Theory]
    [MemberData(nameof(Devices))]
    public void Device_identity_is_stable_across_independent_contexts(string label, IlgpuBackendSelection selection)
    {
        Assert.False(string.IsNullOrEmpty(label), "rótulo do dispositivo deve identificar o backend sob teste");
        using var first = new IlgpuGenomeEvaluator<double>(StandardDoubleFunctionSet.Instance, null, selection, null, null, null);
        using var second = new IlgpuGenomeEvaluator<double>(StandardDoubleFunctionSet.Instance, null, selection, null, null, null);
        Assert.Equal(first.Backend, second.Backend);
        Assert.Equal(first.DeviceName, second.DeviceName);
        Assert.StartsWith("ILGPU", first.Backend);
    }
}
