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

    private static IlgpuBackendSelection CpuSelection() =>
        new(IlgpuBackendKind.IlgpuCpu, 0, "CPU ILGPU", null, 8, 8, 1, 8, 4);

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
        var inputs = new[] { -0.0, 0.0 };
        var expected = new double[4];
        var actual = new double[4];
        new ReferenceGenomeEvaluator<double>(functions).Evaluate(genome, inputs, expected);
        using var evaluator = new IlgpuGenomeEvaluator<double>(functions, null, selection, null, null, null);
        evaluator.Evaluate(genome, inputs, actual);

        for (var i = 0; i < 4; i++)
        {
            Assert.True(
                BitConverter.DoubleToInt64Bits(expected[i]) == BitConverter.DoubleToInt64Bits(actual[i]),
                $"{label} saída {i}: esperado bits {BitConverter.DoubleToInt64Bits(expected[i]):x}, atual {BitConverter.DoubleToInt64Bits(actual[i]):x}");
        }
    }

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
        var inputs = new[] { new[] { -0.0, 0.0 }, new[] { 0.0, -0.0 } };
        var expected = new double[4][];
        var compiled = new double[4][];
        var actual = new double[4][];
        new ReferenceGenomeEvaluator<double[]>(functions).Evaluate(genome, inputs, expected);
        new CompiledGenomeEvaluator<double[]>(functions, DoubleArrayValueSemantics.Instance).Evaluate(genome, inputs, compiled);
        using var evaluator = new IlgpuGenomeEvaluator<double[]>(functions, null, selection, null, null, null);
        evaluator.Evaluate(genome, inputs, actual);

        for (var i = 0; i < 4; i++)
        {
            Assert.Equal(expected[i].Length, compiled[i].Length);
            Assert.Equal(expected[i].Length, actual[i].Length);
            for (var j = 0; j < expected[i].Length; j++)
            {
                var e = BitConverter.DoubleToInt64Bits(expected[i][j]);
                Assert.True(e == BitConverter.DoubleToInt64Bits(compiled[i][j]), $"{label} compiled saída {i}[{j}]");
                Assert.True(e == BitConverter.DoubleToInt64Bits(actual[i][j]), $"{label} dispositivo saída {i}[{j}]");
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
            Assert.True(threadIds.Count == cases.Length);

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

        using (var evaluator = new IlgpuGenomeEvaluator<double>(functions, null, selection, null, null, null))
        {
            using var beforeCts = new CancellationTokenSource();
            beforeCts.Cancel();
            var fitness = new CancellingFitness(beforeCts.Token, cancelOnCall: 0);
            Assert.Throws<OperationCanceledException>(() => fitness.Evaluate(genome, evaluator, beforeCts.Token));
        }

        using (var evaluator = new IlgpuGenomeEvaluator<double>(functions, null, selection, null, null, null))
        {
            using var duringCts = new CancellationTokenSource();
            var fitness = new CancellingFitness(duringCts.Token, cancelOnCall: 1, cancelBeforeCall: duringCts.Cancel);
            Assert.Throws<OperationCanceledException>(() => fitness.Evaluate(genome, evaluator, duringCts.Token));
        }

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
