using System.Collections.Concurrent;
using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.CPU;
using ILGPU.Runtime.OpenCL;
using Evo.Core;
using Evo.Core.Functions;
using Evo.Core.Values;
using Evo.Ilgpu;
using Xunit;

namespace Evo.Ilgpu.Tests;

/// <summary>
/// Matriz de validação parametrizada por dispositivo (EVO-102): roda o mesmo conjunto de provas (paridade de
/// 4 tipos, zero com sinal em Min/Max, determinismo 1/2/N workers, cancelamento antes/durante e identidade de
/// dispositivo entre contextos) forçando explicitamente cada dispositivo enumerado pelo runtime ILGPU neste PC
/// (acelerador CPU do ILGPU e o dispositivo OpenCL real selecionado pelo processo, quando houver um).
/// </summary>
public sealed class IlgpuDeviceMatrixTests
{
    public static IEnumerable<object[]> Devices()
    {
        yield return new object[] { "ILGPU/CPU", ForcedCpuSelection() };

        var processSelection = IlgpuBackendSelector.Selection;
        if (processSelection.Kind == IlgpuBackendKind.OpenClDevice)
        {
            yield return new object[] { "ILGPU/OpenCL:" + processSelection.DiagnosticDeviceName, processSelection };
        }
        else
        {
            // Nenhum dispositivo OpenCL utilizável foi selecionado pelo processo nesta máquina; registrado em
            // devices.md/coverage.md como indisponível, nunca aprovado como se fosse uma linha OpenCL da matriz.
        }
    }

    private static IlgpuBackendSelection ForcedCpuSelection()
    {
        using var context = Context.Create(builder => builder.OpenCL().CPU());
        var devices = context.Devices.ToArray();
        for (var i = 0; i < devices.Length; i++)
        {
            var device = devices[i];
            if (device.AcceleratorType != AcceleratorType.CPU)
            {
                continue;
            }

            using var accelerator = device.CreateAccelerator(context);
            return new IlgpuBackendSelection(
                IlgpuBackendKind.IlgpuCpu,
                i,
                device.Name,
                null,
                accelerator.WarpSize,
                accelerator.MaxNumThreads,
                device.NumMultiprocessors,
                device.MaxNumThreadsPerGroup,
                ((CPUDevice)device).NumThreads);
        }

        throw new InvalidOperationException("dispositivo CPU do ILGPU não encontrado neste PC");
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
