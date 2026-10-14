using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Runtime.CompilerServices;
using System.Threading;
using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.CPU;
using ILGPU.Runtime.Cuda;
using ILGPU.Runtime.OpenCL;
using Evo.Core;
using Evo.Core.Compiled;
using Evo.Core.Functions;
using Evo.Core.Values;
using Evo.Ilgpu;

[assembly: InternalsVisibleTo("Evo.Ilgpu.Tests")]

namespace Evo.Benchmarks;

/// <summary>
/// Reparo R1-6 (EVO-102 / 010-ilgpu-validation-v2): mede, para os quatro tipos oficiais, o CPU compilado
/// (<see cref="CompiledGenomeEvaluator{T}"/>), o ILGPU CPU forçado e cada dispositivo OpenCL utilizável — não
/// apenas o backend selecionado pelo processo (<see cref="IlgpuBackendSelector.Selection"/>). Cada candidato é
/// comparado à referência antes da medição; candidatos inutilizáveis são relatados como INDISPONÍVEL com a causa
/// e nunca entram na medição.
/// </summary>
internal static class IlgpuBenchmarkProgram
{
    private const int WarmupRepetitions = 100;
    private const int MeasureRepetitions = 1000;
    private const int BatchSize = 32;
    private const int BatchWarmupRepetitions = 10;
    private const int BatchMeasureRepetitions = 100;
    private const int TransferLength = 64;
    private const int TransferRepetitions = 1000;

    private static readonly Lazy<string> CudaDriver = new(QueryCudaDriverVersion);

    public static int Run()
    {
        Console.WriteLine("Evo.Benchmarks — benchmark do backend ILGPU/OpenCL (EVO-101/EVO-102)");
        Console.WriteLine($"runtime={RuntimeInformation.FrameworkDescription} arquitetura={RuntimeInformation.ProcessArchitecture}");
        Console.WriteLine($"warmup={WarmupRepetitions} repeticoes={MeasureRepetitions} batchSize={BatchSize} batchWarmup={BatchWarmupRepetitions} batchRepeticoes={BatchMeasureRepetitions}");
        Console.WriteLine($"driverCuda(nvidia-smi)={CudaDriver.Value}");

        var candidates = EnumerateDeviceCandidates();
        Console.WriteLine($"dispositivos ILGPU candidatos detectados: {candidates.Count}");
        foreach (var candidate in candidates)
        {
            Console.WriteLine($"  candidato: {DescribeIdentity(candidate)}");
        }

        RunDouble(candidates);
        RunInt32(candidates);
        RunDoubleArray(candidates);
        RunInt32Array(candidates);
        RunTransferProbe(candidates);
        return 0;
    }

    /// <summary>
    /// Consulta a versão do driver NVIDIA via <c>nvidia-smi</c> (somente leitura, sem rede). Retorna
    /// "indisponivel" quando a ferramenta não existe ou falha; nunca lança exceção.
    /// </summary>
    private static string QueryCudaDriverVersion()
    {
        try
        {
            using var process = Process.Start(new ProcessStartInfo(
                "nvidia-smi",
                "--query-gpu=driver_version,name --format=csv,noheader")
            {
                RedirectStandardOutput = true,
                UseShellExecute = false,
                CreateNoWindow = true,
            });
            if (process is null)
            {
                return "indisponivel";
            }

            var output = process.StandardOutput.ReadToEnd();
            process.WaitForExit(5000);
            var firstLine = output.Split('\n', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries).FirstOrDefault();
            return string.IsNullOrEmpty(firstLine) ? "indisponivel" : firstLine;
        }
        catch (Exception ex) when (ex is not OutOfMemoryException)
        {
            return $"indisponivel: {ex.GetType().Name}";
        }
    }

    /// <summary>
    /// Enumera, via o runtime ILGPU (não a seleção de processo), todos os dispositivos OpenCL visíveis e o
    /// acelerador CPU do ILGPU. Cada candidato carrega identidade completa (plataforma/ids para OpenCL;
    /// NumThreads para CPU), igual à decisão 1 de <c>06-v2-ilgpu-backend.md</c>.
    /// </summary>
    private static IReadOnlyList<IlgpuBackendSelection> EnumerateDeviceCandidates()
    {
        var result = new List<IlgpuBackendSelection>();
        try
        {
            using var context = Context.Create(builder => builder.OpenCL().CPU());
            var devices = context.Devices.ToArray();
            for (var i = 0; i < devices.Length; i++)
            {
                var device = devices[i];
                if (device is CLDevice cl)
                {
                    result.Add(new IlgpuBackendSelection(
                        IlgpuBackendKind.OpenClDevice,
                        i,
                        cl.Name,
                        null,
                        cl.WarpSize,
                        cl.MaxNumThreads,
                        cl.NumMultiprocessors,
                        cl.MaxNumThreadsPerGroup,
                        CpuNumThreads: 0,
                        PlatformName: cl.PlatformName,
                        PlatformId: cl.PlatformId.ToString(),
                        OpenClDeviceId: cl.DeviceId.ToString()));
                }
                else if (device is CPUDevice cpu && device.AcceleratorType == AcceleratorType.CPU)
                {
                    result.Add(new IlgpuBackendSelection(
                        IlgpuBackendKind.IlgpuCpu,
                        i,
                        cpu.Name,
                        null,
                        cpu.WarpSize,
                        cpu.MaxNumThreads,
                        cpu.NumMultiprocessors,
                        cpu.MaxNumThreadsPerGroup,
                        cpu.NumThreads));
                }
            }
        }
        catch (Exception ex) when (ex is not OutOfMemoryException)
        {
            Console.WriteLine($"  falha ao enumerar dispositivos ILGPU: {ex.GetType().Name}: {ex.Message}");
        }

        // CUDA: a seleção só retorna CudaDevice após validar kernels reais (reuso da seleção de 015b).
        try
        {
            var cuda = IlgpuBackendSelector.SelectFor(IlgpuBackendPolicy.CudaPreferred);
            if (cuda.Kind == IlgpuBackendKind.CudaDevice)
            {
                result.Add(cuda);
            }
            else
            {
                Console.WriteLine($"  CUDA indisponível: {cuda.FallbackReason}");
            }
        }
        catch (Exception ex) when (ex is not OutOfMemoryException)
        {
            Console.WriteLine($"  falha ao selecionar CUDA: {ex.GetType().Name}: {ex.Message}");
        }

        return result;
    }

    private static string DescribeIdentity(IlgpuBackendSelection selection) => selection.Kind switch
    {
        IlgpuBackendKind.CudaDevice =>
            $"ILGPU/CUDA :: {selection.DeviceName} [pci={selection.CudaPciId}, driver={CudaDriver.Value}]",
        IlgpuBackendKind.OpenClDevice =>
            $"ILGPU/OpenCL :: {selection.PlatformName} :: {selection.DeviceName} [platformId={selection.PlatformId}, deviceId={selection.OpenClDeviceId}]",
        IlgpuBackendKind.IlgpuCpu => $"ILGPU/CPU :: {selection.DeviceName} [numThreads={selection.CpuNumThreads}]",
        _ => "desconhecido",
    };

    /// <summary>
    /// Mede um tipo em todos os backends candidatos: CPU compilado (sempre), cada dispositivo ILGPU enumerado
    /// (CPU forçado e cada OpenCL). Verifica paridade contra a referência antes de medir; um candidato que falhe
    /// ao inicializar ou cuja paridade divirja é relatado como INDISPONÍVEL com a causa e não é medido.
    /// </summary>
    private static void RunForType<T>(
        string typeLabel,
        IFunctionSet<T> functions,
        IValueSemantics<T> semantics,
        Genome<T> genome,
        T[] inputs,
        IReadOnlyList<IlgpuBackendSelection> candidates,
        Action<T, T> assertEqual)
    {
        var outputCount = genome.Shape.OutputCount;
        var reference = new T[outputCount];
        new ReferenceGenomeEvaluator<T>(functions).Evaluate(genome, inputs, reference);

        // 1) CPU compilado: caminho de produção distinto do ILGPU, sempre disponível.
        var compiled = new T[outputCount];
        var compiledEvaluator = new CompiledGenomeEvaluator<T>(functions, semantics);
        var compiledColdSw = Stopwatch.StartNew();
        compiledEvaluator.Evaluate(genome, inputs, compiled);
        var compiledColdNs = compiledColdSw.Elapsed.TotalMilliseconds * 1_000_000.0;
        for (var i = 0; i < outputCount; i++)
        {
            assertEqual(reference[i], compiled[i]);
        }

        BenchmarkScalar(typeLabel, "CPU compilado", "nenhum", null, "compiled-default", compiledColdNs,
            () => compiledEvaluator.Evaluate(genome, inputs, compiled));

        // 2) Cada dispositivo ILGPU candidato (CPU forçado + cada OpenCL enumerado), independente da seleção
        // de processo.
        foreach (var candidate in candidates)
        {
            var identity = DescribeIdentity(candidate);
            try
            {
                using var evaluator = new IlgpuGenomeEvaluator<T>(functions, null, candidate, null, null, null, null, null, null);
                var actual = new T[outputCount];
                // Primeira execução após a criação do avaliador: inclui compilação de kernel e carga do dispositivo.
                var coldSw = Stopwatch.StartNew();
                evaluator.Evaluate(genome, inputs, actual);
                var coldNs = coldSw.Elapsed.TotalMilliseconds * 1_000_000.0;
                for (var i = 0; i < outputCount; i++)
                {
                    assertEqual(reference[i], actual[i]);
                }

                BenchmarkScalar(typeLabel, identity, evaluator.DeviceName, evaluator.LastFallbackReason, evaluator.FallbackEvaluatorKind, coldNs,
                    () => evaluator.Evaluate(genome, inputs, actual));
            }
            catch (Exception ex) when (ex is not OutOfMemoryException)
            {
                Console.WriteLine($"tipo={typeLabel} backend={identity} status=INDISPONIVEL causa={ex.GetType().Name}: {ex.Message}");
            }
        }
    }

    private static void RunDouble(IReadOnlyList<IlgpuBackendSelection> candidates)
    {
        var functions = StandardDoubleFunctionSet.Instance;
        var add = FindFunctionId(functions, "add");
        var divide = FindFunctionId(functions, "divide");
        var sqrt = FindFunctionId(functions, "sqrt");
        var select = FindFunctionId(functions, "select");
        var greater = FindFunctionId(functions, "greater");
        var constant = FindFunctionId(functions, "const");
        var shape = new GenomeShape(2, 6, 2, 3, 6);
        var genome = Genome<double>.FromGenes(
            shape,
            [divide, constant, sqrt, add, greater, select],
            [
                0, 1, 0,
                0, 0, 0,
                3, 0, 0,
                4, 2, 0,
                0, 1, 0,
                6, 5, 3,
            ],
            [0.0, -4.0, 0.0, 0.0, 0.0, 0.0],
            [7, 5],
            DoubleValueSemantics.Instance,
            functions);
        var inputs = new[] { 7.0, 0.0 };
        RunForType("double", functions, DoubleValueSemantics.Instance, genome, inputs, candidates, AssertDouble);
        RunBatchDouble(functions, genome, inputs, candidates);
    }

    /// <summary>
    /// Cenário batch (EVO-011): avalia um lote de <see cref="BatchSize"/> genomas pela API
    /// <see cref="IBatchFitnessEvaluator{T}.EvaluateBatch"/> com cada backend, após comparar cada resultado à
    /// referência. Mede a primeira chamada do lote (fria) e a média aquecida por lote e por genoma.
    /// </summary>
    private static void RunBatchDouble(
        IFunctionSet<double> functions,
        Genome<double> genome,
        double[] inputs,
        IReadOnlyList<IlgpuBackendSelection> candidates)
    {
        var genomes = Enumerable.Repeat(genome, BatchSize).ToArray();
        var fitness = new BatchGenomeFitness(inputs, 3.0);
        var reference = new FitnessResult[BatchSize];
        var referenceEvaluator = new ReferenceGenomeEvaluator<double>(functions);
        for (var i = 0; i < BatchSize; i++)
        {
            reference[i] = fitness.Evaluate(genome, referenceEvaluator, CancellationToken.None);
        }

        var compiledEvaluator = new CompiledGenomeEvaluator<double>(functions, DoubleValueSemantics.Instance);
        MeasureBatch("CPU compilado", "nenhum", null, "compiled-default", compiledEvaluator, fitness, genomes, reference);

        foreach (var candidate in candidates)
        {
            var identity = DescribeIdentity(candidate);
            try
            {
                using var evaluator = new IlgpuGenomeEvaluator<double>(functions, null, candidate, null, null, null, null, null, null);
                MeasureBatch(identity, evaluator.DeviceName, evaluator.LastFallbackReason, evaluator.FallbackEvaluatorKind, evaluator, fitness, genomes, reference);
            }
            catch (Exception ex) when (ex is not OutOfMemoryException)
            {
                Console.WriteLine($"tipo=double-batch backend={identity} status=INDISPONIVEL causa={ex.GetType().Name}: {ex.Message}");
            }
        }
    }

    private static void MeasureBatch(
        string backend,
        string device,
        string? fallbackReason,
        string fallbackKind,
        IGenomeEvaluator<double> evaluator,
        IBatchFitnessEvaluator<double> fitness,
        Genome<double>[] genomes,
        FitnessResult[] reference)
    {
        var results = new FitnessResult[genomes.Length];
        var coldSw = Stopwatch.StartNew();
        fitness.EvaluateBatch(genomes, evaluator, results, CancellationToken.None);
        var coldBatchMs = coldSw.Elapsed.TotalMilliseconds;
        for (var i = 0; i < genomes.Length; i++)
        {
            if (results[i].IsValid != reference[i].IsValid)
            {
                throw new InvalidOperationException($"paridade batch falhou no genoma {i}");
            }

            AssertDouble(reference[i].Value, results[i].Value);
        }

        for (var i = 0; i < BatchWarmupRepetitions; i++)
        {
            fitness.EvaluateBatch(genomes, evaluator, results, CancellationToken.None);
        }

        var sw = Stopwatch.StartNew();
        for (var i = 0; i < BatchMeasureRepetitions; i++)
        {
            fitness.EvaluateBatch(genomes, evaluator, results, CancellationToken.None);
        }

        sw.Stop();
        var warmBatchMs = sw.Elapsed.TotalMilliseconds / BatchMeasureRepetitions;
        Console.WriteLine(
            $"tipo=double-batch(n={genomes.Length}) parity=PASS backend={backend} device={device} fallbackKind={fallbackKind} fallback={(fallbackReason ?? "(nenhum)")} " +
            $"frio_ms/lote={coldBatchMs:F3} aquecido_ms/lote={warmBatchMs:F4} aquecido_ns/genoma={warmBatchMs * 1_000_000.0 / genomes.Length:F1} repeticoesLote={BatchMeasureRepetitions}");
    }

    /// <summary>
    /// Mede H2D (<c>CopyFromCPU</c>), D2H (<c>CopyToCPU</c>) e alocação de buffer (<c>Allocate1D</c> + descarte) em
    /// CUDA e OpenCL, separados do tempo total de avaliação. O avaliador de produção aloca e descarta buffers por
    /// chamada de <c>Evaluate</c> (ver <c>IlgpuGenomeEvaluator.TryEvaluateOnDevice</c>), então o custo de alocação
    /// medido aqui corresponde ao escopo por chamada do caminho de produção.
    /// </summary>
    private static void RunTransferProbe(IReadOnlyList<IlgpuBackendSelection> candidates)
    {
        foreach (var candidate in candidates)
        {
            if (candidate.Kind is not (IlgpuBackendKind.CudaDevice or IlgpuBackendKind.OpenClDevice))
            {
                continue;
            }

            var identity = DescribeIdentity(candidate);
            try
            {
                using var context = candidate.Kind == IlgpuBackendKind.CudaDevice
                    ? Context.Create(builder => builder.Cuda())
                    : Context.Create(builder => builder.OpenCL().CPU());
                using var accelerator = context.Devices[candidate.DeviceIndex].CreateAccelerator(context);
                var host = new int[TransferLength];
                for (var i = 0; i < host.Length; i++)
                {
                    host[i] = i;
                }

                var hostOut = new int[TransferLength];
                using var buffer = accelerator.Allocate1D<int>(TransferLength);
                buffer.CopyFromCPU(host);
                buffer.CopyToCPU(hostOut);

                var h2dNs = MeasureNsPerCall(() => buffer.CopyFromCPU(host));
                var d2hNs = MeasureNsPerCall(() => buffer.CopyToCPU(hostOut));
                var allocNs = MeasureNsPerCall(() =>
                {
                    using var temp = accelerator.Allocate1D<int>(TransferLength);
                });
                Console.WriteLine(
                    $"transferencia backend={identity} elementos={TransferLength} repeticoes={TransferRepetitions} " +
                    $"h2d_ns/chamada={h2dNs:F1} d2h_ns/chamada={d2hNs:F1} alocacao_alloc+free_ns/chamada(escopo_por_chamada)={allocNs:F1}");
            }
            catch (Exception ex) when (ex is not OutOfMemoryException)
            {
                Console.WriteLine($"transferencia backend={identity} status=INDISPONIVEL causa={ex.GetType().Name}: {ex.Message}");
            }
        }
    }

    private static double MeasureNsPerCall(Action action)
    {
        for (var i = 0; i < WarmupRepetitions; i++)
        {
            action();
        }

        var sw = Stopwatch.StartNew();
        for (var i = 0; i < TransferRepetitions; i++)
        {
            action();
        }

        sw.Stop();
        return sw.Elapsed.TotalMilliseconds * 1_000_000.0 / TransferRepetitions;
    }

    /// <summary>
    /// Fitness de lote de teste: avalia o genoma no avaliador recebido e pontua pela distância ao alvo.
    /// </summary>
    private sealed class BatchGenomeFitness : IBatchFitnessEvaluator<double>
    {
        private readonly double[] _inputs;
        private readonly double _target;

        public BatchGenomeFitness(double[] inputs, double target)
        {
            _inputs = inputs;
            _target = target;
        }

        public int PreferredBatchSize => BatchSize;

        public FitnessResult Evaluate(Genome<double> genome, IGenomeEvaluator<double> evaluator, CancellationToken cancellationToken)
        {
            var output = new double[genome.Shape.OutputCount];
            evaluator.Evaluate(genome, _inputs, output);
            var distance = output[0] - _target;
            return FitnessResult.Valid(-distance * distance);
        }

        public void EvaluateBatch(
            ReadOnlySpan<Genome<double>> genomes,
            IGenomeEvaluator<double> evaluator,
            Span<FitnessResult> results,
            CancellationToken cancellationToken)
        {
            for (var i = 0; i < genomes.Length; i++)
            {
                results[i] = Evaluate(genomes[i], evaluator, cancellationToken);
            }
        }
    }

    private static void RunInt32(IReadOnlyList<IlgpuBackendSelection> candidates)
    {
        var functions = StandardInt32FunctionSet.Instance;
        var add = FindFunctionId(functions, "add");
        var divide = FindFunctionId(functions, "divide");
        var abs = FindFunctionId(functions, "abs");
        var select = FindFunctionId(functions, "select");
        var greater = FindFunctionId(functions, "greater");
        var constant = FindFunctionId(functions, "const");
        var shape = new GenomeShape(2, 6, 2, 3, 6);
        var genome = Genome<int>.FromGenes(
            shape,
            [constant, abs, constant, divide, greater, select],
            [
                0, 0, 0,
                2, 0, 0,
                0, 0, 0,
                2, 4, 0,
                0, 1, 0,
                6, 3, 5,
            ],
            [int.MinValue, 0, -1, 0, 0, 0],
            [7, 5],
            Int32ValueSemantics.Instance,
            functions);
        var inputs = new[] { 2, 1 };
        RunForType("int", functions, Int32ValueSemantics.Instance, genome, inputs, candidates,
            (expected, actual) =>
            {
                if (expected != actual)
                {
                    throw new InvalidOperationException($"paridade int falhou: esperado {expected}, atual {actual}");
                }
            });
    }

    private static void RunDoubleArray(IReadOnlyList<IlgpuBackendSelection> candidates)
    {
        var functions = StandardDoubleArrayFunctionSet.Instance;
        var add = FindFunctionId(functions, "add");
        var divide = FindFunctionId(functions, "divide");
        var sqrt = FindFunctionId(functions, "sqrt");
        var select = FindFunctionId(functions, "select");
        var greater = FindFunctionId(functions, "greater");
        var constant = FindFunctionId(functions, "const");
        var shape = new GenomeShape(2, 6, 2, 3, 6);
        var genome = Genome<double[]>.FromGenes(
            shape,
            [divide, constant, sqrt, add, greater, select],
            [
                0, 1, 0,
                0, 0, 0,
                3, 0, 0,
                4, 2, 0,
                0, 1, 0,
                6, 5, 3,
            ],
            [
                Array.Empty<double>(),
                [-4.0, -4.0, -4.0, -4.0],
                Array.Empty<double>(),
                Array.Empty<double>(),
                Array.Empty<double>(),
                Array.Empty<double>(),
            ],
            [7, 5],
            DoubleArrayValueSemantics.Instance,
            functions);
        var inputs = new[]
        {
            new[] { 7.0, -4.0, 8.0, double.NaN },
            new[] { 0.0, 2.0, 4.0, 1.0 },
        };
        RunForType("double[]", functions, DoubleArrayValueSemantics.Instance, genome, inputs, candidates, AssertDoubleArray);
    }

    private static void RunInt32Array(IReadOnlyList<IlgpuBackendSelection> candidates)
    {
        var functions = StandardInt32ArrayFunctionSet.Instance;
        var add = FindFunctionId(functions, "add");
        var divide = FindFunctionId(functions, "divide");
        var abs = FindFunctionId(functions, "abs");
        var select = FindFunctionId(functions, "select");
        var greater = FindFunctionId(functions, "greater");
        var constant = FindFunctionId(functions, "const");
        var shape = new GenomeShape(2, 6, 2, 3, 6);
        var genome = Genome<int[]>.FromGenes(
            shape,
            [constant, abs, constant, divide, greater, select],
            [
                0, 0, 0,
                2, 0, 0,
                0, 0, 0,
                2, 4, 0,
                0, 1, 0,
                6, 3, 5,
            ],
            [
                [int.MinValue, int.MinValue, int.MinValue, int.MinValue],
                Array.Empty<int>(),
                [-1, -1, -1, -1],
                Array.Empty<int>(),
                Array.Empty<int>(),
                Array.Empty<int>(),
            ],
            [7, 5],
            Int32ArrayValueSemantics.Instance,
            functions);
        var inputs = new[]
        {
            new[] { 2, 0, 5, 1 },
            new[] { 1, 1, 3, 2 },
        };
        RunForType("int[]", functions, Int32ArrayValueSemantics.Instance, genome, inputs, candidates,
            (expected, actual) =>
            {
                if (!expected.SequenceEqual(actual))
                {
                    throw new InvalidOperationException("paridade int[] falhou");
                }
            });
    }

    /// <summary>
    /// Mede tempo e bytes alocados por avaliação. A medição de alocação (<see cref="GC.GetAllocatedBytesForCurrentThread"/>)
    /// cobre apenas o thread chamador; kernels ILGPU executam em threads internos (worker pool ILGPU CPU ou o
    /// driver OpenCL) cujas alocações não são contabilizadas aqui. Isso não é um total global de alocação.
    /// </summary>
    private static void BenchmarkScalar(string type, string backend, string device, string? fallbackReason, string fallbackKind, double coldNs, Action action)
    {
        for (var i = 0; i < WarmupRepetitions; i++)
        {
            action();
        }

        var allocBefore = GC.GetAllocatedBytesForCurrentThread();
        var sw = Stopwatch.StartNew();
        for (var i = 0; i < MeasureRepetitions; i++)
        {
            action();
        }

        sw.Stop();
        var allocTotal = GC.GetAllocatedBytesForCurrentThread() - allocBefore;
        var nsPerEval = sw.Elapsed.TotalMilliseconds * 1_000_000.0 / MeasureRepetitions;
        var bytesPerEval = (double)allocTotal / MeasureRepetitions;
        Console.WriteLine(
            $"tipo={type} parity=PASS backend={backend} device={device} fallbackKind={fallbackKind} fallback={(fallbackReason ?? "(nenhum)")} " +
            $"frio_ns/avaliacao(primeira_execucao)={coldNs:F1} aquecido_ns/avaliacao={nsPerEval:F1} total_ms={sw.Elapsed.TotalMilliseconds:F2} bytes_alocados/avaliacao(thread_chamador)={bytesPerEval:F1} bytes_alocados_total(thread_chamador)={allocTotal}");
    }

    private static int FindFunctionId<T>(IFunctionSet<T> functions, string part)
    {
        for (var i = 0; i < functions.Count; i++)
        {
            if (functions.Get(i).Id.Contains(part, StringComparison.OrdinalIgnoreCase))
            {
                return i;
            }
        }

        throw new InvalidOperationException($"função '{part}' não encontrada");
    }

    internal static void AssertDouble(double expected, double actual)
    {
        if (double.IsNaN(expected) || double.IsNaN(actual))
        {
            if (!(double.IsNaN(expected) && double.IsNaN(actual)))
            {
                throw new InvalidOperationException($"paridade double falhou: esperado {expected}, atual {actual}");
            }

            return;
        }

        if (double.IsInfinity(expected) || double.IsInfinity(actual))
        {
            if (expected != actual)
            {
                throw new InvalidOperationException($"paridade double falhou: esperado {expected}, atual {actual}");
            }

            return;
        }

        var diff = Math.Abs(expected - actual);
        if (!(diff <= 1e-12 || diff <= 1e-12 * Math.Abs(expected)))
        {
            throw new InvalidOperationException($"paridade double falhou: esperado {expected}, atual {actual}");
        }
    }

    internal static void AssertDoubleArray(double[] expected, double[] actual)
    {
        if (expected.Length != actual.Length)
        {
            throw new InvalidOperationException("paridade double[] falhou: comprimento divergente");
        }

        for (var i = 0; i < expected.Length; i++)
        {
            AssertDouble(expected[i], actual[i]);
        }
    }
}
