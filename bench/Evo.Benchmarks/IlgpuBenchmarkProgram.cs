using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Runtime.CompilerServices;
using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.CPU;
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

    public static int Run()
    {
        Console.WriteLine("Evo.Benchmarks — benchmark do backend ILGPU/OpenCL (EVO-101/EVO-102)");
        Console.WriteLine($"runtime={RuntimeInformation.FrameworkDescription} arquitetura={RuntimeInformation.ProcessArchitecture}");
        Console.WriteLine($"warmup={WarmupRepetitions} repeticoes={MeasureRepetitions}");

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
        return 0;
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

        return result;
    }

    private static string DescribeIdentity(IlgpuBackendSelection selection) => selection.Kind switch
    {
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
        compiledEvaluator.Evaluate(genome, inputs, compiled);
        for (var i = 0; i < outputCount; i++)
        {
            assertEqual(reference[i], compiled[i]);
        }

        BenchmarkScalar(typeLabel, "CPU compilado", "nenhum", null, "compiled-default",
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
                evaluator.Evaluate(genome, inputs, actual);
                for (var i = 0; i < outputCount; i++)
                {
                    assertEqual(reference[i], actual[i]);
                }

                BenchmarkScalar(typeLabel, identity, evaluator.DeviceName, evaluator.LastFallbackReason, evaluator.FallbackEvaluatorKind,
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
    private static void BenchmarkScalar(string type, string backend, string device, string? fallbackReason, string fallbackKind, Action action)
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
            $"ns/avaliacao={nsPerEval:F1} total_ms={sw.Elapsed.TotalMilliseconds:F2} bytes_alocados/avaliacao(thread_chamador)={bytesPerEval:F1} bytes_alocados_total(thread_chamador)={allocTotal}");
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
