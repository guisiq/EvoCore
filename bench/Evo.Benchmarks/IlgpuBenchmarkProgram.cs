using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Runtime.CompilerServices;
using Evo.Core;
using Evo.Core.Functions;
using Evo.Core.Values;
using Evo.Ilgpu;

[assembly: InternalsVisibleTo("Evo.Ilgpu.Tests")]

namespace Evo.Benchmarks;

internal static class IlgpuBenchmarkProgram
{
    private const int WarmupRepetitions = 100;
    private const int MeasureRepetitions = 1000;

    public static int Run()
    {
        Console.WriteLine("Evo.Benchmarks — benchmark limitado do backend ILGPU/OpenCL (EVO-101)");
        Console.WriteLine($"runtime={RuntimeInformation.FrameworkDescription} arquitetura={RuntimeInformation.ProcessArchitecture}");
        Console.WriteLine($"warmup={WarmupRepetitions} repeticoes={MeasureRepetitions}");

        RunDouble();
        RunInt32();
        RunDoubleArray();
        RunInt32Array();
        return 0;
    }

    private static void RunDouble()
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
        var reference = new double[2];
        var actual = new double[2];
        new ReferenceGenomeEvaluator<double>(functions).Evaluate(genome, inputs, reference);
        using var evaluator = new IlgpuGenomeEvaluator<double>(functions);
        evaluator.Evaluate(genome, inputs, actual);
        for (var i = 0; i < reference.Length; i++)
        {
            AssertDouble(reference[i], actual[i]);
        }

        BenchmarkScalar(
            "double",
            evaluator.Backend,
            evaluator.DeviceName,
            evaluator.LastFallbackReason,
            evaluator.FallbackEvaluatorKind,
            () => evaluator.Evaluate(genome, inputs, actual));
    }

    private static void RunInt32()
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
        var reference = new int[2];
        var actual = new int[2];
        new ReferenceGenomeEvaluator<int>(functions).Evaluate(genome, inputs, reference);
        using var evaluator = new IlgpuGenomeEvaluator<int>(functions);
        evaluator.Evaluate(genome, inputs, actual);
        if (!reference.SequenceEqual(actual))
        {
            throw new InvalidOperationException("paridade int falhou antes da medição");
        }

        BenchmarkScalar(
            "int",
            evaluator.Backend,
            evaluator.DeviceName,
            evaluator.LastFallbackReason,
            evaluator.FallbackEvaluatorKind,
            () => evaluator.Evaluate(genome, inputs, actual));
    }

    private static void RunDoubleArray()
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
        var reference = new double[2][];
        var actual = new double[2][];
        new ReferenceGenomeEvaluator<double[]>(functions).Evaluate(genome, inputs, reference);
        using var evaluator = new IlgpuGenomeEvaluator<double[]>(functions);
        evaluator.Evaluate(genome, inputs, actual);
        for (var i = 0; i < reference.Length; i++)
        {
            AssertDoubleArray(reference[i], actual[i]);
        }

        BenchmarkScalar(
            "double[]",
            evaluator.Backend,
            evaluator.DeviceName,
            evaluator.LastFallbackReason,
            evaluator.FallbackEvaluatorKind,
            () => evaluator.Evaluate(genome, inputs, actual));
    }

    private static void RunInt32Array()
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
        var reference = new int[2][];
        var actual = new int[2][];
        new ReferenceGenomeEvaluator<int[]>(functions).Evaluate(genome, inputs, reference);
        using var evaluator = new IlgpuGenomeEvaluator<int[]>(functions);
        evaluator.Evaluate(genome, inputs, actual);
        for (var i = 0; i < reference.Length; i++)
        {
            if (!reference[i].SequenceEqual(actual[i]))
            {
                throw new InvalidOperationException("paridade int[] falhou antes da medição");
            }
        }

        BenchmarkScalar(
            "int[]",
            evaluator.Backend,
            evaluator.DeviceName,
            evaluator.LastFallbackReason,
            evaluator.FallbackEvaluatorKind,
            () => evaluator.Evaluate(genome, inputs, actual));
    }

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
        Console.WriteLine($"tipo={type} parity=PASS backend={backend} device={device} fallbackKind={fallbackKind} fallback={(fallbackReason ?? "(nenhum)")} ns/avaliacao={nsPerEval:F1} total_ms={sw.Elapsed.TotalMilliseconds:F2} bytes_alocados/avaliacao={bytesPerEval:F1} bytes_alocados_total={allocTotal}");
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
