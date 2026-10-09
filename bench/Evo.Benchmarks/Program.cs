using System.Diagnostics;
using Evo.Core;
using Evo.Core.Compiled;
using Evo.Core.Functions;
using Evo.Core.Values;

namespace Evo.Benchmarks;

/// <summary>
/// Benchmark host. The measured baseline (individuals/s, active nodes/s, time per
/// generation, allocations and effective backend) is implemented by EVO-010.
/// This EVO-005 diagnostic reports the effective backend of compiled/SIMD evaluation.
/// </summary>
public static class Program
{
    private const int CaseCount = 4096;
    private const int Repetitions = 200;

    /// <summary>Runs the EVO-005 SIMD diagnostic: timing without counters.</summary>
    public static int Main()
    {
        var functions = StandardDoubleFunctionSet.Instance;
        var shape = new GenomeShape(InputCount: 2, NodeCount: 4, OutputCount: 1, MaxArity: 2, LevelsBack: 4);
        var genome = Genome<double>.FromGenes(
            shape,
            new[] { 0, 1, 2, 0 },
            new[] { 0, 1, 2, 2, 3, 0, 4, 0 },
            new[] { 0.0, 0.0, 0.0, 0.0 },
            new[] { 5 },
            DoubleValueSemantics.Instance,
            functions);

        var inputs = new double[CaseCount * shape.InputCount];
        for (var i = 0; i < inputs.Length; i++)
        {
            inputs[i] = (i % 97) * 0.03 - 1.5;
        }

        var outputs = new double[CaseCount * shape.OutputCount];
        var simd = new SimdGenomeEvaluator<double>(functions, DoubleValueSemantics.Instance);
        simd.Evaluate(genome, inputs, CaseCount, outputs);

        var watch = Stopwatch.StartNew();
        EvaluationBackendReport report = default!;
        for (var r = 0; r < Repetitions; r++)
        {
            report = simd.Evaluate(genome, inputs, CaseCount, outputs);
        }

        watch.Stop();
        var casesPerSecond = (double)CaseCount * Repetitions / watch.Elapsed.TotalSeconds;
        Console.WriteLine("Evo.Benchmarks — diagnóstico SIMD EVO-005 (double, indivíduo único, múltiplos casos)");
        Console.WriteLine($"backend={report.Backend} vetorWidth={report.VectorWidth} blocos={report.VectorBlocks} cauda={report.TailCases} casos={report.CaseCount}");
        Console.WriteLine($"fallbackReason={report.FallbackReason ?? "(nenhum)"}");
        Console.WriteLine($"casos/s={casesPerSecond:F0} (sem contadores; {Repetitions} repetições de {CaseCount} casos)");
        return 0;
    }
}
