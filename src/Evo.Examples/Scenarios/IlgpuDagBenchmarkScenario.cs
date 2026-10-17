using System.Globalization;
using Evo.Core.Functions;

namespace Evo.Examples.Scenarios;

/// <summary>
/// Validadores de paridade double/double[] migrados de `Evo.Benchmarks.IlgpuBenchmarkProgram` (rodada de reparo
/// R1-2): NaN só é equivalente a NaN; infinitos de sinais opostos nunca são equivalentes.
/// </summary>
public static class IlgpuBenchmarkValidators
{
    public static void AssertDouble(double expected, double actual)
    {
        var ok = (double.IsNaN(expected) && double.IsNaN(actual))
            || (double.IsPositiveInfinity(expected) && double.IsPositiveInfinity(actual))
            || (double.IsNegativeInfinity(expected) && double.IsNegativeInfinity(actual))
            || (!double.IsNaN(expected) && !double.IsInfinity(expected) && !double.IsNaN(actual) && !double.IsInfinity(actual)
                && Math.Abs(expected - actual) <= 1e-9 * Math.Max(1.0, Math.Max(Math.Abs(expected), Math.Abs(actual))));
        if (!ok)
        {
            throw new InvalidOperationException($"paridade double falhou: esperado={expected} obtido={actual}");
        }
    }

    public static void AssertDoubleArray(double[] expected, double[] actual)
    {
        if (expected.Length != actual.Length)
        {
            throw new InvalidOperationException($"paridade double[] falhou: tamanhos diferentes esperado={expected.Length} obtido={actual.Length}");
        }

        for (var i = 0; i < expected.Length; i++)
        {
            AssertDouble(expected[i], actual[i]);
        }
    }
}

/// <summary>
/// Cenário migrado de `Evo.Benchmarks.IlgpuBenchmarkProgram` (rodada de reparo R1-2): gate de paridade
/// reference-vs-ILGPU (quando disponível) sobre uma cadeia de profundidade 6, com medição de throughput
/// aquecido. Indisponibilidade de acelerador é relatada em `ilgpu_fallback`, nunca como falha de gate
/// (consistente com "available ILGPU" da tabela normativa).
/// </summary>
public sealed class IlgpuDagBenchmarkScenario : IBenchmarkScenario
{
    public string Id => "ilgpu-benchmark";

    public string Description => "Paridade reference-vs-ILGPU (disponível) sobre cadeia de profundidade 6 e throughput aquecido; migrado de Evo.Benchmarks.";

    public BenchmarkScenarioResult Run(CancellationToken cancellationToken)
    {
        const int Rows = 64;
        const int Cols = 8;
        const int Depth = 6;
        const int Repetitions = 20;
        var rnd = new Random(42);
        var inputs = new double[Rows * Cols];
        for (var i = 0; i < inputs.Length; i++)
        {
            inputs[i] = rnd.NextDouble() * 4.0 - 2.0;
        }

        var functions = StandardDoubleFunctionSet.Instance;
        var addId = EvaluatorScenarioSupport.FindFunctionId(functions, 2, "add");
        var genome = EvaluatorScenarioSupport.BuildChain(Cols, Depth, addId);
        var (parityOk, gates, ilgpuFallback) = EvaluatorScenarioSupport.CheckParity(genome, functions, inputs, Rows);

        var measurements = new Dictionary<string, string>(StringComparer.Ordinal);
        if (ilgpuFallback is not null)
        {
            measurements["ilgpu_fallback"] = ilgpuFallback;
        }
        else
        {
            var outputs = new double[Rows];
            using var ilgpu = new Evo.Ilgpu.IlgpuGenomeEvaluator<double>(functions);
            for (var r = 0; r < 2; r++)
            {
                for (var c = 0; c < Rows; c++)
                {
                    ilgpu.Evaluate(genome, inputs.AsSpan(c * Cols, Cols), outputs.AsSpan(c, 1));
                }
            }

            var sw = System.Diagnostics.Stopwatch.StartNew();
            for (var r = 0; r < Repetitions; r++)
            {
                for (var c = 0; c < Rows; c++)
                {
                    ilgpu.Evaluate(genome, inputs.AsSpan(c * Cols, Cols), outputs.AsSpan(c, 1));
                }
            }

            sw.Stop();
            measurements["warm_ilgpu_ns_per_eval"] = (sw.Elapsed.TotalMilliseconds * 1e6 / (Rows * (double)Repetitions)).ToString("F1", CultureInfo.InvariantCulture);
        }

        var raw = new Dictionary<string, string>(StringComparer.Ordinal)
        {
            ["seed"] = "42",
            ["rows"] = Rows.ToString(CultureInfo.InvariantCulture),
            ["cols"] = Cols.ToString(CultureInfo.InvariantCulture),
        };

        return new BenchmarkScenarioResult(
            Id,
            parityOk,
            gates,
            measurements,
            raw,
            EvaluatorScenarioSupport.Params(("rows", Rows), ("cols", Cols), ("depth", Depth), ("repetitions", Repetitions)),
            parityOk ? "paridade reference-vs-ILGPU confirmada (ou indisponível, relatado em ilgpu_fallback)" : "divergência de paridade reference-vs-ILGPU");
    }
}
