using System.Globalization;
using Evo.Core;
using Evo.Core.Compiled;
using Evo.Core.Functions;
using Evo.Core.Values;

namespace Evo.Examples.Scenarios;

/// <summary>
/// Suporte compartilhado pelos cenários migrados de `Evo.Benchmarks` (rodada de reparo R1-2): cadeia, fan-out,
/// diamante e diamantes em níveis. Gate = contagens por endereço (escalar e SIMD/batch) e, quando aplicável,
/// comparação com uma árvore ingênua de expansão limitada; tempo/alocação ficam em <c>Measurements</c>, nunca em
/// <c>Gates</c>.
/// </summary>
internal static class DagWorkloadSupport
{
    private const int CaseCount = 256;
    private const int WarmRepetitions = 20;
    private const int ColdSamples = 3;
    private const long TreeVisitCap = 1_000_000;
    private const int InputCount = 2;

    internal sealed record Workload(string Name, Genome<double> Genome, bool TreeCompare);

    public static BenchmarkScenarioResult Run(string scenarioId, Workload w)
    {
        var functions = StandardDoubleFunctionSet.Instance;
        var g = w.Genome;
        var shape = g.Shape;
        var inputCount = shape.InputCount;
        var nodeCount = shape.NodeCount;
        var outputCount = shape.OutputCount;
        var values = DoubleValueSemantics.Instance;
        var inputs = MakeInputs(inputCount, CaseCount);
        var outputs = new double[CaseCount * outputCount];
        var failures = 0;
        var gates = new Dictionary<string, string>(StringComparer.Ordinal);
        var measurements = new Dictionary<string, string>(StringComparer.Ordinal);

        var active = ReachableNodes(g);
        var activeCount = active.Count(a => a);

        var counters = new CompiledExecutionCounters();
        var countEval = new CompiledGenomeEvaluator<double>(functions, values, 256, counters);
        for (var c = 0; c < CaseCount; c++)
        {
            countEval.Evaluate(g, inputs.AsSpan(c * inputCount, inputCount), outputs.AsSpan(c * outputCount, outputCount));
        }

        var scalarMismatch = 0;
        for (var n = 0; n < nodeCount; n++)
        {
            var expected = active[n] ? CaseCount : 0;
            if (counters.ScalarExecutions[n] != expected)
            {
                scalarMismatch++;
            }
        }

        failures += scalarMismatch;
        gates["scalar_address_counts_ok"] = (scalarMismatch == 0).ToString();

        var simdCounters = new CompiledExecutionCounters();
        var simdCount = new SimdGenomeEvaluator<double>(functions, values, 256, simdCounters);
        var countReport = simdCount.Evaluate(g, inputs, CaseCount, outputs);
        var simdMismatch = 0;
        for (var n = 0; n < nodeCount; n++)
        {
            if (countReport.Backend == EvaluationBackend.Simd)
            {
                if (simdCounters.KernelExecutions[n] != (active[n] ? countReport.VectorBlocks : 0)
                    || simdCounters.TailExecutions[n] != (active[n] ? countReport.TailCases : 0))
                {
                    simdMismatch++;
                }
            }
            else if (simdCounters.ScalarExecutions[n] != (active[n] ? CaseCount : 0))
            {
                simdMismatch++;
            }
        }

        failures += simdMismatch;
        gates["batch_address_counts_ok"] = (simdMismatch == 0).ToString();
        measurements["batch_backend"] = countReport.Backend.ToString();
        measurements["batch_fallback"] = countReport.FallbackReason ?? "(nenhum)";

        var treeOutputs = new double[CaseCount * outputCount];
        var treeOk = false;
        if (w.TreeCompare)
        {
            try
            {
                for (var c = 0; c < CaseCount; c++)
                {
                    long caseBudget = TreeVisitCap;
                    var scratch = new int[nodeCount];
                    for (var o = 0; o < outputCount; o++)
                    {
                        treeOutputs[c * outputCount + o] = TreeEval(g, functions, inputs.AsSpan(c * inputCount, inputCount), OutputAddress(g, o), scratch, ref caseBudget);
                    }
                }

                treeOk = true;
            }
            catch (InvalidOperationException ex)
            {
                measurements["tree_not_expanded"] = ex.Message;
            }
        }

        if (treeOk)
        {
            var treeMismatch = 0;
            for (var i = 0; i < outputs.Length; i++)
            {
                if (BitConverter.DoubleToInt64Bits(treeOutputs[i]) != BitConverter.DoubleToInt64Bits(outputs[i]))
                {
                    treeMismatch++;
                }
            }

            failures += treeMismatch;
            gates["naive_tree_matches_dag_ok"] = (treeMismatch == 0).ToString();
        }

        var coldUs = new List<double>();
        for (var s = 0; s < ColdSamples; s++)
        {
            var coldEval = new CompiledGenomeEvaluator<double>(functions, values);
            var sw = System.Diagnostics.Stopwatch.StartNew();
            coldEval.Evaluate(g, inputs.AsSpan(0, inputCount), outputs.AsSpan(0, outputCount));
            sw.Stop();
            coldUs.Add(sw.Elapsed.TotalMilliseconds * 1000.0);
        }

        coldUs.Sort();
        measurements["cold_compile_us_median"] = coldUs[ColdSamples / 2].ToString("F1", CultureInfo.InvariantCulture);

        var warmEval = new CompiledGenomeEvaluator<double>(functions, values);
        ScalarPass(warmEval, g, inputs, outputs, inputCount, outputCount);
        var sw1 = System.Diagnostics.Stopwatch.StartNew();
        for (var r = 0; r < WarmRepetitions; r++)
        {
            ScalarPass(warmEval, g, inputs, outputs, inputCount, outputCount);
        }

        sw1.Stop();
        var evals = (double)CaseCount * WarmRepetitions;
        measurements["warm_scalar_ns_per_eval"] = (sw1.Elapsed.TotalMilliseconds * 1e6 / evals).ToString("F1", CultureInfo.InvariantCulture);

        var simdTime = new SimdGenomeEvaluator<double>(functions, values);
        var sw2 = System.Diagnostics.Stopwatch.StartNew();
        for (var r = 0; r < WarmRepetitions; r++)
        {
            simdTime.Evaluate(g, inputs, CaseCount, outputs);
        }

        sw2.Stop();
        measurements["warm_batch_ns_per_eval"] = (sw2.Elapsed.TotalMilliseconds * 1e6 / evals).ToString("F1", CultureInfo.InvariantCulture);

        var rawData = new Dictionary<string, string>(StringComparer.Ordinal)
        {
            ["nodeCount"] = nodeCount.ToString(CultureInfo.InvariantCulture),
            ["activeNodeCount"] = activeCount.ToString(CultureInfo.InvariantCulture),
            ["outputCount"] = outputCount.ToString(CultureInfo.InvariantCulture),
            ["caseCount"] = CaseCount.ToString(CultureInfo.InvariantCulture),
        };

        var parameters = EvaluatorScenarioSupport.Params(("workload", w.Name), ("caseCount", CaseCount), ("warmRepetitions", WarmRepetitions));
        var ok = failures == 0;
        return new BenchmarkScenarioResult(
            scenarioId,
            ok,
            gates,
            measurements,
            rawData,
            parameters,
            ok ? $"gate de contagens por endereco ({w.Name}): PASS" : $"gate de contagens por endereco ({w.Name}): FAIL (divergencias={failures})");
    }

    private static void ScalarPass(CompiledGenomeEvaluator<double> evaluator, Genome<double> g, double[] inputs, double[] outputs, int inputCount, int outputCount)
    {
        for (var c = 0; c < CaseCount; c++)
        {
            evaluator.Evaluate(g, inputs.AsSpan(c * inputCount, inputCount), outputs.AsSpan(c * outputCount, outputCount));
        }
    }

    private static int OutputAddress(Genome<double> g, int output) => g.OutputGenes[output];

    private static double TreeEval(Genome<double> g, StandardDoubleFunctionSet functions, ReadOnlySpan<double> inputs, int address, int[] visits, ref long budget)
    {
        var inputCount = g.Shape.InputCount;
        if (address < inputCount)
        {
            return inputs[address];
        }

        if (--budget < 0)
        {
            throw new InvalidOperationException("limite de expansao da arvore atingido");
        }

        var node = address - inputCount;
        visits[node]++;
        Span<double> args = stackalloc double[g.Shape.MaxArity];
        for (var slot = 0; slot < g.Shape.MaxArity; slot++)
        {
            args[slot] = TreeEval(g, functions, inputs, g.GetConnection(node, slot), visits, ref budget);
        }

        return functions.Get(g.FunctionGenes[node]).Invoke(args, g.ParameterGenes[node]);
    }

    private static bool[] ReachableNodes(Genome<double> g)
    {
        var inputCount = g.Shape.InputCount;
        var active = new bool[g.Shape.NodeCount];
        var stack = new Stack<int>();
        foreach (var o in g.OutputGenes)
        {
            if (o >= inputCount)
            {
                stack.Push(o - inputCount);
            }
        }

        while (stack.Count > 0)
        {
            var n = stack.Pop();
            if (active[n])
            {
                continue;
            }

            active[n] = true;
            for (var slot = 0; slot < g.Shape.MaxArity; slot++)
            {
                var a = g.GetConnection(n, slot);
                if (a >= inputCount && !active[a - inputCount])
                {
                    stack.Push(a - inputCount);
                }
            }
        }

        return active;
    }

    private static double[] MakeInputs(int inputCount, int cases)
    {
        var inputs = new double[cases * inputCount];
        for (var i = 0; i < inputs.Length; i++)
        {
            inputs[i] = (i % 97) * 0.03 - 1.5;
        }

        return inputs;
    }

    private static int FindAddFunctionId() => EvaluatorScenarioSupport.FindFunctionId(StandardDoubleFunctionSet.Instance, 2, "add");

    private static Workload Make(string name, int nodeCount, int outputCount, int[] connections, int[] outputs, bool treeCompare)
    {
        var addId = FindAddFunctionId();
        var shape = new GenomeShape(InputCount: InputCount, NodeCount: nodeCount, OutputCount: outputCount, MaxArity: 2, LevelsBack: nodeCount);
        var functionGenes = new int[nodeCount];
        Array.Fill(functionGenes, addId);
        var genome = Genome<double>.FromGenes(
            shape,
            functionGenes,
            connections,
            new double[nodeCount],
            outputs,
            DoubleValueSemantics.Instance,
            StandardDoubleFunctionSet.Instance);
        return new Workload(name, genome, treeCompare);
    }

    /// <summary>Cadeia: n_k = Add(n_{k-1}, x1); nenhum nó compartilhado.</summary>
    public static Workload Chain(int n)
    {
        var conn = new int[n * 2];
        for (var k = 0; k < n; k++)
        {
            conn[2 * k] = k == 0 ? 0 : InputCount + k - 1;
            conn[2 * k + 1] = 1;
        }

        return Make("cadeia", n, 1, conn, [InputCount + n - 1], treeCompare: true);
    }

    /// <summary>Fan-out: n0 é argumento de todos os demais nós; todos os nós são outputs.</summary>
    public static Workload FanOut(int n)
    {
        var conn = new int[n * 2];
        conn[0] = 0;
        conn[1] = 1;
        for (var k = 1; k < n; k++)
        {
            conn[2 * k] = InputCount;
            conn[2 * k + 1] = 1;
        }

        var outputs = Enumerable.Range(0, n).Select(k => InputCount + k).ToArray();
        return Make("fan-out", n, n, conn, outputs, treeCompare: true);
    }

    /// <summary>Diamante: n0=Add(x0,x1); n1=Add(n0,x0); n2=Add(n0,x1); n3=Add(n1,n2).</summary>
    public static Workload Diamond()
    {
        var conn = new[]
        {
            0, 1,
            InputCount, 0,
            InputCount, 1,
            InputCount + 1, InputCount + 2,
        };
        return Make("diamante", 4, 1, conn, [InputCount + 3], treeCompare: true);
    }

    /// <summary>Diamantes em níveis: cada nível reconverge (a=Add(t,x0), b=Add(t,x1), c=Add(a,b)).</summary>
    public static Workload LevelsDiamond(int levels)
    {
        var n = 3 * levels;
        var conn = new int[n * 2];
        for (var j = 0; j < levels; j++)
        {
            var baseAddress = j == 0 ? 0 : InputCount + 3 * j - 1;
            var a = 3 * j;
            var b = 3 * j + 1;
            var c = 3 * j + 2;
            conn[2 * a] = baseAddress;
            conn[2 * a + 1] = 0;
            conn[2 * b] = baseAddress;
            conn[2 * b + 1] = 1;
            conn[2 * c] = InputCount + a;
            conn[2 * c + 1] = InputCount + b;
        }

        return Make("diamantes em niveis", n, 1, conn, [InputCount + n - 1], treeCompare: true);
    }
}

/// <summary>Cenário migrado de `Evo.Benchmarks` (cadeia sem compartilhamento, 32 nós, gate de contagens por endereço).</summary>
public sealed class ChainDagBenchmarkScenario : IBenchmarkScenario
{
    public string Id => "dag-chain-benchmark";

    public string Description => "Cadeia sem compartilhamento (32 nós); gate de contagens por endereço escalar/batch e paridade com árvore ingênua.";

    public BenchmarkScenarioResult Run(CancellationToken cancellationToken) => DagWorkloadSupport.Run(Id, DagWorkloadSupport.Chain(32));
}

/// <summary>Cenário migrado de `Evo.Benchmarks` (fan-out: n0 compartilhado por 31 nós, 32 outputs).</summary>
public sealed class FanOutDagBenchmarkScenario : IBenchmarkScenario
{
    public string Id => "dag-fanout-benchmark";

    public string Description => "Fan-out (n0 compartilhado por 31 nós, 32 outputs); gate de contagens por endereço escalar/batch e paridade com árvore ingênua.";

    public BenchmarkScenarioResult Run(CancellationToken cancellationToken) => DagWorkloadSupport.Run(Id, DagWorkloadSupport.FanOut(32));
}

/// <summary>Cenário migrado de `Evo.Benchmarks` (diamante simples: n0 -> (n1, n2) -> n3).</summary>
public sealed class DiamondDagBenchmarkScenario : IBenchmarkScenario
{
    public string Id => "dag-diamond-benchmark";

    public string Description => "Diamante simples (n0 -> (n1, n2) -> n3); gate de contagens por endereço escalar/batch e paridade com árvore ingênua.";

    public BenchmarkScenarioResult Run(CancellationToken cancellationToken) => DagWorkloadSupport.Run(Id, DagWorkloadSupport.Diamond());
}

/// <summary>Cenário migrado de `Evo.Benchmarks` (8 diamantes encadeados, reconvergência por nível).</summary>
public sealed class LevelsDiamondDagBenchmarkScenario : IBenchmarkScenario
{
    public string Id => "dag-levels-diamond-benchmark";

    public string Description => "Diamantes em níveis (8 níveis encadeados, 24 nós); gate de contagens por endereço escalar/batch e paridade com árvore ingênua.";

    public BenchmarkScenarioResult Run(CancellationToken cancellationToken) => DagWorkloadSupport.Run(Id, DagWorkloadSupport.LevelsDiamond(8));
}
