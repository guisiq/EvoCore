using System.Diagnostics;
using System.Numerics;
using System.Runtime.InteropServices;
using Evo.Core;
using Evo.Core.Compiled;
using Evo.Core.Functions;
using Evo.Core.Values;

namespace Evo.Benchmarks;

/// <summary>
/// Benchmark limitado da execução DAG (EVO-014). Contagens por endereço são gate
/// (exit code 1 em divergência); tempo e alocação são relatório medido, sem thresholds.
/// Rodadas instrumentadas (contadores) são separadas das rodadas de timing (sem contadores).
/// A árvore ingênua existe somente aqui, com expansão limitada por <see cref="TreeVisitCap"/>.
/// </summary>
public static class Program
{
    private const int CaseCount = 1024;
    private const int WarmRepetitions = 200;
    private const int ColdSamples = 5;
    private const long TreeVisitCap = 1_000_000;
    private const int InputCount = 2;

    private sealed record Workload(string Name, string Description, Genome<double> Genome, bool TreeCompare);

    public static int Main(string[]? args)
    {
        if (args is { Length: > 0 } && string.Equals(args[0], "ilgpu", StringComparison.OrdinalIgnoreCase))
        {
            return IlgpuBenchmarkProgram.Run();
        }

        var functions = StandardDoubleFunctionSet.Instance;
        var addId = FindAddFunctionId(functions);

        var workloads = new[]
        {
            ChainWorkload(addId, 32),
            FanOutWorkload(addId, 32),
            DiamondWorkload(addId),
            LevelsDiamondWorkload(addId, 8),
        };

        Console.WriteLine("Evo.Benchmarks — benchmark limitado da execução DAG (EVO-014)");
        Console.WriteLine($"configuracao={ConfigurationName()} runtime={RuntimeInformation.FrameworkDescription}");
        Console.WriteLine($"arquitetura={RuntimeInformation.ProcessArchitecture} vetorWidthDouble={Vector<double>.Count} vetorHardwareAcelerado={Vector.IsHardwareAccelerated}");
        Console.WriteLine($"tipo=double casos={CaseCount} repeticoesWarm={WarmRepetitions} repeticoesCold={ColdSamples} addId={addId}");
        Console.WriteLine("entradas=deterministicas ((i%97)*0.03-1.5); seed=nao aplicavel (sem geracao)");
        Console.WriteLine("arrays (double[], int[]) nao sao medidos como SIMD; apenas double e medido aqui.");

        var gateFailures = 0;
        foreach (var w in workloads)
        {
            gateFailures += RunWorkload(w, functions);
        }

        Console.WriteLine();
        Console.WriteLine(gateFailures == 0
            ? "GATE contagens e saidas: PASS"
            : $"GATE contagens e saidas: FAIL (divergencias={gateFailures})");
        return gateFailures == 0 ? 0 : 1;
    }

    private static int RunWorkload(Workload w, StandardDoubleFunctionSet functions)
    {
        var g = w.Genome;
        var shape = g.Shape;
        var inputCount = shape.InputCount;
        var nodeCount = shape.NodeCount;
        var outputCount = shape.OutputCount;
        var values = DoubleValueSemantics.Instance;
        var inputs = MakeInputs(inputCount, CaseCount);
        var outputs = new double[CaseCount * outputCount];
        var failures = 0;

        var active = ReachableNodes(g);
        var activeCount = active.Count(a => a);
        var activeArcs = 0;
        for (var n = 0; n < nodeCount; n++)
        {
            if (!active[n])
            {
                continue;
            }

            for (var slot = 0; slot < shape.MaxArity; slot++)
            {
                if (g.GetConnection(n, slot) >= inputCount)
                {
                    activeArcs++;
                }
            }
        }

        Console.WriteLine();
        Console.WriteLine($"== {w.Name}: {w.Description}");
        Console.WriteLine($"shape: inputs={shape.InputCount} nodes={nodeCount} outputs={outputCount} maxArity={shape.MaxArity} levelsBack={shape.LevelsBack}");
        Console.WriteLine($"nos totais={nodeCount} nos ativos={activeCount} arestas ativas (no->no)={activeArcs} outputs={outputCount}");

        // Rodada instrumentada escalar (contadores ativos; sem medição de tempo).
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
        Console.WriteLine($"contagem escalar por endereco: divergencias={scalarMismatch} (esperado: ativo={CaseCount}, inativo=0)");

        // Rodada instrumentada SIMD/batch (contadores ativos; sem medição de tempo).
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
        Console.WriteLine($"contagem batch: backend={countReport.Backend} fallback={countReport.FallbackReason ?? "(nenhum)"} divergencias={simdMismatch}");

        // Árvore ingênua (somente benchmark) com expansão limitada; contagens do caso 0.
        var treeOutputs = new double[CaseCount * outputCount];
        var treeVisits = new int[nodeCount];
        var treeOk = false;
        long treeTotalVisits = 0;
        if (w.TreeCompare)
        {
            try
            {
                for (var c = 0; c < CaseCount; c++)
                {
                    long caseBudget = TreeVisitCap;
                    var scratch = c == 0 ? treeVisits : new int[nodeCount];
                    for (var o = 0; o < outputCount; o++)
                    {
                        treeOutputs[c * outputCount + o] = TreeEval(g, functions, inputs.AsSpan(c * inputCount, inputCount), OutputAddress(g, o), scratch, ref caseBudget);
                    }
                }

                treeOk = true;
                foreach (var v in treeVisits)
                {
                    treeTotalVisits += v;
                }
            }
            catch (InvalidOperationException ex)
            {
                Console.WriteLine($"arvore ingenua: nao expandida ({ex.Message})");
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
            Console.WriteLine("arvore ingenua (caso 0, por endereco): " + string.Join(" ", treeVisits.Select((v, n) => $"n{n}:dag={(active[n] ? 1 : 0)}/arvore={v}")));
            Console.WriteLine($"arvore ingenua: visitas totais por caso={treeTotalVisits} execucoes DAG por caso={activeCount} multiplicacao={(double)treeTotalVisits / Math.Max(1, activeCount):F2}x saidas divergentes={treeMismatch}");
        }

        // Compilação fria: avaliador novo a cada amostra, uma avaliação (inclui compilação do plano).
        var coldUs = new List<double>();
        for (var s = 0; s < ColdSamples; s++)
        {
            var coldEval = new CompiledGenomeEvaluator<double>(functions, values);
            var sw = Stopwatch.StartNew();
            coldEval.Evaluate(g, inputs.AsSpan(0, inputCount), outputs.AsSpan(0, outputCount));
            sw.Stop();
            coldUs.Add(sw.Elapsed.TotalMilliseconds * 1000.0);
        }

        coldUs.Sort();
        Console.WriteLine($"compilacao fria (mediana de {ColdSamples}, 1 avaliacao): {coldUs[ColdSamples / 2]:F1} us");

        // Avaliação aquecida escalar com reuso de plano (rodada de timing, sem contadores).
        var warmEval = new CompiledGenomeEvaluator<double>(functions, values);
        ScalarPass(warmEval, g, inputs, outputs, inputCount, outputCount);
        var allocBefore = GC.GetAllocatedBytesForCurrentThread();
        var sw1 = Stopwatch.StartNew();
        for (var r = 0; r < WarmRepetitions; r++)
        {
            ScalarPass(warmEval, g, inputs, outputs, inputCount, outputCount);
        }

        sw1.Stop();
        var allocScalar = GC.GetAllocatedBytesForCurrentThread() - allocBefore;
        var evals = (double)CaseCount * WarmRepetitions;
        var secScalar = sw1.Elapsed.TotalSeconds;
        Console.WriteLine($"escalar aquecido (reuso de plano): ns/avaliacao={sw1.Elapsed.TotalMilliseconds * 1e6 / evals:F1} casos/s={evals / secScalar:F0} nos ativos/s={evals * activeCount / secScalar:F0} bytes/avaliacao={(double)allocScalar / evals:F1} cacheHits={warmEval.CacheHits} cacheMisses={warmEval.CacheMisses}");

        // Batch SIMD aquecido (rodada de timing, sem contadores).
        var simdTime = new SimdGenomeEvaluator<double>(functions, values);
        var report = simdTime.Evaluate(g, inputs, CaseCount, outputs);
        var allocBeforeSimd = GC.GetAllocatedBytesForCurrentThread();
        var sw2 = Stopwatch.StartNew();
        for (var r = 0; r < WarmRepetitions; r++)
        {
            report = simdTime.Evaluate(g, inputs, CaseCount, outputs);
        }

        sw2.Stop();
        var allocSimd = GC.GetAllocatedBytesForCurrentThread() - allocBeforeSimd;
        var secSimd = sw2.Elapsed.TotalSeconds;
        Console.WriteLine($"batch aquecido: solicitado=Simd efetivo={report.Backend} fallback={report.FallbackReason ?? "(nenhum)"} vetorWidth={report.VectorWidth} blocos={report.VectorBlocks} cauda={report.TailCases} ns/avaliacao={sw2.Elapsed.TotalMilliseconds * 1e6 / evals:F1} casos/s={evals / secSimd:F0} nos ativos/s={evals * activeCount / secSimd:F0} bytes/avaliacao={(double)allocSimd / evals:F1}");

        // Árvore ingênua aquecida (somente benchmark).
        if (treeOk)
        {
            var allocBeforeTree = GC.GetAllocatedBytesForCurrentThread();
            var sw3 = Stopwatch.StartNew();
            var scratch = new int[nodeCount];
            for (var r = 0; r < WarmRepetitions; r++)
            {
                for (var c = 0; c < CaseCount; c++)
                {
                    long budget = long.MaxValue;
                    for (var o = 0; o < outputCount; o++)
                    {
                        treeOutputs[c * outputCount + o] = TreeEval(g, functions, inputs.AsSpan(c * inputCount, inputCount), OutputAddress(g, o), scratch, ref budget);
                    }
                }
            }

            sw3.Stop();
            var allocTree = GC.GetAllocatedBytesForCurrentThread() - allocBeforeTree;
            var secTree = sw3.Elapsed.TotalSeconds;
            Console.WriteLine($"arvore ingenua aquecida (somente benchmark): ns/avaliacao={sw3.Elapsed.TotalMilliseconds * 1e6 / evals:F1} casos/s={evals / secTree:F0} bytes/avaliacao={(double)allocTree / evals:F1} razaoTempoArvore/DAG={(secTree / secScalar):F2}x razaoAlocArvore/DAG={(allocScalar == 0 ? double.NaN : (double)allocTree / allocScalar):F2}x");
        }

        Console.WriteLine($"divergencias nesta workload={failures}");
        return failures;
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

    private static int FindAddFunctionId(StandardDoubleFunctionSet functions)
    {
        for (var i = 0; i < functions.Count; i++)
        {
            var f = functions.Get(i);
            if (f.Arity == 2 && f.Id.Contains("add", StringComparison.OrdinalIgnoreCase))
            {
                return i;
            }
        }

        throw new InvalidOperationException("função binária Add não encontrada no conjunto double");
    }

    private static Workload Make(string name, string description, int nodeCount, int outputCount, int addId, int[] connections, int[] outputs, bool treeCompare)
    {
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
        return new Workload(name, description, genome, treeCompare);
    }

    // Cadeia: n_k = Add(n_{k-1}, x1); nenhum nó compartilhado.
    private static Workload ChainWorkload(int addId, int n)
    {
        var conn = new int[n * 2];
        for (var k = 0; k < n; k++)
        {
            conn[2 * k] = k == 0 ? 0 : InputCount + k - 1;
            conn[2 * k + 1] = 1;
        }

        return Make("cadeia", $"cadeia sem compartilhamento, {n} nos, 1 output", n, 1, addId, conn, new[] { InputCount + n - 1 }, treeCompare: true);
    }

    // Fan-out: n0 é argumento de todos os demais nós; todos os nós são outputs.
    private static Workload FanOutWorkload(int addId, int n)
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
        return Make("fan-out", $"n0 compartilhado por {n - 1} nos, {n} outputs", n, n, addId, conn, outputs, treeCompare: true);
    }

    // Diamante: n0=Add(x0,x1); n1=Add(n0,x0); n2=Add(n0,x1); n3=Add(n1,n2).
    private static Workload DiamondWorkload(int addId)
    {
        var conn = new[]
        {
            0, 1,
            InputCount, 0,
            InputCount, 1,
            InputCount + 1, InputCount + 2,
        };
        return Make("diamante", "diamante simples: n0 -> (n1, n2) -> n3", 4, 1, addId, conn, new[] { InputCount + 3 }, treeCompare: true);
    }

    // Diamantes em níveis: cada nível j reconverge (a=Add(t,x0), b=Add(t,x1), c=Add(a,b)).
    private static Workload LevelsDiamondWorkload(int addId, int levels)
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

        return Make("diamantes em niveis", $"{levels} diamantes encadeados ({n} nos), reconvergencia por nível", n, 1, addId, conn, new[] { InputCount + n - 1 }, treeCompare: true);
    }

    private static string ConfigurationName()
    {
#if DEBUG
        return "Debug";
#else
        return "Release";
#endif
    }
}