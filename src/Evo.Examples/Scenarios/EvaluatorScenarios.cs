using System.Globalization;
using Evo.Core;
using Evo.Core.Compiled;
using Evo.Core.Functions;
using Evo.Core.Values;

namespace Evo.Examples.Scenarios;

/// <summary>Function set que expõe apenas os primeiros <c>N</c> IDs de um conjunto fonte (comparação de complexidade).</summary>
internal sealed class SubsetDoubleFunctionSet : IFunctionSet<double>
{
    private readonly IFunctionSet<double> _source;

    public SubsetDoubleFunctionSet(string id, IFunctionSet<double> source, int count)
    {
        Id = id;
        _source = source;
        Count = Math.Max(1, Math.Min(count, source.Count));
    }

    public string Id { get; }

    public int Count { get; }

    public IFunction<double> Get(int functionId) => _source.Get(functionId % Count);
}

/// <summary>Helpers compartilhados pelos cenários de equivalência de avaliadores.</summary>
internal static class EvaluatorScenarioSupport
{
    public static Dictionary<string, string> Params(params (string Key, object Value)[] items) =>
        EvolutionScenarioSupport.Params(items);

    public static bool Equivalent(double a, double b) =>
        (double.IsNaN(a) && double.IsNaN(b)) || Math.Abs(a - b) <= 1e-9 * Math.Max(1.0, Math.Max(Math.Abs(a), Math.Abs(b)));

    public static int FindFunctionId(IFunctionSet<double> functions, int arity, string nameContains)
    {
        for (var i = 0; i < functions.Count; i++)
        {
            var f = functions.Get(i);
            if (f.Arity == arity && f.Id.Contains(nameContains, StringComparison.OrdinalIgnoreCase))
            {
                return i;
            }
        }

        throw new InvalidOperationException($"função arity={arity} contendo '{nameContains}' não encontrada.");
    }

    /// <summary>Cadeia esquerda-associativa de <paramref name="levels"/> nós Add(anterior, novo input).</summary>
    public static Genome<double> BuildChain(int inputCount, int levels, int addId)
    {
        var conn = new int[levels * 2];
        for (var k = 0; k < levels; k++)
        {
            conn[2 * k] = k == 0 ? 0 : inputCount + k - 1;
            conn[2 * k + 1] = Math.Min(k + 1, inputCount - 1);
        }

        var shape = new GenomeShape(InputCount: inputCount, NodeCount: levels, OutputCount: 1, MaxArity: 2, LevelsBack: levels);
        var functionGenes = new int[levels];
        Array.Fill(functionGenes, addId);
        return Genome<double>.FromGenes(shape, functionGenes, conn, new double[levels], [inputCount + levels - 1],
            DoubleValueSemantics.Instance, StandardDoubleFunctionSet.Instance);
    }

    /// <summary>Compara reference/compiled/simd/ilgpu (quando disponível) para um genoma sobre um lote de casos.</summary>
    public static (bool Ok, Dictionary<string, string> Gates, string? IlgpuFallback) CheckParity(
        Genome<double> genome, IFunctionSet<double> functions, double[] inputs, int caseCount)
    {
        var inputCount = genome.Shape.InputCount;
        var outputCount = genome.Shape.OutputCount;
        var values = DoubleValueSemantics.Instance;
        var reference = new ReferenceGenomeEvaluator<double>(functions);
        var compiled = new CompiledGenomeEvaluator<double>(functions, values);
        var simd = new SimdGenomeEvaluator<double>(functions, values);

        var refOut = new double[caseCount * outputCount];
        var compOut = new double[caseCount * outputCount];
        for (var c = 0; c < caseCount; c++)
        {
            reference.Evaluate(genome, inputs.AsSpan(c * inputCount, inputCount), refOut.AsSpan(c * outputCount, outputCount));
            compiled.Evaluate(genome, inputs.AsSpan(c * inputCount, inputCount), compOut.AsSpan(c * outputCount, outputCount));
        }

        var simdOut = new double[caseCount * outputCount];
        simd.Evaluate(genome, inputs, caseCount, simdOut);

        var compiledOk = true;
        var simdOk = true;
        for (var i = 0; i < refOut.Length; i++)
        {
            compiledOk &= Equivalent(refOut[i], compOut[i]);
            simdOk &= Equivalent(refOut[i], simdOut[i]);
        }

        var gates = new Dictionary<string, string>(StringComparer.Ordinal)
        {
            ["reference_vs_compiled"] = compiledOk.ToString(),
            ["reference_vs_simd"] = simdOk.ToString(),
        };

        string? ilgpuFallback = null;
        var ilgpuOk = true;
        try
        {
            using var ilgpu = new Evo.Ilgpu.IlgpuGenomeEvaluator<double>(functions);
            var ilgpuOut = new double[caseCount * outputCount];
            for (var c = 0; c < caseCount; c++)
            {
                ilgpu.Evaluate(genome, inputs.AsSpan(c * inputCount, inputCount), ilgpuOut.AsSpan(c * outputCount, outputCount));
            }

            for (var i = 0; i < refOut.Length; i++)
            {
                ilgpuOk &= Equivalent(refOut[i], ilgpuOut[i]);
            }

            gates["reference_vs_ilgpu"] = ilgpuOk.ToString();
        }
        catch (Exception ex)
        {
            ilgpuFallback = ex.GetType().Name + ": " + ex.Message;
        }

        var ok = compiledOk && simdOk && ilgpuOk;
        return (ok, gates, ilgpuFallback);
    }
}

/// <summary>Throughput frio/quente de avaliadores sobre uma matriz sintética determinística 1000x50 (seed 42).</summary>
public sealed class EvaluatorThroughputSyntheticScenario : IBenchmarkScenario
{
    public string Id => "evaluator-throughput-synthetic";

    public string Description => "Throughput frio/quente de referência/compilado/SIMD/ILGPU disponível, matriz 1000x50 (seed 42), genoma de profundidade 8, 100 repetições.";

    public BenchmarkScenarioResult Run(CancellationToken cancellationToken)
    {
        const int Rows = 1000;
        const int Cols = 50;
        const int Depth = 8;
        const int Repetitions = 100;
        var rnd = new Random(42);
        var inputs = new double[Rows * Cols];
        for (var i = 0; i < inputs.Length; i++)
        {
            inputs[i] = rnd.NextDouble() * 4.0 - 2.0;
        }

        var functions = StandardDoubleFunctionSet.Instance;
        var addId = EvaluatorScenarioSupport.FindFunctionId(functions, 2, "add");
        var genome = EvaluatorScenarioSupport.BuildChain(Cols, Depth, addId);
        var (parityOk, parityGates, ilgpuFallback) = EvaluatorScenarioSupport.CheckParity(genome, functions, inputs, Rows);

        var values = DoubleValueSemantics.Instance;
        var outputs = new double[Rows];
        var measurements = new Dictionary<string, string>(StringComparer.Ordinal);

        MeasureCold("reference", () => new ReferenceGenomeEvaluator<double>(functions), (e, c) => e.Evaluate(genome, inputs.AsSpan(c * Cols, Cols), outputs.AsSpan(c, 1)), Rows, measurements);
        MeasureCold("compiled", () => new CompiledGenomeEvaluator<double>(functions, values), (e, c) => e.Evaluate(genome, inputs.AsSpan(c * Cols, Cols), outputs.AsSpan(c, 1)), Rows, measurements);

        var warmReference = new ReferenceGenomeEvaluator<double>(functions);
        var swRef = System.Diagnostics.Stopwatch.StartNew();
        for (var r = 0; r < Repetitions; r++)
        {
            for (var c = 0; c < Rows; c++)
            {
                warmReference.Evaluate(genome, inputs.AsSpan(c * Cols, Cols), outputs.AsSpan(c, 1));
            }
        }

        swRef.Stop();
        measurements["warm_reference_ns_per_eval"] = (swRef.Elapsed.TotalMilliseconds * 1e6 / (Rows * (double)Repetitions)).ToString("F1", CultureInfo.InvariantCulture);

        var warmCompiled = new CompiledGenomeEvaluator<double>(functions, values);
        var swComp = System.Diagnostics.Stopwatch.StartNew();
        for (var r = 0; r < Repetitions; r++)
        {
            for (var c = 0; c < Rows; c++)
            {
                warmCompiled.Evaluate(genome, inputs.AsSpan(c * Cols, Cols), outputs.AsSpan(c, 1));
            }
        }

        swComp.Stop();
        measurements["warm_compiled_ns_per_eval"] = (swComp.Elapsed.TotalMilliseconds * 1e6 / (Rows * (double)Repetitions)).ToString("F1", CultureInfo.InvariantCulture);

        var warmSimd = new SimdGenomeEvaluator<double>(functions, values);
        var simdOutputs = new double[Rows];
        var swSimd = System.Diagnostics.Stopwatch.StartNew();
        for (var r = 0; r < Repetitions; r++)
        {
            warmSimd.Evaluate(genome, inputs, Rows, simdOutputs);
        }

        swSimd.Stop();
        measurements["warm_simd_ns_per_eval"] = (swSimd.Elapsed.TotalMilliseconds * 1e6 / (Rows * (double)Repetitions)).ToString("F1", CultureInfo.InvariantCulture);

        if (ilgpuFallback is not null)
        {
            measurements["ilgpu_fallback"] = ilgpuFallback;
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
            parityGates,
            measurements,
            raw,
            EvaluatorScenarioSupport.Params(("rows", Rows), ("cols", Cols), ("depth", Depth), ("repetitions", Repetitions)),
            parityOk ? "paridade confirmada entre avaliadores disponíveis; tempo/alocação são medições" : "divergência de paridade entre avaliadores");
    }

    private static void MeasureCold(string name, Func<IGenomeEvaluator<double>> factory, Action<IGenomeEvaluator<double>, int> evaluateOne, int rows, Dictionary<string, string> measurements)
    {
        var sw = System.Diagnostics.Stopwatch.StartNew();
        var evaluator = factory();
        evaluateOne(evaluator, 0);
        sw.Stop();
        measurements[$"cold_{name}_us"] = (sw.Elapsed.TotalMilliseconds * 1000.0).ToString("F1", CultureInfo.InvariantCulture);
    }
}

/// <summary>Matriz de paridade reference/compiled/SIMD/ILGPU para (inputs, LevelsBack) requeridos.</summary>
public sealed class EvaluatorParityMatrixScenario : IBenchmarkScenario
{
    public string Id => "evaluator-parity-matrix";

    public string Description => "Paridade reference/compiled/SIMD/ILGPU disponível para (inputs, LevelsBack) = (5,4),(10,4),(20,6),(50,8).";

    private static readonly (int Inputs, int LevelsBack)[] Cases = [(5, 4), (10, 4), (20, 6), (50, 8)];

    public BenchmarkScenarioResult Run(CancellationToken cancellationToken)
    {
        var functions = StandardDoubleFunctionSet.Instance;
        var addId = EvaluatorScenarioSupport.FindFunctionId(functions, 2, "add");
        var gates = new Dictionary<string, string>(StringComparer.Ordinal);
        var measurements = new Dictionary<string, string>(StringComparer.Ordinal);
        var allOk = true;
        var rnd = new Random(42);
        foreach (var (inputs, levelsBack) in Cases)
        {
            const int CaseCount = 64;
            var matrix = new double[CaseCount * inputs];
            for (var i = 0; i < matrix.Length; i++)
            {
                matrix[i] = rnd.NextDouble() * 6.0 - 3.0;
            }

            var genome = EvaluatorScenarioSupport.BuildChain(inputs, levelsBack, addId);
            var (ok, caseGates, ilgpuFallback) = EvaluatorScenarioSupport.CheckParity(genome, functions, matrix, CaseCount);
            allOk &= ok;
            var key = $"inputs{inputs}_levelsBack{levelsBack}";
            foreach (var (gateKey, gateValue) in caseGates)
            {
                gates[$"{key}_{gateKey}"] = gateValue;
            }

            if (ilgpuFallback is not null)
            {
                measurements[$"{key}_ilgpu_fallback"] = ilgpuFallback;
            }
        }

        var raw = new Dictionary<string, string>(StringComparer.Ordinal) { ["seed"] = "42" };
        return new BenchmarkScenarioResult(
            Id,
            allOk,
            gates,
            measurements,
            raw,
            EvaluatorScenarioSupport.Params(("cases", "5x4,10x4,20x6,50x8")),
            allOk ? "paridade confirmada em todas as combinações (inputs, LevelsBack)" : "divergência de paridade em ao menos uma combinação");
    }
}

/// <summary>Equivalência de avaliadores para uma função binária, com vetores positivos e negativos.</summary>
public sealed class BinaryOperatorParityScenario : IBenchmarkScenario
{
    public string Id => "binary-operator-parity";

    public string Description => "Equivalência de avaliadores para um operador binário com valores positivos e negativos.";

    public BenchmarkScenarioResult Run(CancellationToken cancellationToken)
    {
        var functions = StandardDoubleFunctionSet.Instance;
        var addId = EvaluatorScenarioSupport.FindFunctionId(functions, 2, "add");
        var shape = new GenomeShape(InputCount: 2, NodeCount: 1, OutputCount: 1, MaxArity: 2, LevelsBack: 1);
        var genome = Genome<double>.FromGenes(shape, [addId], [0, 1], [0.0], [2],
            DoubleValueSemantics.Instance, functions);
        var inputs = new double[] { 3.5, -2.0, -7.25, 4.0, 0.0, 0.0, -1.0, -1.0 };
        var (ok, gates, ilgpuFallback) = EvaluatorScenarioSupport.CheckParity(genome, functions, inputs, 4);
        var measurements = ilgpuFallback is null
            ? new Dictionary<string, string>(StringComparer.Ordinal)
            : new Dictionary<string, string>(StringComparer.Ordinal) { ["ilgpu_fallback"] = ilgpuFallback };

        return new BenchmarkScenarioResult(
            Id, ok, gates, measurements,
            new Dictionary<string, string>(StringComparer.Ordinal) { ["cases"] = "4" },
            EvaluatorScenarioSupport.Params(("function", "add")),
            ok ? "avaliadores equivalentes para a função binária" : "divergência de avaliadores");
    }
}

/// <summary>Equivalência para um grafo binário aninhado com um parâmetro explícito, valores positivos e negativos.</summary>
public sealed class ParameterizedOperatorParityScenario : IBenchmarkScenario
{
    public string Id => "parameterized-operator-parity";

    public string Description => "Equivalência de avaliadores para um grafo binário com parâmetro explícito, valores positivos e negativos.";

    public BenchmarkScenarioResult Run(CancellationToken cancellationToken)
    {
        var functions = StandardDoubleFunctionSet.Instance;
        var addId = EvaluatorScenarioSupport.FindFunctionId(functions, 2, "add");
        var constantId = EvaluatorScenarioSupport.FindFunctionId(functions, 0, "constant");
        // n0 = Constant(param=2.5); n1 = Add(x0, n0); n2 = Add(n1, x1).
        var shape = new GenomeShape(InputCount: 2, NodeCount: 3, OutputCount: 1, MaxArity: 2, LevelsBack: 3);
        var functionGenes = new[] { constantId, addId, addId };
        var connections = new[] { 0, 0, 0, 2, 3, 1 };
        var parameters = new[] { 2.5, 0.0, 0.0 };
        var genome = Genome<double>.FromGenes(shape, functionGenes, connections, parameters, [4],
            DoubleValueSemantics.Instance, functions);
        var inputs = new double[] { 3.0, -5.0, -1.5, 2.0 };
        var (ok, gates, ilgpuFallback) = EvaluatorScenarioSupport.CheckParity(genome, functions, inputs, 2);
        var measurements = ilgpuFallback is null
            ? new Dictionary<string, string>(StringComparer.Ordinal)
            : new Dictionary<string, string>(StringComparer.Ordinal) { ["ilgpu_fallback"] = ilgpuFallback };

        return new BenchmarkScenarioResult(
            Id, ok, gates, measurements,
            new Dictionary<string, string>(StringComparer.Ordinal) { ["cases"] = "2", ["parameter"] = "2.5" },
            EvaluatorScenarioSupport.Params(("function", "add+constant")),
            ok ? "avaliadores equivalentes para o grafo parametrizado" : "divergência de avaliadores");
    }
}

/// <summary>Equivalência de avaliadores para uma cadeia esquerda-associativa de dez níveis.</summary>
public sealed class DeepChainParityScenario : IBenchmarkScenario
{
    public string Id => "deep-chain-parity";

    public string Description => "Equivalência de avaliadores para um grafo ativo encadeado (esquerda-associativo) de dez níveis.";

    public BenchmarkScenarioResult Run(CancellationToken cancellationToken)
    {
        var functions = StandardDoubleFunctionSet.Instance;
        var addId = EvaluatorScenarioSupport.FindFunctionId(functions, 2, "add");
        const int Inputs = 10;
        var genome = EvaluatorScenarioSupport.BuildChain(Inputs, 10, addId);
        var inputs = new double[Inputs];
        for (var i = 0; i < Inputs; i++)
        {
            inputs[i] = (i % 2 == 0 ? 1 : -1) * (0.5 + i);
        }

        var (ok, gates, ilgpuFallback) = EvaluatorScenarioSupport.CheckParity(genome, functions, inputs, 1);
        var measurements = ilgpuFallback is null
            ? new Dictionary<string, string>(StringComparer.Ordinal)
            : new Dictionary<string, string>(StringComparer.Ordinal) { ["ilgpu_fallback"] = ilgpuFallback };

        return new BenchmarkScenarioResult(
            Id, ok, gates, measurements,
            new Dictionary<string, string>(StringComparer.Ordinal) { ["inputs"] = Inputs.ToString(CultureInfo.InvariantCulture) },
            EvaluatorScenarioSupport.Params(("levels", 10)),
            ok ? "avaliadores equivalentes para a cadeia de dez níveis" : "divergência de avaliadores");
    }
}

/// <summary>Equivalência para um grafo balanceado envolto por operações parametrizadas (oito inputs, dois parâmetros).</summary>
public sealed class BalancedParameterParityScenario : IBenchmarkScenario
{
    public string Id => "balanced-parameter-parity";

    public string Description => "Equivalência de avaliadores para um grafo balanceado com dois parâmetros explícitos (oito inputs).";

    public BenchmarkScenarioResult Run(CancellationToken cancellationToken)
    {
        var functions = StandardDoubleFunctionSet.Instance;
        var addId = EvaluatorScenarioSupport.FindFunctionId(functions, 2, "add");
        var constantId = EvaluatorScenarioSupport.FindFunctionId(functions, 0, "constant");
        const int Inputs = 8;
        // n0..n3: pares Add(x_{2i}, x_{2i+1}); n4 = Add(n0,n1); n5 = Add(n2,n3); n6 = Constant(p0);
        // n7 = Constant(p1); n8 = Add(n4,n6); n9 = Add(n5,n7); output = Add(n8,n9).
        var shape = new GenomeShape(InputCount: Inputs, NodeCount: 11, OutputCount: 1, MaxArity: 2, LevelsBack: 11);
        var functionGenes = new[] { addId, addId, addId, addId, addId, addId, constantId, constantId, addId, addId, addId };
        var connections = new[]
        {
            0, 1,
            2, 3,
            4, 5,
            6, 7,
            Inputs + 0, Inputs + 1,
            Inputs + 2, Inputs + 3,
            0, 0,
            0, 0,
            Inputs + 4, Inputs + 6,
            Inputs + 5, Inputs + 7,
            Inputs + 8, Inputs + 9,
        };
        var parameters = new double[11];
        parameters[6] = 1.25;
        parameters[7] = -0.75;
        var genome = Genome<double>.FromGenes(shape, functionGenes, connections, parameters, [Inputs + 10],
            DoubleValueSemantics.Instance, functions);
        var inputs = new double[] { 1.0, -2.0, 3.0, -4.0, -5.0, 6.0, -7.0, 8.0 };
        var (ok, gates, ilgpuFallback) = EvaluatorScenarioSupport.CheckParity(genome, functions, inputs, 1);
        var measurements = ilgpuFallback is null
            ? new Dictionary<string, string>(StringComparer.Ordinal)
            : new Dictionary<string, string>(StringComparer.Ordinal) { ["ilgpu_fallback"] = ilgpuFallback };

        return new BenchmarkScenarioResult(
            Id, ok, gates, measurements,
            new Dictionary<string, string>(StringComparer.Ordinal) { ["inputs"] = Inputs.ToString(CultureInfo.InvariantCulture), ["parameters"] = "2" },
            EvaluatorScenarioSupport.Params(("parameters", "1.25,-0.75")),
            ok ? "avaliadores equivalentes para o grafo balanceado parametrizado" : "divergência de avaliadores");
    }
}
