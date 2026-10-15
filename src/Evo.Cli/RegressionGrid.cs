using System.Collections.Concurrent;
using System.Globalization;
using System.Text;
using Evo.Core;

namespace Evo.Cli;

/// <summary>Estratégia de busca usada por uma configuração da grade.</summary>
public enum RegressionGridStrategy
{
    /// <summary>Engine evolutivo completo.</summary>
    Engine,

    /// <summary>Amostragem aleatória independente com o mesmo orçamento de avaliações.</summary>
    RandomSearch,
}

/// <summary>Papel da configuração no gráfico: referência pontilhada ou variação medida.</summary>
public enum RegressionGridKind
{
    /// <summary>Referência de controle; desenhada pontilhada.</summary>
    Baseline,

    /// <summary>Variação do engine sob estudo; desenhada sólida.</summary>
    Variation,
}

/// <summary>Problema da grade, identificado por família e nível de complexidade.</summary>
public sealed record RegressionGridProblem(
    string Family,
    int Level,
    RegressionSuiteProblem Definition)
{
    public string Id => Definition.Id;
}

/// <summary>Configuração reproduzível de busca aplicada a todos os problemas da grade.</summary>
public sealed record RegressionGridConfiguration(
    string Id,
    RegressionGridKind Kind,
    RegressionGridStrategy Strategy,
    EvolutionOptions Options,
    string Description);

/// <summary>
/// Famílias de regressão univariada representáveis pelo conjunto padrão de funções
/// (<c>Add</c>, <c>Subtract</c>, <c>Multiply</c>, <c>ProtectedDivide</c>, <c>Negate</c>,
/// <c>Abs</c>, <c>Square</c>, <c>ProtectedSqrt</c>, <c>ProtectedLog</c>, <c>Min</c>,
/// <c>Max</c>, <c>GreaterThan</c>, <c>Constant</c>). Cada família tem três níveis de
/// complexidade crescente sobre o mesmo domínio, isolando uma dificuldade distinta:
/// grau polinomial, composição logarítmica, divisão protegida, radiciação aninhada e
/// não suavidade.
/// </summary>
public static class RegressionGridCatalog
{
    public static IReadOnlyList<RegressionGridProblem> All { get; } =
    [
        new("poly", 1, new("poly-1", "x^3 + x^2 + x", -1.0, 1.0, x => PowSum(x, 3))),
        new("poly", 2, new("poly-2", "x^4 + x^3 + x^2 + x", -1.0, 1.0, x => PowSum(x, 4))),
        new("poly", 3, new("poly-3", "x^6 + x^5 + x^4 + x^3 + x^2 + x", -1.0, 1.0, x => PowSum(x, 6))),

        new("log", 1, new("log-1", "log(x + 1)", 0.0, 2.0, x => Math.Log(x + 1.0))),
        new("log", 2, new("log-2", "log(x + 1) + log(x^2 + 1)", 0.0, 2.0,
            x => Math.Log(x + 1.0) + Math.Log((x * x) + 1.0))),
        new("log", 3, new("log-3", "log(x + 1) + log(x^2 + 1) + log(x^3 + 1)", 0.0, 2.0,
            x => Math.Log(x + 1.0) + Math.Log((x * x) + 1.0) + Math.Log((x * x * x) + 1.0))),

        new("rational", 1, new("rational-1", "1 / (1 + x^2)", -1.0, 1.0, x => 1.0 / (1.0 + (x * x)))),
        new("rational", 2, new("rational-2", "x / (1 + x^2)", -1.0, 1.0, x => x / (1.0 + (x * x)))),
        new("rational", 3, new("rational-3", "(x^2 + x) / (1 + x^2 + x^4)", -1.0, 1.0,
            x => ((x * x) + x) / (1.0 + (x * x) + (x * x * x * x)))),

        new("root", 1, new("root-1", "sqrt(x)", 0.0, 4.0, Math.Sqrt)),
        new("root", 2, new("root-2", "sqrt(x) + x", 0.0, 4.0, x => Math.Sqrt(x) + x)),
        new("root", 3, new("root-3", "sqrt(x + sqrt(x))", 0.0, 4.0, x => Math.Sqrt(x + Math.Sqrt(x)))),

        new("nonsmooth", 1, new("nonsmooth-1", "|x|", -2.0, 2.0, Math.Abs)),
        new("nonsmooth", 2, new("nonsmooth-2", "max(x, x^2)", -2.0, 2.0, x => Math.Max(x, x * x))),
        new("nonsmooth", 3, new("nonsmooth-3", "|x| * x + max(x, 0)", -2.0, 2.0,
            x => (Math.Abs(x) * x) + Math.Max(x, 0.0))),
    ];

    private static double PowSum(double x, int highestPower)
    {
        var sum = x;
        var power = x;
        for (var exponent = 2; exponent <= highestPower; exponent++)
        {
            power *= x;
            sum += power;
        }

        return sum;
    }
}

/// <summary>
/// Duas referências de controle pontilhadas e três variações do engine. As referências
/// delimitam o que é atribuível à busca evolutiva: amostragem aleatória sem nenhuma
/// pressão seletiva e deriva com torneio de tamanho 1 e sem crossover (apenas elitismo
/// e mutação). As variações só mexem em orçamento, nunca em contratos normativos.
/// </summary>
public static class RegressionGridConfigurations
{
    public static IReadOnlyList<RegressionGridConfiguration> All { get; } =
    [
        new("baseline-random-search", RegressionGridKind.Baseline, RegressionGridStrategy.RandomSearch,
            new EvolutionOptions(),
            "Amostragem aleatória independente com o mesmo orçamento de avaliações do evo-default."),
        new("baseline-drift", RegressionGridKind.Baseline, RegressionGridStrategy.Engine,
            new EvolutionOptions { TournamentSize = 1, CrossoverRate = 0.0 },
            "Deriva: pai escolhido uniformemente, sem crossover, apenas elitismo e mutação."),
        new("evo-default", RegressionGridKind.Variation, RegressionGridStrategy.Engine,
            new EvolutionOptions(),
            "Defaults normativos: população 500, 200 gerações, torneio 5, crossover 0,5."),
        new("evo-pop1000", RegressionGridKind.Variation, RegressionGridStrategy.Engine,
            new EvolutionOptions { PopulationSize = 1000 },
            "Dobra a população mantendo 200 gerações."),
        new("evo-gen400", RegressionGridKind.Variation, RegressionGridStrategy.Engine,
            new EvolutionOptions { MaxGenerations = 400 },
            "Dobra as gerações mantendo população 500."),
    ];
}

/// <summary>Uma linha tidy: uma geração observada de uma execução.</summary>
public sealed record RegressionGridObservation(
    string Experiment,
    RegressionGridKind Kind,
    string Family,
    int Level,
    string Problem,
    ulong Seed,
    int Generation,
    long Evaluations,
    double TrainingMse,
    double ValidationMse,
    double ValidationNmse,
    DagReuseMetrics Best,
    DagReuseAverages PopulationMean);

/// <summary>Resumo de uma execução completa (configuração x problema x seed).</summary>
public sealed record RegressionGridRun(
    string Experiment,
    RegressionGridKind Kind,
    string Family,
    int Level,
    string Problem,
    ulong Seed,
    int GenerationsRun,
    long Evaluations,
    double TrainingMse,
    double ValidationMse,
    double ValidationNmse,
    double ElapsedSeconds,
    DagReuseMetrics Best,
    DagReuseAverages InitialPopulationMean,
    DagReuseAverages FinalPopulationMean);

/// <summary>
/// <see cref="IProgress{T}"/> síncrono: <see cref="Progress{T}"/> enfileira os callbacks no
/// ThreadPool e, com o pool ocupado pelas execuções, o progresso nunca aparecia.
/// </summary>
public sealed class SynchronousProgress<T>(Action<T> handler) : IProgress<T>
{
    private readonly Lock gate = new();

    public void Report(T value)
    {
        lock (this.gate)
        {
            handler(value);
        }
    }
}

/// <summary>
/// Executa a grade (configuração x família x nível x seed) e grava relatórios tidy em CSV
/// prontos para o notebook poliglota. Nada aqui altera o núcleo evolutivo.
/// </summary>
public static class RegressionGridRunner
{
    /// <summary>Fluxo RNG exclusivo da amostragem aleatória do CLI, fora dos fluxos 1 e 2 do engine.</summary>
    private const ulong RandomSearchStreamId = 101;

    public static RegressionGridReport Run(int seedCount, int stride, int scale = 1, IProgress<string>? progress = null)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(seedCount, 1);
        ArgumentOutOfRangeException.ThrowIfLessThan(stride, 1);
        ArgumentOutOfRangeException.ThrowIfLessThan(scale, 1);

        var jobs =
            from configuration in RegressionGridConfigurations.All
            from problem in RegressionGridCatalog.All
            from seed in Enumerable.Range(0, seedCount)
            select (configuration, problem, seed: (ulong)seed);

        var observations = new ConcurrentBag<RegressionGridObservation>();
        var runs = new ConcurrentBag<RegressionGridRun>();
        var completed = 0;
        var jobList = jobs.ToArray();

        // Cada item é longo (uma evolução inteira) e o engine ainda usa Parallel.For interno.
        // Sem o teto explícito o ThreadPool interpreta a saturação como starvation e injeta
        // centenas de threads, derrubando o throughput em máquinas de poucos núcleos.
        var parallelOptions = new ParallelOptions { MaxDegreeOfParallelism = Environment.ProcessorCount };

        Parallel.ForEach(jobList, parallelOptions, job =>
        {
            var run = Execute(job.configuration, job.problem, job.seed, stride, scale, observations);
            runs.Add(run);
            var done = Interlocked.Increment(ref completed);
            progress?.Report(
                $"[{done}/{jobList.Length}] {run.Experiment} {run.Problem} seed={run.Seed} " +
                $"NMSE={run.ValidationNmse.ToString("E3", CultureInfo.InvariantCulture)}");
        });

        return new RegressionGridReport(
            [.. observations.OrderBy(o => o.Experiment, StringComparer.Ordinal)
                .ThenBy(o => o.Problem, StringComparer.Ordinal)
                .ThenBy(o => o.Seed)
                .ThenBy(o => o.Generation)],
            [.. runs.OrderBy(r => r.Experiment, StringComparer.Ordinal)
                .ThenBy(r => r.Problem, StringComparer.Ordinal)
                .ThenBy(r => r.Seed)]);
    }

    private static RegressionGridRun Execute(
        RegressionGridConfiguration configuration,
        RegressionGridProblem gridProblem,
        ulong seed,
        int stride,
        int scale,
        ConcurrentBag<RegressionGridObservation> observations)
    {
        var definition = gridProblem.Definition;
        var problem = definition.CreateGeneticProblem();
        var metricEvaluator = new ReferenceGenomeEvaluator<double>(problem.Functions);

        // WorkerCount=1 mantém o resultado bit a bit (a equivalência entre workers é
        // garantida pelo engine) e deixa o paralelismo no nível das execuções.
        // scale divide o orçamento preservando a razão entre as configurações; é usado
        // apenas por smoke tests, nunca pela grade publicada (scale = 1).
        var options = configuration.Options with
        {
            Seed = seed,
            WorkerCount = 1,
            PopulationSize = Math.Max(2, configuration.Options.PopulationSize / scale),
            MaxGenerations = Math.Max(1, configuration.Options.MaxGenerations / scale),
        };
        var lastGeneration = options.MaxGenerations;

        DagReuseAverages? initialPopulationMean = null;
        var finalPopulationMean = default(DagReuseAverages);
        var stopwatch = System.Diagnostics.Stopwatch.StartNew();

        void Observe(int generation, Genome<double>[] population, Genome<double> best, double bestTrainingMse)
        {
            if (generation != 0 && generation != lastGeneration && generation % stride != 0)
            {
                return;
            }

            var populationMean = DagReuseAverages.From(
                [.. population.Select(g => DagReuseMetrics.Measure(g, problem.Functions))]);
            initialPopulationMean ??= populationMean;
            finalPopulationMean = populationMean;

            var validationMse = MeanSquaredError(best, metricEvaluator, definition.Dataset.ValidationInputs, definition.Target);
            observations.Add(new RegressionGridObservation(
                configuration.Id,
                configuration.Kind,
                gridProblem.Family,
                gridProblem.Level,
                gridProblem.Id,
                seed,
                generation,
                (long)(generation + 1) * options.PopulationSize,
                bestTrainingMse,
                validationMse,
                NormalizedMeanSquaredError(validationMse, definition.Dataset.ValidationInputs, definition.Target),
                DagReuseMetrics.Measure(best, problem.Functions),
                populationMean));
        }

        Genome<double> bestGenome;
        double bestTrainingMseFinal;
        int generationsRun;

        if (configuration.Strategy == RegressionGridStrategy.RandomSearch)
        {
            (bestGenome, bestTrainingMseFinal, generationsRun) = RunRandomSearch(problem, options, Observe);
        }
        else
        {
            var engine = new EvolutionEngine<double>(checkpoint: state =>
                Observe(state.Generation, state.Population, state.BestGenome!, -state.BestFitness.Value));
            var result = engine.Run(problem, options);
            bestGenome = result.BestGenome;
            bestTrainingMseFinal = -result.BestFitness.Value;
            generationsRun = result.GenerationsRun;
        }

        stopwatch.Stop();
        var validationFinal = MeanSquaredError(bestGenome, metricEvaluator, definition.Dataset.ValidationInputs, definition.Target);
        return new RegressionGridRun(
            configuration.Id,
            configuration.Kind,
            gridProblem.Family,
            gridProblem.Level,
            gridProblem.Id,
            seed,
            generationsRun,
            (long)(generationsRun + 1) * options.PopulationSize,
            bestTrainingMseFinal,
            validationFinal,
            NormalizedMeanSquaredError(validationFinal, definition.Dataset.ValidationInputs, definition.Target),
            stopwatch.Elapsed.TotalSeconds,
            DagReuseMetrics.Measure(bestGenome, problem.Functions),
            initialPopulationMean ?? default,
            finalPopulationMean);
    }

    private static (Genome<double> Best, double TrainingMse, int Generations) RunRandomSearch(
        IGeneticProblem<double> problem,
        EvolutionOptions options,
        Action<int, Genome<double>[], Genome<double>, double> observe)
    {
        var operators = new EvolutionOperators<double>(problem);
        var fitnessEvaluator = problem.CreateFitnessEvaluator();
        var genomeEvaluator = new ReferenceGenomeEvaluator<double>(problem.Functions);
        var population = new Genome<double>[options.PopulationSize];

        Genome<double>? best = null;
        var bestFitness = FitnessResult.Invalid();

        for (var generation = 0; generation <= options.MaxGenerations; generation++)
        {
            for (var index = 0; index < population.Length; index++)
            {
                var rng = RandomStream.Create(options.Seed, (ulong)generation, (ulong)index, RandomSearchStreamId);
                var candidate = operators.CreateInitial(ref rng);
                population[index] = candidate;
                var fitness = fitnessEvaluator.Evaluate(candidate, genomeEvaluator, CancellationToken.None);
                if (best is null || EvolutionOperators<double>.CompareFitness(fitness, bestFitness) > 0)
                {
                    best = candidate;
                    bestFitness = fitness;
                }
            }

            observe(generation, population, best!, -bestFitness.Value);
        }

        return (best!, -bestFitness.Value, options.MaxGenerations);
    }

    internal static double MeanSquaredError(
        Genome<double> genome,
        IGenomeEvaluator<double> evaluator,
        double[] inputs,
        Func<double, double> target)
    {
        var input = new double[1];
        var output = new double[1];
        var sum = 0.0;
        foreach (var value in inputs)
        {
            input[0] = value;
            evaluator.Evaluate(genome, input, output);
            var difference = output[0] - target(value);
            sum += difference * difference;
        }

        return sum / inputs.Length;
    }

    internal static double NormalizedMeanSquaredError(
        double mse,
        double[] inputs,
        Func<double, double> target)
    {
        var mean = inputs.Average(target);
        var variance = inputs.Average(value =>
        {
            var difference = target(value) - mean;
            return difference * difference;
        });
        return variance == 0.0 ? (mse == 0.0 ? 0.0 : double.PositiveInfinity) : mse / variance;
    }
}

/// <summary>Resultado completo da grade.</summary>
public sealed record RegressionGridReport(
    IReadOnlyList<RegressionGridObservation> Observations,
    IReadOnlyList<RegressionGridRun> Runs)
{
    private const string ObservationHeader =
        "experiment,kind,family,level,problem,seed,generation,evaluations," +
        "trainingMse,validationMse,validationNmse,accuracyDigits," +
        "bestActiveNodes,bestReusedActiveNodes,bestReuseEdges,bestSemanticReferences," +
        "bestReusedActiveNodeRatio,bestReuseEdgeRatio," +
        "populationMeanActiveNodes,populationMeanReusedActiveNodeRatio,populationMeanReuseEdgeRatio\n";

    private const string RunHeader =
        "experiment,kind,family,level,problem,seed,generationsRun,evaluations," +
        "trainingMse,validationMse,validationNmse,accuracyDigits,elapsedSeconds," +
        "bestActiveNodes,bestReusedActiveNodeRatio,bestReuseEdgeRatio," +
        "initialPopulationReuseEdgeRatio,finalPopulationReuseEdgeRatio,populationReuseEdgeRatioDelta," +
        "initialPopulationReusedActiveNodeRatio,finalPopulationReusedActiveNodeRatio," +
        "populationReusedActiveNodeRatioDelta\n";

    /// <summary>
    /// Dígitos de precisão: <c>-log10(NMSE)</c>. É a medida contínua que substitui o
    /// limiar discreto de sucesso; 3 dígitos equivalem exatamente a <c>NMSE &lt; 1e-3</c>.
    /// </summary>
    public static double AccuracyDigits(double nmse)
        => nmse <= 0.0 ? 16.0 : Math.Clamp(-Math.Log10(nmse), -16.0, 16.0);

    public void Write(string outDir)
    {
        Directory.CreateDirectory(outDir);
        File.WriteAllText(Path.Combine(outDir, "grid-generations.csv"), BuildObservationCsv());
        File.WriteAllText(Path.Combine(outDir, "grid-runs.csv"), BuildRunCsv());
        File.WriteAllText(Path.Combine(outDir, "grid-configurations.csv"), BuildConfigurationCsv());
        File.WriteAllText(Path.Combine(outDir, "grid-problems.csv"), BuildProblemCsv());
    }

    private string BuildObservationCsv()
    {
        var csv = new StringBuilder(ObservationHeader);
        foreach (var o in Observations)
        {
            csv.Append(CultureInfo.InvariantCulture,
                $"{o.Experiment},{o.Kind},{o.Family},{o.Level},{o.Problem},{o.Seed},{o.Generation},{o.Evaluations}," +
                $"{o.TrainingMse:R},{o.ValidationMse:R},{o.ValidationNmse:R},{AccuracyDigits(o.ValidationNmse):R}," +
                $"{o.Best.ActiveNodes},{o.Best.ReusedActiveNodes},{o.Best.ReuseEdges},{o.Best.SemanticReferences}," +
                $"{o.Best.ReusedActiveNodeRatio:R},{o.Best.ReuseEdgeRatio:R}," +
                $"{o.PopulationMean.ActiveNodes:R},{o.PopulationMean.ReusedActiveNodeRatio:R}," +
                $"{o.PopulationMean.ReuseEdgeRatio:R}\n");
        }

        return csv.ToString();
    }

    private string BuildRunCsv()
    {
        var csv = new StringBuilder(RunHeader);
        foreach (var r in Runs)
        {
            csv.Append(CultureInfo.InvariantCulture,
                $"{r.Experiment},{r.Kind},{r.Family},{r.Level},{r.Problem},{r.Seed},{r.GenerationsRun},{r.Evaluations}," +
                $"{r.TrainingMse:R},{r.ValidationMse:R},{r.ValidationNmse:R},{AccuracyDigits(r.ValidationNmse):R}," +
                $"{r.ElapsedSeconds:F3},{r.Best.ActiveNodes},{r.Best.ReusedActiveNodeRatio:R},{r.Best.ReuseEdgeRatio:R}," +
                $"{r.InitialPopulationMean.ReuseEdgeRatio:R},{r.FinalPopulationMean.ReuseEdgeRatio:R}," +
                $"{r.FinalPopulationMean.ReuseEdgeRatio - r.InitialPopulationMean.ReuseEdgeRatio:R}," +
                $"{r.InitialPopulationMean.ReusedActiveNodeRatio:R},{r.FinalPopulationMean.ReusedActiveNodeRatio:R}," +
                $"{r.FinalPopulationMean.ReusedActiveNodeRatio - r.InitialPopulationMean.ReusedActiveNodeRatio:R}\n");
        }

        return csv.ToString();
    }

    private static string BuildConfigurationCsv()
    {
        var csv = new StringBuilder("experiment,kind,strategy,populationSize,maxGenerations,tournamentSize,crossoverRate,eliteCount,description\n");
        foreach (var c in RegressionGridConfigurations.All)
        {
            csv.Append(CultureInfo.InvariantCulture,
                $"{c.Id},{c.Kind},{c.Strategy},{c.Options.PopulationSize},{c.Options.MaxGenerations}," +
                $"{c.Options.TournamentSize},{c.Options.CrossoverRate:R},{c.Options.EliteCount},\"{c.Description}\"\n");
        }

        return csv.ToString();
    }

    private static string BuildProblemCsv()
    {
        var csv = new StringBuilder("problem,family,level,expression,inputMinimum,inputMaximum,trainingPoints,validationPoints\n");
        foreach (var p in RegressionGridCatalog.All)
        {
            csv.Append(CultureInfo.InvariantCulture,
                $"{p.Id},{p.Family},{p.Level},\"{p.Definition.Expression}\"," +
                $"{p.Definition.InputMinimum:R},{p.Definition.InputMaximum:R}," +
                $"{p.Definition.Dataset.TrainingInputs.Length},{p.Definition.Dataset.ValidationInputs.Length}\n");
        }

        return csv.ToString();
    }
}
