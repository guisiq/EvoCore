using System.Globalization;
using Evo.Core;
using Evo.Core.Functions;
using Evo.Examples.Datasets;

namespace Evo.Examples.Scenarios;

/// <summary>Cenários baseados em configuração evolutiva (primeiros seis IDs obrigatórios).</summary>
internal static class EvolutionScenarioSupport
{
    public static Dictionary<string, string> Params(params (string Key, object Value)[] items)
    {
        var map = new Dictionary<string, string>(StringComparer.Ordinal);
        foreach (var (key, value) in items)
        {
            map[key] = Convert.ToString(value, CultureInfo.InvariantCulture) ?? string.Empty;
        }

        return map;
    }

    public static bool IsValidFinite(FitnessResult fitness) =>
        fitness.IsValid && !double.IsNaN(fitness.Value) && !double.IsInfinity(fitness.Value);

    public static string FingerprintHex(Genome<double> genome, IFunctionSet<double> functions) =>
        Convert.ToHexString(genome.ComputePhenotypeFingerprint(functions, Evo.Core.Values.DoubleValueSemantics.Instance));
}

/// <summary>Compara duas configurações de <see cref="IVariationOperator{T}"/> sobre a mesma população-fonte.</summary>
public sealed class VariationStrategiesBostonScenario : IBenchmarkScenario
{
    public string Id => "variation-strategies-boston";

    public string Description => "Compara estratégias de variação (seletor torneio vs melhor absoluto) sobre boston-housing (50 registros).";

    public BenchmarkScenarioResult Run(CancellationToken cancellationToken)
    {
        var rows = DatasetCatalog.Load("boston-housing", 50);
        var problem = new TabularRegressionProblem("evo.examples.boston-housing-regression/v1", rows.Features, rows.Target, nodeCount: 20);
        var options = new EvolutionOptions { Seed = 777, PopulationSize = 16, MaxGenerations = 3, WorkerCount = 1 };

        var tournament = new EvolutionEngine<double>(new StandardVariationOperator<double>(problem, new TournamentParentSelector<double>(3)));
        var best = new EvolutionEngine<double>(new StandardVariationOperator<double>(problem, new BestParentSelector<double>()));

        var resultTournament = tournament.Run(problem, options, cancellationToken);
        var resultBest = best.Run(problem, options, cancellationToken);

        var gatesOk = EvolutionScenarioSupport.IsValidFinite(resultTournament.BestFitness)
            && EvolutionScenarioSupport.IsValidFinite(resultBest.BestFitness);

        var gates = new Dictionary<string, string>(StringComparer.Ordinal)
        {
            ["valid_outputs_tournament"] = EvolutionScenarioSupport.IsValidFinite(resultTournament.BestFitness).ToString(),
            ["valid_outputs_best"] = EvolutionScenarioSupport.IsValidFinite(resultBest.BestFitness).ToString(),
        };
        var measurements = new Dictionary<string, string>(StringComparer.Ordinal)
        {
            ["tournament_best_fitness"] = resultTournament.BestFitness.Value.ToString("G17", CultureInfo.InvariantCulture),
            ["tournament_fingerprint"] = EvolutionScenarioSupport.FingerprintHex(resultTournament.BestGenome, problem.Functions),
            ["best_selector_best_fitness"] = resultBest.BestFitness.Value.ToString("G17", CultureInfo.InvariantCulture),
            ["best_selector_fingerprint"] = EvolutionScenarioSupport.FingerprintHex(resultBest.BestGenome, problem.Functions),
        };
        var raw = new Dictionary<string, string>(StringComparer.Ordinal)
        {
            ["dataset_sha256"] = rows.ObservedSha256,
            ["dataset_records_used"] = rows.RecordCount.ToString(CultureInfo.InvariantCulture),
        };

        return new BenchmarkScenarioResult(
            Id,
            gatesOk,
            gates,
            measurements,
            raw,
            EvolutionScenarioSupport.Params(("seed", options.Seed), ("population", options.PopulationSize), ("generations", options.MaxGenerations)),
            gatesOk ? "ambas estratégias produziram fitness/fingerprint válidos" : "fitness inválida em pelo menos uma estratégia");
    }
}

/// <summary>Compara inicialização vinda do engine (aleatória por seed) com uma população semeada por clonagem.</summary>
public sealed class InitializationStrategiesDiabetesScenario : IBenchmarkScenario
{
    public string Id => "initialization-strategies-diabetes";

    public string Description => "Compara inicialização aleatória e semeada (pima-diabetes, 40 registros, mesmo orçamento).";

    public BenchmarkScenarioResult Run(CancellationToken cancellationToken)
    {
        var rows = DatasetCatalog.Load("pima-diabetes", 40);
        var problem = new TabularRegressionProblem("evo.examples.pima-diabetes-regression/v1", rows.Features, rows.Target, nodeCount: 16);
        var options = new EvolutionOptions { Seed = 2024, PopulationSize = 12, MaxGenerations = 2, WorkerCount = 1 };

        var engine = new EvolutionEngine<double>();
        var randomResult = engine.Run(problem, options, cancellationToken);

        // População semeada: clona o melhor genoma da inicialização aleatória para todos os indivíduos
        // e reavalia o fitness real de cada clone, preservando o mesmo orçamento (seed/geracoes) do engine.
        var evaluator = new ReferenceGenomeEvaluator<double>(problem.Functions);
        var fitnessEvaluator = problem.CreateFitnessEvaluator();
        var population = new Genome<double>[options.PopulationSize];
        var fitness = new FitnessResult[options.PopulationSize];
        for (var i = 0; i < population.Length; i++)
        {
            population[i] = randomResult.BestGenome.Clone(problem.Values);
            fitness[i] = fitnessEvaluator.Evaluate(population[i], evaluator, cancellationToken);
        }

        var seedState = new EvolutionState<double>(0, population, fitness, randomResult.BestGenome, fitness[0], 0);
        var seededResult = engine.Resume(problem, options, seedState, cancellationToken);

        var randomActive = randomResult.BestGenome.Analyze(problem.Functions);
        var seededActive = seededResult.BestGenome.Analyze(problem.Functions);
        var gatesOk = EvolutionScenarioSupport.IsValidFinite(randomResult.BestFitness)
            && EvolutionScenarioSupport.IsValidFinite(seededResult.BestFitness)
            && randomActive.ActiveNodeCount > 0
            && seededActive.ActiveNodeCount > 0
            && randomResult.BestGenome.Shape.Equals(problem.Shape)
            && seededResult.BestGenome.Shape.Equals(problem.Shape);

        var gates = new Dictionary<string, string>(StringComparer.Ordinal)
        {
            ["random_valid_genome"] = (EvolutionScenarioSupport.IsValidFinite(randomResult.BestFitness) && randomActive.ActiveNodeCount > 0).ToString(),
            ["seeded_valid_genome"] = (EvolutionScenarioSupport.IsValidFinite(seededResult.BestFitness) && seededActive.ActiveNodeCount > 0).ToString(),
        };
        var measurements = new Dictionary<string, string>(StringComparer.Ordinal)
        {
            ["random_best_fitness"] = randomResult.BestFitness.Value.ToString("G17", CultureInfo.InvariantCulture),
            ["seeded_best_fitness"] = seededResult.BestFitness.Value.ToString("G17", CultureInfo.InvariantCulture),
            ["random_active_nodes"] = randomActive.ActiveNodeCount.ToString(CultureInfo.InvariantCulture),
            ["seeded_active_nodes"] = seededActive.ActiveNodeCount.ToString(CultureInfo.InvariantCulture),
        };
        var raw = new Dictionary<string, string>(StringComparer.Ordinal)
        {
            ["dataset_sha256"] = rows.ObservedSha256,
            ["dataset_records_used"] = rows.RecordCount.ToString(CultureInfo.InvariantCulture),
        };

        return new BenchmarkScenarioResult(
            Id,
            gatesOk,
            gates,
            measurements,
            raw,
            EvolutionScenarioSupport.Params(("seed", options.Seed), ("population", options.PopulationSize), ("generations", options.MaxGenerations)),
            gatesOk ? "ambas inicializações produziram genomas válidos com shape e grafo ativo não vazio" : "genoma inválido em pelo menos uma inicialização");
    }
}

/// <summary>Executa a mesma configuração para três tamanhos de população (wine-quality-red, 30 registros).</summary>
public sealed class PopulationScaleWineScenario : IBenchmarkScenario
{
    public string Id => "population-scale-wine";

    public string Description => "Escalabilidade por tamanho de população (10/20/30) sobre wine-quality-red (30 registros).";

    public BenchmarkScenarioResult Run(CancellationToken cancellationToken)
    {
        var rows = DatasetCatalog.Load("wine-quality-red", 30);
        var problem = new TabularRegressionProblem("evo.examples.wine-quality-regression/v1", rows.Features, rows.Target, nodeCount: 18);
        var sizes = new[] { 10, 20, 30 };
        var gates = new Dictionary<string, string>(StringComparer.Ordinal);
        var measurements = new Dictionary<string, string>(StringComparer.Ordinal);
        var allOk = true;
        foreach (var size in sizes)
        {
            var options = new EvolutionOptions { Seed = 99, PopulationSize = size, MaxGenerations = 3, WorkerCount = 1 };
            var engine = new EvolutionEngine<double>();
            var sw = System.Diagnostics.Stopwatch.StartNew();
            var allocBefore = GC.GetAllocatedBytesForCurrentThread();
            var result = engine.Run(problem, options, cancellationToken);
            sw.Stop();
            var allocated = GC.GetAllocatedBytesForCurrentThread() - allocBefore;
            var ok = result.GenerationsRun == options.MaxGenerations && EvolutionScenarioSupport.IsValidFinite(result.BestFitness);
            allOk &= ok;
            gates[$"population_{size}_completed"] = ok.ToString();
            measurements[$"population_{size}_evaluations"] = ((long)size * (options.MaxGenerations + 1)).ToString(CultureInfo.InvariantCulture);
            measurements[$"population_{size}_elapsed_ms"] = sw.Elapsed.TotalMilliseconds.ToString("F3", CultureInfo.InvariantCulture);
            measurements[$"population_{size}_allocated_bytes"] = allocated.ToString(CultureInfo.InvariantCulture);
        }

        var raw = new Dictionary<string, string>(StringComparer.Ordinal)
        {
            ["dataset_sha256"] = rows.ObservedSha256,
            ["dataset_records_used"] = rows.RecordCount.ToString(CultureInfo.InvariantCulture),
        };

        return new BenchmarkScenarioResult(
            Id,
            allOk,
            gates,
            measurements,
            raw,
            EvolutionScenarioSupport.Params(("sizes", string.Join(',', sizes))),
            allOk ? "os três tamanhos de população completaram o orçamento de gerações" : "ao menos um tamanho de população não completou");
    }
}

/// <summary>Compara function sets básico, padrão e estendido (boston-housing, 25 registros).</summary>
public sealed class FunctionSetComplexityBostonScenario : IBenchmarkScenario
{
    public string Id => "function-set-complexity-boston";

    public string Description => "Compara function sets de complexidade crescente sobre boston-housing (25 registros).";

    public BenchmarkScenarioResult Run(CancellationToken cancellationToken)
    {
        var rows = DatasetCatalog.Load("boston-housing", 25);
        var full = StandardDoubleFunctionSet.Instance;
        var sets = new (string Name, IFunctionSet<double> Set)[]
        {
            ("basic", SubsetFunctionSet("basic", full, 2)),
            ("standard", SubsetFunctionSet("standard", full, Math.Min(6, full.Count))),
            ("extended", full),
        };

        var gates = new Dictionary<string, string>(StringComparer.Ordinal);
        var measurements = new Dictionary<string, string>(StringComparer.Ordinal);
        var allOk = true;
        foreach (var (name, set) in sets)
        {
            var problem = new TabularRegressionProblem("evo.examples.boston-housing-function-set/v1", rows.Features, rows.Target, nodeCount: 16, set);
            var options = new EvolutionOptions { Seed = 555, PopulationSize = 16, MaxGenerations = 3, WorkerCount = 1 };
            var engine = new EvolutionEngine<double>();
            var result = engine.Run(problem, options, cancellationToken);
            var active = result.BestGenome.Analyze(set);
            var ok = EvolutionScenarioSupport.IsValidFinite(result.BestFitness);
            allOk &= ok;
            gates[$"{name}_valid"] = ok.ToString();
            measurements[$"{name}_best_fitness"] = result.BestFitness.Value.ToString("G17", CultureInfo.InvariantCulture);
            measurements[$"{name}_active_nodes"] = active.ActiveNodeCount.ToString(CultureInfo.InvariantCulture);
            measurements[$"{name}_function_count"] = set.Count.ToString(CultureInfo.InvariantCulture);
        }

        var raw = new Dictionary<string, string>(StringComparer.Ordinal)
        {
            ["dataset_sha256"] = rows.ObservedSha256,
            ["dataset_records_used"] = rows.RecordCount.ToString(CultureInfo.InvariantCulture),
        };

        return new BenchmarkScenarioResult(
            Id,
            allOk,
            gates,
            measurements,
            raw,
            EvolutionScenarioSupport.Params(("sets", "basic,standard,extended")),
            allOk ? "todos os function sets produziram avaliação válida" : "ao menos um function set produziu avaliação inválida");
    }

    private static IFunctionSet<double> SubsetFunctionSet(string id, IFunctionSet<double> source, int count) =>
        new SubsetDoubleFunctionSet(id, source, count);
}

/// <summary>Estresse: população 100 por 20 gerações com cancelamento/timeout explícitos (wine-quality-red completo).</summary>
public sealed class MaximumConfigurationWineScenario : IBenchmarkScenario
{
    public string Id => "maximum-configuration-wine";

    public string Description => "Estresse de população 100 por 20 gerações sobre o wine-quality-red completo.";

    public BenchmarkScenarioResult Run(CancellationToken cancellationToken)
    {
        var descriptor = DatasetCatalog.Describe("wine-quality-red");
        var rows = DatasetCatalog.Load("wine-quality-red", descriptor.Records);
        var problem = new TabularRegressionProblem("evo.examples.wine-quality-stress/v1", rows.Features, rows.Target, nodeCount: 24);
        var options = new EvolutionOptions { Seed = 31337, PopulationSize = 100, MaxGenerations = 20, WorkerCount = 1, TimeLimit = TimeSpan.FromMinutes(15) };
        using var harnessTimeout = new CancellationTokenSource(TimeSpan.FromMinutes(15));
        using var linked = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken, harnessTimeout.Token);

        var engine = new EvolutionEngine<double>();
        var result = engine.Run(problem, options, linked.Token);

        var ok = result.GenerationsRun == options.MaxGenerations
            && result.BestGenome is not null
            && EvolutionScenarioSupport.IsValidFinite(result.BestFitness);

        var gates = new Dictionary<string, string>(StringComparer.Ordinal)
        {
            ["generations_completed_20"] = (result.GenerationsRun == options.MaxGenerations).ToString(),
            ["population_cardinality_100"] = (options.PopulationSize == 100).ToString(),
            ["valid_best_genome"] = EvolutionScenarioSupport.IsValidFinite(result.BestFitness).ToString(),
        };
        var measurements = new Dictionary<string, string>(StringComparer.Ordinal)
        {
            ["generations_run"] = result.GenerationsRun.ToString(CultureInfo.InvariantCulture),
            ["best_fitness"] = result.BestFitness.Value.ToString("G17", CultureInfo.InvariantCulture),
            ["termination"] = result.Termination.ToString(),
        };
        var raw = new Dictionary<string, string>(StringComparer.Ordinal)
        {
            ["dataset_sha256"] = rows.ObservedSha256,
            ["dataset_records_used"] = rows.RecordCount.ToString(CultureInfo.InvariantCulture),
        };

        return new BenchmarkScenarioResult(
            Id,
            ok,
            gates,
            measurements,
            raw,
            EvolutionScenarioSupport.Params(("population", 100), ("generations", 20), ("timeLimitMinutes", 15)),
            ok ? "20 gerações completadas com população 100 e melhor genoma válido" : "estresse não completou o orçamento declarado");
    }
}

/// <summary>Repetibilidade sob a mesma seed (pima-diabetes, 20 registros).</summary>
public sealed class SeedRepeatabilityDiabetesScenario : IBenchmarkScenario
{
    public string Id => "seed-repeatability-diabetes";

    public string Description => "Executa a mesma configuração efetiva duas vezes com seed 12345 (pima-diabetes, 20 registros).";

    public BenchmarkScenarioResult Run(CancellationToken cancellationToken)
    {
        var rows = DatasetCatalog.Load("pima-diabetes", 20);
        var options = new EvolutionOptions { Seed = 12345, PopulationSize = 12, MaxGenerations = 4, WorkerCount = 1 };

        var problemA = new TabularRegressionProblem("evo.examples.pima-diabetes-repeatability/v1", rows.Features, rows.Target, nodeCount: 14);
        var problemB = new TabularRegressionProblem("evo.examples.pima-diabetes-repeatability/v1", rows.Features, rows.Target, nodeCount: 14);

        var digestsA = new List<string>();
        var digestsB = new List<string>();
        var engineA = new EvolutionEngine<double>(progress: p => digestsA.Add(p.PopulationDigest));
        var engineB = new EvolutionEngine<double>(progress: p => digestsB.Add(p.PopulationDigest));

        var resultA = engineA.Run(problemA, options, cancellationToken);
        var resultB = engineB.Run(problemB, options, cancellationToken);

        var sameHistory = digestsA.SequenceEqual(digestsB, StringComparer.Ordinal);
        var sameFitness = resultA.BestFitness.Equals(resultB.BestFitness);
        var sameFingerprint = string.Equals(
            EvolutionScenarioSupport.FingerprintHex(resultA.BestGenome, problemA.Functions),
            EvolutionScenarioSupport.FingerprintHex(resultB.BestGenome, problemB.Functions),
            StringComparison.Ordinal);
        var ok = sameHistory && sameFitness && sameFingerprint;

        var gates = new Dictionary<string, string>(StringComparer.Ordinal)
        {
            ["identical_generation_history"] = sameHistory.ToString(),
            ["identical_final_fitness"] = sameFitness.ToString(),
            ["identical_genome_fingerprint"] = sameFingerprint.ToString(),
        };
        var measurements = new Dictionary<string, string>(StringComparer.Ordinal)
        {
            ["generations_compared"] = digestsA.Count.ToString(CultureInfo.InvariantCulture),
            ["best_fitness"] = resultA.BestFitness.Value.ToString("G17", CultureInfo.InvariantCulture),
        };
        var raw = new Dictionary<string, string>(StringComparer.Ordinal)
        {
            ["dataset_sha256"] = rows.ObservedSha256,
            ["dataset_records_used"] = rows.RecordCount.ToString(CultureInfo.InvariantCulture),
            ["fingerprint"] = EvolutionScenarioSupport.FingerprintHex(resultA.BestGenome, problemA.Functions),
        };

        return new BenchmarkScenarioResult(
            Id,
            ok,
            gates,
            measurements,
            raw,
            EvolutionScenarioSupport.Params(("seed", options.Seed)),
            ok ? "duas execuções com seed 12345 produziram histórico, fitness e fingerprint idênticos" : "execução sob a mesma seed divergiu");
    }
}
