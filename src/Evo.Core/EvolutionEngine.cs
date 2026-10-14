using System.Diagnostics;
using System.Security.Cryptography;

namespace Evo.Core;

/// <summary>Relatório emitido ao final de cada geração completa (sem logging obrigatório).</summary>
/// <param name="Generation">Índice da geração avaliada (0 é a população inicial).</param>
/// <param name="BestFitness">Melhor fitness acumulado até a geração.</param>
/// <param name="PopulationDigest">SHA-256 hex das fingerprints da população, em ordem de índice.</param>
public sealed record EvolutionProgress(int Generation, FitnessResult BestFitness, string PopulationDigest);

/// <summary>
/// Estado completo de uma geração avaliada, suficiente para retomar a execução de forma determinística.
/// </summary>
/// <param name="Generation">Geração avaliada.</param>
/// <param name="Population">População da geração (não deve ser mutada após a emissão).</param>
/// <param name="Fitness">Fitness de cada indivíduo, na mesma ordem da população.</param>
/// <param name="BestGenome">Melhor genoma acumulado até a geração.</param>
/// <param name="BestFitness">Melhor fitness acumulado até a geração.</param>
/// <param name="LastImprovement">Última geração em que o best melhorou (histórico de estagnação).</param>
/// <param name="ActiveElapsed">Tempo ativo de evolução acumulado até esta geração (pausas entre processos excluídas).</param>
/// <param name="ElapsedKnown">Falso quando o estado veio de checkpoint legado sem tempo ativo; <paramref name="ActiveElapsed"/> é zero.</param>
public sealed record EvolutionState<T>(
    int Generation,
    Genome<T>[] Population,
    FitnessResult[] Fitness,
    Genome<T>? BestGenome,
    FitnessResult BestFitness,
    int LastImprovement,
    TimeSpan ActiveElapsed = default,
    bool ElapsedKnown = true);

/// <summary>
/// Engine evolutivo determinístico. A avaliação de indivíduos é paralela, mas cada resultado
/// depende apenas do indivíduo, e os RNGs são chaveados por (seed, geração, índice, fluxo).
/// </summary>
public sealed class EvolutionEngine<T> : IEvolutionEngine<T>
{
    private const ulong BreedStreamId = 2;

    private readonly Action<EvolutionProgress>? _progress;
    private readonly Action<EvolutionState<T>>? _checkpoint;
    private readonly Func<TimeSpan>? _clock;

    /// <summary>
    /// Cria o engine. <paramref name="progress"/> recebe um relatório por geração completa;
    /// <paramref name="checkpoint"/> recebe o estado completo de cada geração avaliada.
    /// </summary>
    public EvolutionEngine(Action<EvolutionProgress>? progress = null, Action<EvolutionState<T>>? checkpoint = null)
        : this(progress, checkpoint, null)
    {
    }

    /// <summary>Construtor de teste: <paramref name="clock"/> substitui o relógio de parede do segmento atual.</summary>
    internal EvolutionEngine(Action<EvolutionProgress>? progress, Action<EvolutionState<T>>? checkpoint, Func<TimeSpan>? clock)
    {
        _progress = progress;
        _checkpoint = checkpoint;
        _clock = clock;
    }

    /// <inheritdoc />
    public EvolutionResult<T> Run(
        IGeneticProblem<T> problem,
        EvolutionOptions options,
        CancellationToken cancellationToken = default) =>
        Execute(problem, options, null, cancellationToken);

    /// <summary>
    /// Retoma a execução a partir de um estado salvo. Com as mesmas opções, o resultado é igual ao da
    /// execução contínua; o progresso volta a ser emitido a partir da geração do estado.
    /// </summary>
    public EvolutionResult<T> Resume(
        IGeneticProblem<T> problem,
        EvolutionOptions options,
        EvolutionState<T> state,
        CancellationToken cancellationToken = default) =>
        Execute(problem, options, state ?? throw new ArgumentNullException(nameof(state)), cancellationToken);

    private EvolutionResult<T> Execute(
        IGeneticProblem<T> problem,
        EvolutionOptions options,
        EvolutionState<T>? resume,
        CancellationToken cancellationToken)
    {
        ArgumentNullException.ThrowIfNull(problem);
        ArgumentNullException.ThrowIfNull(options);
        Validate(options);
        cancellationToken.ThrowIfCancellationRequested();

        var values = problem.Values;
        var operators = new EvolutionOperators<T>(problem);
        var stopwatch = Stopwatch.StartNew();
        var baseElapsed = resume?.ActiveElapsed ?? TimeSpan.Zero;
        TimeSpan ActiveElapsed() => baseElapsed + (_clock?.Invoke() ?? stopwatch.Elapsed);
        var n = options.PopulationSize;

        var workers = options.WorkerCount > 0 ? options.WorkerCount : Environment.ProcessorCount;
        var fitnessEvaluators = new IFitnessEvaluator<T>[workers];
        var genomeEvaluators = new IGenomeEvaluator<T>[workers];
        for (var w = 0; w < workers; w++)
        {
            fitnessEvaluators[w] = problem.CreateFitnessEvaluator();
            genomeEvaluators[w] = new ReferenceGenomeEvaluator<T>(problem.Functions);
        }

        var cache = new Dictionary<string, FitnessResult>(StringComparer.Ordinal);
        var cacheLimit = 4 * n;

        var population = new Genome<T>[n];
        if (resume is null)
        {
            for (var i = 0; i < n; i++)
            {
                var rng = RandomStream.Create(options.Seed, 0, (ulong)i, 1);
                population[i] = operators.CreateInitial(ref rng);
            }
        }
        else
        {
            if (resume.Population.Length != n || resume.Fitness.Length != n)
            {
                throw new ArgumentException("O estado retomado não corresponde a PopulationSize.", nameof(resume));
            }

            for (var i = 0; i < n; i++)
            {
                population[i] = resume.Population[i].Clone(values);
            }
        }

        var fitness = new FitnessResult[n];
        var digest = Evaluate(problem, population, fitness, fitnessEvaluators, genomeEvaluators, cache, cacheLimit, cancellationToken);
        if (resume is not null && !fitness.AsSpan().SequenceEqual(resume.Fitness))
        {
            throw new InvalidOperationException("O fitness do estado retomado não coincide com a reavaliação da população.");
        }

        var generation = resume?.Generation ?? 0;
        var lastImprovement = resume?.LastImprovement ?? 0;
        Genome<T>? bestGenome = resume?.BestGenome?.Clone(values);
        var bestFitness = resume?.BestFitness ?? FitnessResult.Invalid();

        while (true)
        {
            if (UpdateBest(population, fitness, ref bestGenome, ref bestFitness))
            {
                lastImprovement = generation;
            }

            _progress?.Invoke(new EvolutionProgress(generation, bestFitness, digest));
            var elapsed = ActiveElapsed();
            _checkpoint?.Invoke(new EvolutionState<T>(generation, population, fitness, bestGenome, bestFitness, lastImprovement, elapsed));

            var reason = CheckTermination(options, generation, lastImprovement, bestFitness, elapsed, cancellationToken);
            if (reason is { } termination)
            {
                return new EvolutionResult<T>(bestGenome!.Clone(values), bestFitness, generation, termination);
            }

            try
            {
                var next = new Genome<T>[n];
                var nextFitness = new FitnessResult[n];
                var elites = SortedIndices(fitness);
                for (var e = 0; e < options.EliteCount; e++)
                {
                    next[e] = population[elites[e]].Clone(values);
                }

                for (var i = options.EliteCount; i < n; i++)
                {
                    var rng = RandomStream.Create(options.Seed, (ulong)(generation + 1), (ulong)i, BreedStreamId);
                    next[i] = operators.Breed(population, fitness, options, ref rng);
                }

                digest = Evaluate(problem, next, nextFitness, fitnessEvaluators, genomeEvaluators, cache, cacheLimit, cancellationToken);
                population = next;
                fitness = nextFitness;
                generation++;
            }
            catch (OperationCanceledException) when (cancellationToken.IsCancellationRequested)
            {
                // A geração parcial é descartada; o estado é o da última geração completa.
                return new EvolutionResult<T>(bestGenome!.Clone(values), bestFitness, generation, TerminationReason.Cancellation);
            }
        }
    }

    private static TerminationReason? CheckTermination(
        EvolutionOptions options,
        int generation,
        int lastImprovement,
        FitnessResult bestFitness,
        TimeSpan elapsed,
        CancellationToken cancellationToken)
    {
        if (options.TargetFitness is { } target && bestFitness.IsValid && bestFitness.Value >= target)
        {
            return TerminationReason.TargetFitness;
        }

        if (options.StagnationGenerations is { } stagnation && generation - lastImprovement >= stagnation)
        {
            return TerminationReason.Stagnation;
        }

        if (generation >= options.MaxGenerations)
        {
            return TerminationReason.MaxGenerations;
        }

        if (cancellationToken.IsCancellationRequested)
        {
            return TerminationReason.Cancellation;
        }

        if (options.TimeLimit is { } limit && elapsed >= limit)
        {
            return TerminationReason.TimeLimit;
        }

        return null;
    }

    private static bool UpdateBest(
        Genome<T>[] population,
        FitnessResult[] fitness,
        ref Genome<T>? bestGenome,
        ref FitnessResult bestFitness)
    {
        var index = 0;
        for (var i = 1; i < fitness.Length; i++)
        {
            if (EvolutionOperators<T>.CompareFitness(fitness[i], fitness[index]) > 0)
            {
                index = i;
            }
        }

        if (bestGenome is null || EvolutionOperators<T>.CompareFitness(fitness[index], bestFitness) > 0)
        {
            bestGenome = population[index];
            bestFitness = fitness[index];
            return true;
        }

        return false;
    }

    /// <summary>Índices ordenados do melhor para o pior; empates favorecem o menor índice.</summary>
    private static int[] SortedIndices(FitnessResult[] fitness)
    {
        var order = Enumerable.Range(0, fitness.Length).ToArray();
        Array.Sort(order, (x, y) =>
        {
            var cmp = EvolutionOperators<T>.CompareFitness(fitness[y], fitness[x]);
            return cmp != 0 ? cmp : x.CompareTo(y);
        });
        return order;
    }

    /// <summary>
    /// Avalia a população. Lookups e atualização do cache ocorrem na thread principal; apenas
    /// indivíduos com chave ausente são avaliados, em paralelo, com um avaliador por worker.
    /// </summary>
    private static string Evaluate(
        IGeneticProblem<T> problem,
        Genome<T>[] population,
        FitnessResult[] fitness,
        IFitnessEvaluator<T>[] fitnessEvaluators,
        IGenomeEvaluator<T>[] genomeEvaluators,
        Dictionary<string, FitnessResult> cache,
        int cacheLimit,
        CancellationToken cancellationToken)
    {
        var n = population.Length;
        var fingerprints = new byte[n][];
        var keys = new string[n];
        var resolved = new bool[n];
        var missing = new List<int>();
        var firstPosition = new Dictionary<string, int>(StringComparer.Ordinal);

        for (var i = 0; i < n; i++)
        {
            fingerprints[i] = population[i].ComputePhenotypeFingerprint(problem.Functions, problem.Values);
            keys[i] = problem.ProblemId + "|" + problem.Functions.Id + "|" + problem.Values.ValueTypeId + "|" + Convert.ToHexString(fingerprints[i]);
            if (cache.TryGetValue(keys[i], out var cached))
            {
                fitness[i] = cached;
                resolved[i] = true;
            }
            else if (!firstPosition.ContainsKey(keys[i]))
            {
                firstPosition[keys[i]] = missing.Count;
                missing.Add(i);
            }
        }

        var results = new FitnessResult[missing.Count];
        var workers = fitnessEvaluators.Length;
        if (fitnessEvaluators[0] is IBatchFitnessEvaluator<T> batchEvaluator)
        {
            // Lotes são fatias contíguas de `missing` (ordem fixa); cada lote é atribuído a um worker
            // por índice. O resultado não depende do tamanho do lote nem do número de workers.
            var batchSize = Math.Max(1, batchEvaluator.PreferredBatchSize);
            var batchCount = (missing.Count + batchSize - 1) / batchSize;
            Parallel.For(0, Math.Min(workers, batchCount), new ParallelOptions { MaxDegreeOfParallelism = workers, CancellationToken = cancellationToken }, w =>
            {
                for (var b = w; b < batchCount; b += workers)
                {
                    var start = b * batchSize;
                    var length = Math.Min(batchSize, missing.Count - start);
                    var genomes = new Genome<T>[length];
                    for (var k = 0; k < length; k++)
                    {
                        genomes[k] = population[missing[start + k]];
                    }

                    var evaluator = (IBatchFitnessEvaluator<T>)fitnessEvaluators[w];
                    evaluator.EvaluateBatch(genomes, genomeEvaluators[w], results.AsSpan(start, length), cancellationToken);
                }
            });
        }
        else
        {
            Parallel.For(0, workers, new ParallelOptions { MaxDegreeOfParallelism = workers, CancellationToken = cancellationToken }, w =>
            {
                for (var p = w; p < missing.Count; p += workers)
                {
                    results[p] = fitnessEvaluators[w].Evaluate(population[missing[p]], genomeEvaluators[w], cancellationToken);
                }
            });
        }

        if (cache.Count + missing.Count > cacheLimit)
        {
            cache.Clear();
        }

        for (var p = 0; p < missing.Count; p++)
        {
            cache[keys[missing[p]]] = results[p];
        }

        for (var i = 0; i < n; i++)
        {
            if (!resolved[i])
            {
                fitness[i] = results[firstPosition[keys[i]]];
            }
        }

        using var hash = IncrementalHash.CreateHash(HashAlgorithmName.SHA256);
        foreach (var fingerprint in fingerprints)
        {
            hash.AppendData(fingerprint);
        }

        return Convert.ToHexString(hash.GetHashAndReset());
    }

    private static void Validate(EvolutionOptions options)
    {
        if (options.PopulationSize < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(options), "PopulationSize must be >= 1.");
        }

        if (options.MaxGenerations < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(options), "MaxGenerations must be >= 0.");
        }

        if (options.TournamentSize < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(options), "TournamentSize must be >= 1.");
        }

        if (options.EliteCount < 0 || options.EliteCount > options.PopulationSize)
        {
            throw new ArgumentOutOfRangeException(nameof(options), "EliteCount must be in [0, PopulationSize].");
        }

        if (options.CrossoverRate < 0 || options.CrossoverRate > 1)
        {
            throw new ArgumentOutOfRangeException(nameof(options), "CrossoverRate must be in [0, 1].");
        }

        if (options.UniformCrossoverGeneProbability < 0 || options.UniformCrossoverGeneProbability > 1)
        {
            throw new ArgumentOutOfRangeException(nameof(options), "UniformCrossoverGeneProbability must be in [0, 1].");
        }

        if (options.WorkerCount < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(options), "WorkerCount must be >= 0.");
        }

        if (options.StagnationGenerations is { } stagnation && stagnation < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(options), "StagnationGenerations must be >= 1.");
        }
    }
}
