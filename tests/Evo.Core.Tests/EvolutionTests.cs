using System.Collections.Concurrent;
using System.Threading;
using Evo.Core.Functions;
using Evo.Core.Values;
using Xunit;

namespace Evo.Core.Tests;

internal sealed class EvolutionTestProblem<T> : IGeneticProblem<T>
{
    private readonly Func<IFitnessEvaluator<T>> _factory;

    public EvolutionTestProblem(
        GenomeShape shape,
        IFunctionSet<T> functions,
        IValueSemantics<T> values,
        IParameterGeneStrategy<T> parameters,
        Func<IFitnessEvaluator<T>> factory)
    {
        Shape = shape;
        Functions = functions;
        Values = values;
        Parameters = parameters;
        _factory = factory;
    }

    public string ProblemId => "evolution-test-problem";

    public GenomeShape Shape { get; }

    public IFunctionSet<T> Functions { get; }

    public IValueSemantics<T> Values { get; }

    public IParameterGeneStrategy<T> Parameters { get; }

    public IFitnessEvaluator<T> CreateFitnessEvaluator() => _factory();
}

/// <summary>Fitness = -MSE de f(x) contra x², com inválido quando a saída não é finita.</summary>
internal sealed class QuadraticFitness : IFitnessEvaluator<double>
{
    private static readonly double[] Inputs = { -2, -1, -0.5, 0, 0.5, 1, 2 };
    private readonly Action? _onEvaluate;
    private readonly Func<double, double> _target;

    public QuadraticFitness(Action? onEvaluate = null, Func<double, double>? target = null)
    {
        _onEvaluate = onEvaluate;
        _target = target ?? (x => x * x);
    }

    public FitnessResult Evaluate(Genome<double> genome, IGenomeEvaluator<double> evaluator, CancellationToken cancellationToken)
    {
        _onEvaluate?.Invoke();
        var output = new double[1];
        var error = 0.0;
        foreach (var x in Inputs)
        {
            evaluator.Evaluate(genome, new[] { x }, output);
            var d = output[0] - _target(x);
            if (double.IsNaN(d) || double.IsInfinity(d))
            {
                return FitnessResult.Invalid();
            }

            error += d * d;
        }

        return FitnessResult.Valid(-error / Inputs.Length);
    }
}

/// <summary>Conta avaliações por fingerprint fenotípica; delega o fitness a <see cref="QuadraticFitness"/>.</summary>
internal sealed class CountingFitness : IFitnessEvaluator<double>
{
    private readonly QuadraticFitness _inner = new();
    private readonly ConcurrentDictionary<string, int> _counts;

    public CountingFitness(ConcurrentDictionary<string, int> counts)
    {
        _counts = counts;
    }

    public FitnessResult Evaluate(Genome<double> genome, IGenomeEvaluator<double> evaluator, CancellationToken cancellationToken)
    {
        var key = Convert.ToHexString(genome.ComputePhenotypeFingerprint(StandardDoubleFunctionSet.Instance, DoubleValueSemantics.Instance));
        _counts.AddOrUpdate(key, 1, static (_, current) => current + 1);

        return _inner.Evaluate(genome, evaluator, cancellationToken);
    }
}

public sealed class EvolutionOperatorTests
{
    private static readonly GenomeShape Shape = new(1, 8, 1, 2, 3);

    private static EvolutionTestProblem<double> DoubleProblem() => new(
        Shape,
        StandardDoubleFunctionSet.Instance,
        DoubleValueSemantics.Instance,
        DoubleParameterStrategy.Instance,
        () => new QuadraticFitness());

    private static Genome<double>[] InitialPopulation(EvolutionOperators<double> operators, int count, ulong seed)
    {
        var population = new Genome<double>[count];
        for (var i = 0; i < count; i++)
        {
            var rng = RandomStream.Create(seed, 0, (ulong)i, 1);
            population[i] = operators.CreateInitial(ref rng);
        }

        return population;
    }

    private static EvolutionOptions Options(double crossoverRate) => new()
    {
        TournamentSize = 1,
        CrossoverRate = crossoverRate,
        UniformCrossoverGeneProbability = 0.5,
        Seed = 42,
    };

    /// <summary>Genoma com todas as funções iguais a <paramref name="functionId"/> (Add=0, Subtract=1).</summary>
    private static Genome<double> UniformFunctionGenome(EvolutionOperators<double> operators, int functionId)
    {
        var rng = RandomStream.Create(9, 0, 0, 1);
        var genome = operators.CreateInitial(ref rng);
        for (var node = 0; node < Shape.NodeCount; node++)
        {
            genome.SetFunctionGene(node, functionId);
        }

        return genome;
    }

    private static int CountFunctionsOf(Genome<double> genome, int functionId)
    {
        var count = 0;
        foreach (var gene in genome.FunctionGenes)
        {
            if (gene == functionId)
            {
                count++;
            }
        }

        return count;
    }

    [Fact]
    public void Operator_ten_thousand_children_are_valid()
    {
        var problem = DoubleProblem();
        var operators = new EvolutionOperators<double>(problem);
        var population = InitialPopulation(operators, 50, 7);
        var fitness = Enumerable.Range(0, 50).Select(i => FitnessResult.Valid(i)).ToArray();
        var options = new EvolutionOptions { TournamentSize = 5, Seed = 42 };

        for (var k = 0; k < 10_000; k++)
        {
            var rng = RandomStream.Create(42, 1, (ulong)k, 2);
            var child = operators.Breed(population, fitness, options, ref rng);
            child.Validate(problem.Functions);
            Assert.Equal(Shape, child.Shape);
            _ = child.ComputePhenotypeFingerprint(problem.Functions, problem.Values);
        }
    }

    [Fact]
    public void Operator_crossover_mixes_parents()
    {
        var problem = DoubleProblem();
        var operators = new EvolutionOperators<double>(problem);
        var population = new[] { UniformFunctionGenome(operators, 0), UniformFunctionGenome(operators, 1) };
        var fitness = new[] { FitnessResult.Valid(1), FitnessResult.Valid(1) };

        var mixed = 0;
        const int children = 2000;
        for (var k = 0; k < children; k++)
        {
            var rng = RandomStream.Create(42, 1, (ulong)k, 2);
            var child = operators.Breed(population, fitness, Options(1.0), ref rng);
            if (CountFunctionsOf(child, 0) > 0 && CountFunctionsOf(child, 1) > 0)
            {
                mixed++;
            }
        }

        Assert.True(mixed > children * 0.3, $"Crossover produced only {mixed} mixed children of {children}.");
    }

    [Fact]
    public void Operator_rate_zero_is_mutation_only()
    {
        var problem = DoubleProblem();
        var operators = new EvolutionOperators<double>(problem);
        var parents = new[] { UniformFunctionGenome(operators, 0), UniformFunctionGenome(operators, 1) };
        var fitness = new[] { FitnessResult.Valid(1), FitnessResult.Valid(1) };

        var mixed = 0;
        const int children = 2000;
        for (var k = 0; k < children; k++)
        {
            var rng = RandomStream.Create(42, 1, (ulong)k, 2);
            var child = operators.Breed(parents, fitness, Options(0.0), ref rng);
            var distanceToNearest = Math.Min(
                Enumerable.Range(0, Shape.NodeCount).Count(n => child.FunctionGenes[n] != 0),
                Enumerable.Range(0, Shape.NodeCount).Count(n => child.FunctionGenes[n] != 1));
            if (distanceToNearest > 3)
            {
                mixed++;
            }
        }

        Assert.True(mixed < children * 0.02, $"Rate 0 produced {mixed} non-mutation-only children of {children}.");
    }

    [Fact]
    public void Operator_neutral_mutation_preserves_fingerprint()
    {
        var problem = DoubleProblem();
        var operators = new EvolutionOperators<double>(problem);
        var parent = InitialPopulation(operators, 1, 11)[0];
        parent.SetOutputGene(0, 1);
        var parentFingerprint = parent.ComputePhenotypeFingerprint(problem.Functions, problem.Values);
        var fitness = new[] { FitnessResult.Valid(1) };

        var neutral = 0;
        const int children = 2000;
        for (var k = 0; k < children; k++)
        {
            var rng = RandomStream.Create(42, 1, (ulong)k, 2);
            var child = operators.Breed(new[] { parent }, fitness, Options(0.0), ref rng);
            if (child.ComputePhenotypeFingerprint(problem.Functions, problem.Values).SequenceEqual(parentFingerprint))
            {
                neutral++;
            }
        }

        Assert.True(neutral > 0, "No neutral mutation was observed with inactive nodes present.");
    }

    [Fact]
    public void Operator_same_seed_reproduces_population_and_offspring()
    {
        var problem = DoubleProblem();
        var operators = new EvolutionOperators<double>(problem);
        var first = InitialPopulation(operators, 20, 7);
        var second = InitialPopulation(operators, 20, 7);
        var other = InitialPopulation(operators, 20, 8);

        for (var i = 0; i < first.Length; i++)
        {
            Assert.Equal(first[i].FunctionGenes.ToArray(), second[i].FunctionGenes.ToArray());
            Assert.Equal(
                first[i].ComputePhenotypeFingerprint(problem.Functions, problem.Values),
                second[i].ComputePhenotypeFingerprint(problem.Functions, problem.Values));
        }

        Assert.Contains(Enumerable.Range(0, first.Length), i =>
            !first[i].FunctionGenes.ToArray().SequenceEqual(other[i].FunctionGenes.ToArray()));

        var fitness = Enumerable.Range(0, 20).Select(i => FitnessResult.Valid(i)).ToArray();
        var options = new EvolutionOptions { TournamentSize = 3, Seed = 42 };
        for (var k = 0; k < 200; k++)
        {
            var rngA = RandomStream.Create(42, 1, (ulong)k, 2);
            var rngB = RandomStream.Create(42, 1, (ulong)k, 2);
            var childA = operators.Breed(first, fitness, options, ref rngA);
            var childB = operators.Breed(second, fitness, options, ref rngB);
            Assert.Equal(childA.FunctionGenes.ToArray(), childB.FunctionGenes.ToArray());
            Assert.Equal(
                childA.ComputePhenotypeFingerprint(problem.Functions, problem.Values),
                childB.ComputePhenotypeFingerprint(problem.Functions, problem.Values));
        }
    }

    [Fact]
    public void Operator_array_values_are_deep_cloned()
    {
        var problem = new EvolutionTestProblem<double[]>(
            Shape,
            StandardDoubleArrayFunctionSet.Instance,
            DoubleArrayValueSemantics.Instance,
            new DoubleArrayParameterStrategy(3),
            () => throw new NotSupportedException());
        var operators = new EvolutionOperators<double[]>(problem);
        var parentRng = RandomStream.Create(13, 0, 0, 1);
        var parent = operators.CreateInitial(ref parentRng);
        var snapshot = new double[Shape.NodeCount][];
        for (var node = 0; node < Shape.NodeCount; node++)
        {
            snapshot[node] = (double[])parent.ParameterGenes[node].Clone();
        }

        var fitness = new[] { FitnessResult.Valid(1) };
        for (var k = 0; k < 1000; k++)
        {
            var rng = RandomStream.Create(42, 1, (ulong)k, 2);
            var child = operators.Breed(new[] { parent }, fitness, Options(0.5), ref rng);
            for (var node = 0; node < Shape.NodeCount; node++)
            {
                Assert.False(ReferenceEquals(child.ParameterGenes[node], parent.ParameterGenes[node]));
            }
        }

        for (var node = 0; node < Shape.NodeCount; node++)
        {
            Assert.Equal(snapshot[node], parent.ParameterGenes[node]);
        }
    }
}

public sealed class EvolutionEngineTests
{
    private static readonly GenomeShape Shape = new(1, 8, 1, 2, 3);

    private static EvolutionTestProblem<double> Problem(Action? onEvaluate = null, Func<double, double>? target = null) => new(
        Shape,
        StandardDoubleFunctionSet.Instance,
        DoubleValueSemantics.Instance,
        DoubleParameterStrategy.Instance,
        () => new QuadraticFitness(onEvaluate, target));

    private static EvolutionOptions Options(int workers) => new()
    {
        PopulationSize = 30,
        MaxGenerations = 15,
        TournamentSize = 3,
        EliteCount = 1,
        Seed = 42,
        WorkerCount = workers,
    };

    [Fact]
    public void Evolution_worker_count_is_equivalent()
    {
        var sequences = new List<List<EvolutionProgress>>();
        var results = new List<EvolutionResult<double>>();
        foreach (var workers in new[] { 1, 2, 4, 8 })
        {
            var reports = new List<EvolutionProgress>();
            var engine = new EvolutionEngine<double>(reports.Add);
            results.Add(engine.Run(Problem(), Options(workers)));
            sequences.Add(reports);
        }

        foreach (var sequence in sequences.Skip(1))
        {
            Assert.Equal(sequences[0], sequence);
        }

        foreach (var result in results)
        {
            Assert.Equal(TerminationReason.MaxGenerations, result.Termination);
            Assert.Equal(15, result.GenerationsRun);
            Assert.Equal(results[0].BestFitness, result.BestFitness);
        }
    }

    [Fact]
    public void Evolution_best_never_worsens()
    {
        var reports = new List<EvolutionProgress>();
        var result = new EvolutionEngine<double>(reports.Add).Run(Problem(), Options(2));

        Assert.Equal(16, reports.Count);
        for (var i = 1; i < reports.Count; i++)
        {
            Assert.True(reports[i].BestFitness.Value >= reports[i - 1].BestFitness.Value);
        }

        Assert.True(result.BestFitness.Value >= reports[0].BestFitness.Value);
    }

    [Fact]
    public void Evolution_cancellation_leaves_consistent_state()
    {
        using var cancellation = new CancellationTokenSource();
        var reports = new List<EvolutionProgress>();
        var engine = new EvolutionEngine<double>(progress =>
        {
            reports.Add(progress);
            if (progress.Generation == 3)
            {
                cancellation.Cancel();
            }
        });

        var result = engine.Run(Problem(), Options(2), cancellation.Token);

        Assert.Equal(TerminationReason.Cancellation, result.Termination);
        Assert.Equal(3, result.GenerationsRun);
        Assert.Equal(reports[^1].BestFitness, result.BestFitness);
    }

    [Fact]
    public void Evolution_error_during_breeding_propagates_without_fallback_and_discards_partial_generation()
    {
        var checkpoints = new List<EvolutionState<double>>();
        var progressReports = new List<EvolutionProgress>();
        var variation = new ThrowingVariationOperator();
        var engine = new EvolutionEngine<double>(variation, progressReports.Add, checkpoints.Add);

        var thrown = Assert.Throws<InvalidOperationException>(() => engine.Run(Problem(), Options(1)));
        Assert.Equal("boom", thrown.Message);

        // Somente a geração 0 (completa) foi publicada; a geração parcial que falhou nunca aparece
        // em progresso nem em checkpoint, e nenhum resultado de fallback é produzido.
        Assert.Equal(new[] { 0 }, progressReports.Select(p => p.Generation));
        Assert.Equal(new[] { 0 }, checkpoints.Select(c => c.Generation));
    }

    private sealed class ThrowingVariationOperator : IVariationOperator<double>
    {
        public Genome<double> CreateOffspring(VariationContext<double> context, ref RandomStream random) =>
            throw new InvalidOperationException("boom");
    }

    [Fact]
    public void Evolution_target_fitness_stops_at_first_generation()
    {
        var options = Options(1) with { TargetFitness = double.NegativeInfinity };
        var result = new EvolutionEngine<double>().Run(Problem(), options);

        Assert.Equal(TerminationReason.TargetFitness, result.Termination);
        Assert.Equal(0, result.GenerationsRun);
    }

    [Fact]
    public void Evolution_fake_problem_improves_best()
    {
        var reports = new List<EvolutionProgress>();
        var options = Options(2) with { PopulationSize = 60, MaxGenerations = 300 };
        var result = new EvolutionEngine<double>(reports.Add).Run(Problem(target: x => x * x * x), options);

        Assert.True(result.BestFitness.Value > reports[0].BestFitness.Value,
            $"best gen0={reports[0].BestFitness.Value} final={result.BestFitness.Value} gens={result.GenerationsRun}");
    }

    [Fact]
    public void Evolution_cache_reuses_repeated_individuals()
    {
        var evaluations = 0;
        var options = Options(2);
        new EvolutionEngine<double>().Run(Problem(() => Interlocked.Increment(ref evaluations)), options);

        // Elites keep the same phenotype across generations, so their fitness must come from the cache.
        Assert.True(evaluations < options.PopulationSize * (options.MaxGenerations + 1));
    }

    private static EvolutionTestProblem<double> CountingProblem(ConcurrentDictionary<string, int> counts) => new(
        Shape,
        StandardDoubleFunctionSet.Instance,
        DoubleValueSemantics.Instance,
        DoubleParameterStrategy.Instance,
        () => new CountingFitness(counts));

    private static void AssertWorkerEquivalence(EvolutionOptions baseOptions, TerminationReason expected)
    {
        var sequences = new List<List<EvolutionProgress>>();
        var results = new List<EvolutionResult<double>>();
        foreach (var workers in new[] { 1, 2, 4, 8 })
        {
            var reports = new List<EvolutionProgress>();
            results.Add(new EvolutionEngine<double>(reports.Add).Run(Problem(), baseOptions with { WorkerCount = workers }));
            sequences.Add(reports);
        }

        Assert.Equal(expected, results[0].Termination);
        for (var i = 1; i < results.Count; i++)
        {
            Assert.Equal(sequences[0], sequences[i]);
            Assert.Equal(results[0].BestFitness, results[i].BestFitness);
            Assert.Equal(results[0].GenerationsRun, results[i].GenerationsRun);
            Assert.Equal(results[0].Termination, results[i].Termination);
        }
    }

    [Fact]
    public void Evolution_elites_are_copied_unchanged()
    {
        // Com EliteCount == PopulationSize, a geração g+1 é só clone da g: nenhum cruzamento ou mutação.
        var counts0 = new ConcurrentDictionary<string, int>();
        var options0 = Options(1) with { PopulationSize = 10, EliteCount = 10, MaxGenerations = 0 };
        var first = new EvolutionEngine<double>().Run(CountingProblem(counts0), options0);

        var counts1 = new ConcurrentDictionary<string, int>();
        var second = new EvolutionEngine<double>().Run(CountingProblem(counts1), options0 with { MaxGenerations = 1 });

        Assert.Equal(counts0.Values.Sum(), counts1.Values.Sum());
        Assert.Equal(first.BestFitness, second.BestFitness);
        Assert.Equal(first.BestGenome.FunctionGenes.ToArray(), second.BestGenome.FunctionGenes.ToArray());
        Assert.Equal(
            first.BestGenome.ComputePhenotypeFingerprint(StandardDoubleFunctionSet.Instance, DoubleValueSemantics.Instance),
            second.BestGenome.ComputePhenotypeFingerprint(StandardDoubleFunctionSet.Instance, DoubleValueSemantics.Instance));
    }

    [Fact]
    public void Evolution_stagnation_is_worker_count_equivalent()
    {
        AssertWorkerEquivalence(Options(1) with { MaxGenerations = 1000, StagnationGenerations = 3 }, TerminationReason.Stagnation);
    }

    [Fact]
    public void Evolution_target_fitness_is_worker_count_equivalent()
    {
        var cubic = (Func<double, double>)(x => x * x * x);
        var baseline = Options(1) with { PopulationSize = 60, MaxGenerations = 300 };
        var target = new EvolutionEngine<double>().Run(Problem(target: cubic), baseline).BestFitness.Value;

        var sequences = new List<List<EvolutionProgress>>();
        var results = new List<EvolutionResult<double>>();
        foreach (var workers in new[] { 1, 2, 4, 8 })
        {
            var reports = new List<EvolutionProgress>();
            var options = baseline with { WorkerCount = workers, TargetFitness = target };
            results.Add(new EvolutionEngine<double>(reports.Add).Run(Problem(target: cubic), options));
            sequences.Add(reports);
        }

        foreach (var sequence in sequences.Skip(1))
        {
            Assert.Equal(sequences[0], sequence);
        }

        foreach (var result in results)
        {
            Assert.Equal(TerminationReason.TargetFitness, result.Termination);
            Assert.True(result.GenerationsRun >= 1);
            Assert.Equal(results[0].GenerationsRun, result.GenerationsRun);
            Assert.Equal(results[0].BestFitness, result.BestFitness);
        }
    }

    [Fact]
    public void Evolution_cache_reuses_only_matching_fingerprint()
    {
        var problem = Problem();
        var operators = new EvolutionOperators<double>(problem);
        var distinct = new HashSet<string>(StringComparer.Ordinal);
        for (var i = 0; i < 30; i++)
        {
            var rng = RandomStream.Create(42, 0, (ulong)i, 1);
            var genome = operators.CreateInitial(ref rng);
            distinct.Add(Convert.ToHexString(genome.ComputePhenotypeFingerprint(problem.Functions, problem.Values)));
        }

        // Geração 0: indivíduos com a mesma fingerprint são avaliados uma única vez; cada fingerprint distinta é avaliada.
        var counts0 = new ConcurrentDictionary<string, int>();
        new EvolutionEngine<double>().Run(CountingProblem(counts0), Options(2) with { MaxGenerations = 0 });
        Assert.Equal(distinct.Count, counts0.Count);
        Assert.All(counts0.Values, c => Assert.Equal(1, c));
        Assert.Equal(distinct.OrderBy(k => k, StringComparer.Ordinal), counts0.Keys.OrderBy(k => k, StringComparer.Ordinal));

        // Execução completa: o cache deve reduzir avaliações e manter todas as fingerprints avaliadas.
        var counts = new ConcurrentDictionary<string, int>();
        var options = Options(2);
        new EvolutionEngine<double>().Run(CountingProblem(counts), options);
        Assert.True(counts.Values.Sum() < options.PopulationSize * (options.MaxGenerations + 1));
        Assert.True(counts.Keys.Count >= distinct.Count);
    }

    [Fact]
    public void Evolution_time_limit_returns_consistent_result()
    {
        var reports = new List<EvolutionProgress>();
        var result = new EvolutionEngine<double>(reports.Add).Run(Problem(), Options(2) with { TimeLimit = TimeSpan.Zero });

        Assert.Equal(TerminationReason.TimeLimit, result.Termination);
        Assert.True(result.BestFitness.IsValid);
        Assert.Equal(reports[^1].BestFitness, result.BestFitness);
        Assert.Equal(reports[^1].Generation, result.GenerationsRun);
    }

    [Fact]
    public void Evolution_api_runs_with_default_options_shape()
    {
        var result = new EvolutionEngine<double>().Run(Problem(), new EvolutionOptions { MaxGenerations = 2, PopulationSize = 10, WorkerCount = 1 });

        Assert.Equal(TerminationReason.MaxGenerations, result.Termination);
        Assert.Equal(2, result.GenerationsRun);
        Assert.True(result.BestFitness.IsValid);
    }
}
