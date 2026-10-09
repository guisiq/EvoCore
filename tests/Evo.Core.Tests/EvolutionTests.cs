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

    public QuadraticFitness(Action? onEvaluate = null)
    {
        _onEvaluate = onEvaluate;
    }

    public FitnessResult Evaluate(Genome<double> genome, IGenomeEvaluator<double> evaluator, CancellationToken cancellationToken)
    {
        _onEvaluate?.Invoke();
        var output = new double[1];
        var error = 0.0;
        foreach (var x in Inputs)
        {
            evaluator.Evaluate(genome, new[] { x }, output);
            var d = output[0] - (x * x);
            if (double.IsNaN(d) || double.IsInfinity(d))
            {
                return FitnessResult.Invalid();
            }

            error += d * d;
        }

        return FitnessResult.Valid(-error / Inputs.Length);
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

    private static EvolutionTestProblem<double> Problem(Action? onEvaluate = null) => new(
        Shape,
        StandardDoubleFunctionSet.Instance,
        DoubleValueSemantics.Instance,
        DoubleParameterStrategy.Instance,
        () => new QuadraticFitness(onEvaluate));

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
    public void Evolution_target_fitness_stops_at_first_generation()
    {
        var options = Options(1) with { TargetFitness = double.NegativeInfinity };
        var result = new EvolutionEngine<double>().Run(Problem(), options);

        Assert.Equal(TerminationReason.TargetFitness, result.Termination);
        Assert.Equal(0, result.GenerationsRun);
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

    [Fact]
    public void Evolution_api_runs_with_default_options_shape()
    {
        var result = new EvolutionEngine<double>().Run(Problem(), new EvolutionOptions { MaxGenerations = 2, PopulationSize = 10, WorkerCount = 1 });

        Assert.Equal(TerminationReason.MaxGenerations, result.Termination);
        Assert.Equal(2, result.GenerationsRun);
        Assert.True(result.BestFitness.IsValid);
    }
}
