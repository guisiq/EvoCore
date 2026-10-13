using System.Threading;
using Evo.Core.Functions;
using Evo.Core.Values;
using Xunit;

namespace Evo.Core.Tests;

/// <summary>Contadores compartilhados pelos avaliadores de lote de um mesmo teste.</summary>
internal sealed class BatchCounter
{
    public int Calls;
    public int Genomes;
}

/// <summary>Avaliador de lote que delega cada genoma ao avaliador individual de referência.</summary>
internal sealed class BatchQuadraticFitness : IBatchFitnessEvaluator<double>
{
    private readonly QuadraticFitness _single = new();
    private readonly BatchCounter _counter;

    public BatchQuadraticFitness(int preferredBatchSize, BatchCounter counter)
    {
        PreferredBatchSize = preferredBatchSize;
        _counter = counter;
    }

    public int PreferredBatchSize { get; }

    public FitnessResult Evaluate(Genome<double> genome, IGenomeEvaluator<double> evaluator, CancellationToken cancellationToken) =>
        _single.Evaluate(genome, evaluator, cancellationToken);

    public void EvaluateBatch(
        ReadOnlySpan<Genome<double>> genomes,
        IGenomeEvaluator<double> evaluator,
        Span<FitnessResult> results,
        CancellationToken cancellationToken)
    {
        Interlocked.Increment(ref _counter.Calls);
        Interlocked.Add(ref _counter.Genomes, genomes.Length);
        for (var i = 0; i < genomes.Length; i++)
        {
            results[i] = _single.Evaluate(genomes[i], evaluator, cancellationToken);
        }
    }
}

public sealed class BatchEvaluationTests
{
    private static readonly GenomeShape Shape = new(1, 8, 1, 2, 3);

    private static EvolutionOptions Options(int workers) => new()
    {
        PopulationSize = 30,
        MaxGenerations = 15,
        TournamentSize = 3,
        EliteCount = 1,
        Seed = 42,
        WorkerCount = workers,
    };

    private static EvolutionTestProblem<double> IndividualProblem() => new(
        Shape,
        StandardDoubleFunctionSet.Instance,
        DoubleValueSemantics.Instance,
        DoubleParameterStrategy.Instance,
        () => new QuadraticFitness());

    private static EvolutionTestProblem<double> BatchProblem(int preferredBatchSize, BatchCounter counter) => new(
        Shape,
        StandardDoubleFunctionSet.Instance,
        DoubleValueSemantics.Instance,
        DoubleParameterStrategy.Instance,
        () => new BatchQuadraticFitness(preferredBatchSize, counter));

    [Fact]
    public void Batch_and_individual_evaluation_produce_identical_results()
    {
        var reference = new EvolutionEngine<double>().Run(IndividualProblem(), Options(1));

        var counter = new BatchCounter();
        var batched = new EvolutionEngine<double>().Run(BatchProblem(4, counter), Options(1));

        Assert.True(counter.Calls > 0, "O caminho de lote não foi usado.");
        Assert.Equal(reference.GenerationsRun, batched.GenerationsRun);
        Assert.Equal(reference.Termination, batched.Termination);
        Assert.Equal(reference.BestFitness.Value, batched.BestFitness.Value);
        Assert.Equal(reference.BestFitness.IsValid, batched.BestFitness.IsValid);
    }

    [Theory]
    [InlineData(1, 1)]
    [InlineData(3, 2)]
    [InlineData(7, 4)]
    [InlineData(64, 8)]
    public void Batch_partitioning_and_worker_count_do_not_change_results(int preferredBatchSize, int workers)
    {
        var reference = new EvolutionEngine<double>().Run(IndividualProblem(), Options(1));

        var counter = new BatchCounter();
        var result = new EvolutionEngine<double>().Run(BatchProblem(preferredBatchSize, counter), Options(workers));

        Assert.Equal(reference.GenerationsRun, result.GenerationsRun);
        Assert.Equal(reference.Termination, result.Termination);
        Assert.Equal(reference.BestFitness.Value, result.BestFitness.Value);
    }

    [Fact]
    public void Non_positive_preferred_batch_size_falls_back_to_single_genome_batches()
    {
        var counter = new BatchCounter();
        var reference = new EvolutionEngine<double>().Run(IndividualProblem(), Options(2));

        var result = new EvolutionEngine<double>().Run(BatchProblem(0, counter), Options(2));

        Assert.True(counter.Calls > 0);
        Assert.Equal(counter.Calls, counter.Genomes);
        Assert.Equal(reference.BestFitness.Value, result.BestFitness.Value);
    }
}
