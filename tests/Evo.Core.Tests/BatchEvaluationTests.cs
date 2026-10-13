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

/// <summary>Fitness genérico: erro médio entre a saída e <c>target</c> sobre entradas fixas; inválido se o erro não for finito.</summary>
internal sealed class ScalarFitness<T> : IFitnessEvaluator<T>
{
    private readonly T[] _inputs;
    private readonly Func<T, T> _target;
    private readonly Func<T, T, double> _error;

    public ScalarFitness(T[] inputs, Func<T, T> target, Func<T, T, double> error)
    {
        _inputs = inputs;
        _target = target;
        _error = error;
    }

    public FitnessResult Evaluate(Genome<T> genome, IGenomeEvaluator<T> evaluator, CancellationToken cancellationToken)
    {
        var output = new T[1];
        var total = 0.0;
        foreach (var x in _inputs)
        {
            evaluator.Evaluate(genome, new[] { x }, output);
            var d = _error(output[0], _target(x));
            if (double.IsNaN(d) || double.IsInfinity(d))
            {
                return FitnessResult.Invalid();
            }

            total += d;
        }

        return FitnessResult.Valid(-total / _inputs.Length);
    }
}

/// <summary>Avaliador de lote genérico que delega cada genoma ao <see cref="ScalarFitness{T}"/> de referência.</summary>
internal sealed class BatchScalarFitness<T> : IBatchFitnessEvaluator<T>
{
    private readonly ScalarFitness<T> _single;
    private readonly BatchCounter _counter;

    public BatchScalarFitness(ScalarFitness<T> single, int preferredBatchSize, BatchCounter counter)
    {
        _single = single;
        PreferredBatchSize = preferredBatchSize;
        _counter = counter;
    }

    public int PreferredBatchSize { get; }

    public FitnessResult Evaluate(Genome<T> genome, IGenomeEvaluator<T> evaluator, CancellationToken cancellationToken) =>
        _single.Evaluate(genome, evaluator, cancellationToken);

    public void EvaluateBatch(
        ReadOnlySpan<Genome<T>> genomes,
        IGenomeEvaluator<T> evaluator,
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

/// <summary>Avaliador de lote que invoca um gancho a cada genoma (para injetar cancelamento ou falha).</summary>
internal sealed class HookedBatchFitness : IBatchFitnessEvaluator<double>
{
    private readonly QuadraticFitness _single = new();
    private readonly Action<int> _hook;
    private int _count;

    public HookedBatchFitness(int preferredBatchSize, Action<int> hook)
    {
        PreferredBatchSize = preferredBatchSize;
        _hook = hook;
    }

    public int PreferredBatchSize { get; }

    public FitnessResult Evaluate(Genome<double> genome, IGenomeEvaluator<double> evaluator, CancellationToken cancellationToken)
    {
        _hook(Interlocked.Increment(ref _count));
        return _single.Evaluate(genome, evaluator, cancellationToken);
    }

    public void EvaluateBatch(
        ReadOnlySpan<Genome<double>> genomes,
        IGenomeEvaluator<double> evaluator,
        Span<FitnessResult> results,
        CancellationToken cancellationToken)
    {
        for (var i = 0; i < genomes.Length; i++)
        {
            results[i] = Evaluate(genomes[i], evaluator, cancellationToken);
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

    private static EvolutionOptions ParityOptions() => Options(3);

    [Theory]
    [InlineData(1, 1)]
    [InlineData(4, 2)]
    [InlineData(16, 4)]
    public void Batch_cancellation_surfaces_operation_canceled_without_partial_results(int preferredBatchSize, int workers)
    {
        using var cts = new CancellationTokenSource();
        var problem = new EvolutionTestProblem<double>(
            Shape,
            StandardDoubleFunctionSet.Instance,
            DoubleValueSemantics.Instance,
            DoubleParameterStrategy.Instance,
            () => new HookedBatchFitness(preferredBatchSize, n => { if (n == 40) cts.Cancel(); }));

        var cancelled = new EvolutionEngine<double>().Run(problem, Options(workers), cts.Token);
        Assert.Equal(TerminationReason.Cancellation, cancelled.Termination);
        Assert.True(cancelled.GenerationsRun < Options(workers).MaxGenerations);
        Assert.True(cancelled.BestFitness.IsValid);

        var reference = new EvolutionEngine<double>().Run(IndividualProblem(), Options(workers));
        var rerun = new EvolutionEngine<double>().Run(IndividualProblem(), Options(workers));
        Assert.Equal(reference.BestFitness.Value, rerun.BestFitness.Value);
        Assert.Equal(reference.GenerationsRun, rerun.GenerationsRun);
    }

    [Theory]
    [InlineData(1, 1)]
    [InlineData(4, 2)]
    [InlineData(16, 4)]
    public void Failing_genome_in_batch_does_not_corrupt_subsequent_evaluations(int preferredBatchSize, int workers)
    {
        var problem = new EvolutionTestProblem<double>(
            Shape,
            StandardDoubleFunctionSet.Instance,
            DoubleValueSemantics.Instance,
            DoubleParameterStrategy.Instance,
            () => new HookedBatchFitness(preferredBatchSize, n => { if (n == 25) throw new InvalidOperationException("falha injetada"); }));

        Assert.ThrowsAny<Exception>(() => new EvolutionEngine<double>().Run(problem, Options(workers)));

        var reference = new EvolutionEngine<double>().Run(IndividualProblem(), Options(workers));
        var batched = new EvolutionEngine<double>().Run(BatchProblem(preferredBatchSize, new BatchCounter()), Options(workers));
        Assert.Equal(reference.GenerationsRun, batched.GenerationsRun);
        Assert.Equal(reference.BestFitness.Value, batched.BestFitness.Value);
    }

    [Fact]
    public void Batch_contract_parity_holds_for_int()
    {
        var single = new ScalarFitness<int>(new[] { -2, -1, 0, 1, 2 }, x => x * x, (a, b) => Math.Abs((double)a - b));
        var shape = new GenomeShape(1, 8, 1, 2, 3);
        var reference = new EvolutionEngine<int>().Run(
            new EvolutionTestProblem<int>(shape, StandardInt32FunctionSet.Instance, Int32ValueSemantics.Instance, IntParameterStrategy.Instance, () => single),
            ParityOptions());

        var counter = new BatchCounter();
        var batched = new EvolutionEngine<int>().Run(
            new EvolutionTestProblem<int>(shape, StandardInt32FunctionSet.Instance, Int32ValueSemantics.Instance, IntParameterStrategy.Instance, () => new BatchScalarFitness<int>(single, 4, counter)),
            ParityOptions());

        Assert.True(counter.Calls > 0, "O caminho de lote não foi usado.");
        Assert.Equal(reference.GenerationsRun, batched.GenerationsRun);
        Assert.Equal(reference.Termination, batched.Termination);
        Assert.Equal(reference.BestFitness.Value, batched.BestFitness.Value);
    }

    [Fact]
    public void Batch_contract_parity_holds_for_double_array()
    {
        var inputs = new[] { new[] { -1.0, 0.5 }, new[] { 1.0, 2.0 }, new[] { 0.0, -2.0 } };
        var single = new ScalarFitness<double[]>(inputs, v => v.Select(e => e * e).ToArray(), (a, b) => a.Zip(b, (p, q) => Math.Abs(p - q)).Sum());
        var shape = new GenomeShape(1, 8, 1, 2, 3);
        var reference = new EvolutionEngine<double[]>().Run(
            new EvolutionTestProblem<double[]>(shape, StandardDoubleArrayFunctionSet.Instance, DoubleArrayValueSemantics.Instance, new DoubleArrayParameterStrategy(2), () => single),
            ParityOptions());

        var counter = new BatchCounter();
        var batched = new EvolutionEngine<double[]>().Run(
            new EvolutionTestProblem<double[]>(shape, StandardDoubleArrayFunctionSet.Instance, DoubleArrayValueSemantics.Instance, new DoubleArrayParameterStrategy(2), () => new BatchScalarFitness<double[]>(single, 4, counter)),
            ParityOptions());

        Assert.True(counter.Calls > 0, "O caminho de lote não foi usado.");
        Assert.Equal(reference.GenerationsRun, batched.GenerationsRun);
        Assert.Equal(reference.Termination, batched.Termination);
        Assert.Equal(reference.BestFitness.Value, batched.BestFitness.Value);
    }

    [Fact]
    public void Batch_contract_parity_holds_for_int_array()
    {
        var inputs = new[] { new[] { -1, 2 }, new[] { 1, 0 }, new[] { 3, -2 } };
        var single = new ScalarFitness<int[]>(inputs, v => v.Select(e => e * e).ToArray(), (a, b) => a.Zip(b, (p, q) => Math.Abs((double)p - q)).Sum());
        var shape = new GenomeShape(1, 8, 1, 2, 3);
        var reference = new EvolutionEngine<int[]>().Run(
            new EvolutionTestProblem<int[]>(shape, StandardInt32ArrayFunctionSet.Instance, Int32ArrayValueSemantics.Instance, new IntArrayParameterStrategy(2), () => single),
            ParityOptions());

        var counter = new BatchCounter();
        var batched = new EvolutionEngine<int[]>().Run(
            new EvolutionTestProblem<int[]>(shape, StandardInt32ArrayFunctionSet.Instance, Int32ArrayValueSemantics.Instance, new IntArrayParameterStrategy(2), () => new BatchScalarFitness<int[]>(single, 4, counter)),
            ParityOptions());

        Assert.True(counter.Calls > 0, "O caminho de lote não foi usado.");
        Assert.Equal(reference.GenerationsRun, batched.GenerationsRun);
        Assert.Equal(reference.Termination, batched.Termination);
        Assert.Equal(reference.BestFitness.Value, batched.BestFitness.Value);
    }
}
