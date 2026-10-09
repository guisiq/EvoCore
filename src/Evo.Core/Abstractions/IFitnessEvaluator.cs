namespace Evo.Core;

/// <summary>
/// Computes the fitness of a single individual. The evaluator owns its dataset
/// and decides how many cases to run, so the core never needs to know datasets.
/// </summary>
/// <remarks>
/// V1 canonical contract: one individual per call. Batch evaluation of multiple
/// individuals is deferred to v2 (05-v2-batch-evaluation.md) and must not be
/// implemented here. Instances are not required to be thread-safe; the engine
/// creates one per worker via <see cref="IGeneticProblem{T}.CreateFitnessEvaluator"/>.
/// </remarks>
/// <typeparam name="T">Logical value produced by a terminal/node.</typeparam>
public interface IFitnessEvaluator<T>
{
    /// <summary>Evaluates the fitness of one genome.</summary>
    FitnessResult Evaluate(
        Genome<T> genome,
        IGenomeEvaluator<T> evaluator,
        CancellationToken cancellationToken);
}
