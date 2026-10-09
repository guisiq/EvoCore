namespace Evo.Core;

/// <summary>
/// Evaluates the active DAG of one genome for one case.
/// </summary>
/// <remarks>
/// Implementations may own mutable scratch buffers and are therefore not
/// required to be thread-safe: each worker owns its own evaluator instance.
/// This is the zero-allocation hot path; convenience array overloads live in
/// <see cref="GenomeEvaluatorExtensions"/>.
/// </remarks>
/// <typeparam name="T">Logical value produced by a terminal/node.</typeparam>
public interface IGenomeEvaluator<T>
{
    /// <summary>Evaluates <paramref name="genome"/> writing one value per output gene.</summary>
    void Evaluate(
        Genome<T> genome,
        ReadOnlySpan<T> inputs,
        Span<T> outputs);
}
