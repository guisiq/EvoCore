namespace Evo.Core;

/// <summary>
/// Runs a deterministic evolutionary search.
/// </summary>
/// <typeparam name="T">Logical value produced by a terminal/node.</typeparam>
public interface IEvolutionEngine<T>
{
    /// <summary>
    /// Runs the search. For deterministic termination conditions the evolutionary
    /// sequence must be identical with 1, 2 or N workers for the same seed and
    /// configuration.
    /// </summary>
    EvolutionResult<T> Run(
        IGeneticProblem<T> problem,
        EvolutionOptions options,
        CancellationToken cancellationToken = default);
}
