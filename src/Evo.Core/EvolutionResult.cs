namespace Evo.Core;

/// <summary>
/// Outcome of an evolutionary run.
/// </summary>
/// <typeparam name="T">Logical value produced by a terminal/node.</typeparam>
public sealed record EvolutionResult<T>(
    Genome<T> BestGenome,
    FitnessResult BestFitness,
    int GenerationsRun,
    TerminationReason Termination);
