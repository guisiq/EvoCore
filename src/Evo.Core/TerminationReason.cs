namespace Evo.Core;

/// <summary>
/// Reason why an evolutionary run stopped (02-dag-e-evolucao.md section 8).
/// </summary>
public enum TerminationReason
{
    /// <summary>Generation budget exhausted.</summary>
    MaxGenerations = 0,

    /// <summary>Target fitness reached.</summary>
    TargetFitness = 1,

    /// <summary>No improvement for the configured number of generations.</summary>
    Stagnation = 2,

    /// <summary>Wall-clock budget exhausted. Excluded from worker-count equivalence.</summary>
    TimeLimit = 3,

    /// <summary>External cancellation. Excluded from worker-count equivalence.</summary>
    Cancellation = 4,
}
