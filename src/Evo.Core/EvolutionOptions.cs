namespace Evo.Core;

/// <summary>
/// Configuration of an evolutionary run. Defaults are the normative demo defaults
/// of 02-dag-e-evolucao.md section 6 and must not be tuned to make a quality
/// benchmark pass.
/// </summary>
public sealed record EvolutionOptions
{
    /// <summary>Number of individuals per generation.</summary>
    public int PopulationSize { get; init; } = 500;

    /// <summary>Maximum number of generations.</summary>
    public int MaxGenerations { get; init; } = 200;

    /// <summary>Tournament size used by parent selection.</summary>
    public int TournamentSize { get; init; } = 5;

    /// <summary>Number of elite individuals copied unchanged.</summary>
    public int EliteCount { get; init; } = 1;

    /// <summary>Probability of performing uniform crossover for a non-elite offspring.</summary>
    public double CrossoverRate { get; init; } = 0.50;

    /// <summary>Per gene slot probability of taking the gene from parent B.</summary>
    public double UniformCrossoverGeneProbability { get; init; } = 0.50;

    /// <summary>Root seed of the deterministic RNG.</summary>
    public ulong Seed { get; init; } = 42;

    /// <summary>Worker count. Zero means autodetect.</summary>
    public int WorkerCount { get; init; }

    /// <summary>Optional fitness value that stops the run when reached.</summary>
    public double? TargetFitness { get; init; }

    /// <summary>Optional number of generations without improvement that stops the run.</summary>
    public int? StagnationGenerations { get; init; }

    /// <summary>
    /// Optional wall-clock budget. Excluded from the bit-for-bit equivalence
    /// guarantee across worker counts.
    /// </summary>
    public TimeSpan? TimeLimit { get; init; }
}
