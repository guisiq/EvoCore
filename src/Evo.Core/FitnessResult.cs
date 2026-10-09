namespace Evo.Core;

/// <summary>
/// Result of evaluating the fitness of one individual.
/// </summary>
/// <param name="Value">Fitness value. Higher is better.</param>
/// <param name="IsValid">
/// False when the individual produced an invalid evaluation (for example a
/// NaN/Infinity output). Invalid fitness must never contaminate ranking.
/// </param>
public readonly record struct FitnessResult(double Value, bool IsValid)
{
    /// <summary>Creates a valid fitness result.</summary>
    public static FitnessResult Valid(double value) => new(value, true);

    /// <summary>Creates an invalid fitness result.</summary>
    public static FitnessResult Invalid() => new(double.NegativeInfinity, false);
}
