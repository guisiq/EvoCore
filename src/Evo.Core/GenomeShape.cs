namespace Evo.Core;

/// <summary>
/// Fixed CGP genome shape. Frozen public contract (01-escopo-e-api.md).
/// </summary>
public readonly record struct GenomeShape(
    int InputCount,
    int NodeCount,
    int OutputCount,
    int MaxArity,
    int LevelsBack);
