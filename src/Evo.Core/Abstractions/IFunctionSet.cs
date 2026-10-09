namespace Evo.Core;

/// <summary>
/// Ordered, immutable set of node functions. The index of a function is its
/// function gene id and is part of the persisted genome and of the phenotype
/// fingerprint, so the order must not change without a schema version change.
/// </summary>
/// <typeparam name="T">Logical value produced by a terminal/node.</typeparam>
public interface IFunctionSet<T>
{
    /// <summary>Stable identifier of the function set, persisted and fingerprinted.</summary>
    string Id { get; }

    /// <summary>Number of functions in the set.</summary>
    int Count { get; }

    /// <summary>Returns the function with the given function gene id.</summary>
    IFunction<T> Get(int functionId);
}
