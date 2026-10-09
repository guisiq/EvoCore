namespace Evo.Core;

/// <summary>
/// Pure node function. Implementations must be stateless, deterministic and
/// safe for concurrent use across workers.
/// </summary>
/// <typeparam name="T">Logical value produced by a terminal/node.</typeparam>
public interface IFunction<T>
{
    /// <summary>Stable identifier of the function inside its function set.</summary>
    string Id { get; }

    /// <summary>Number of connection genes effectively used by this function.</summary>
    int Arity { get; }

    /// <summary>Evaluates the function for one case.</summary>
    T Invoke(ReadOnlySpan<T> arguments, T parameter);
}
