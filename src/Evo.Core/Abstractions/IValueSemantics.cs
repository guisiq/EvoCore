namespace Evo.Core;

/// <summary>
/// Deep, content-based semantics of the value domain <typeparamref name="T"/>.
/// Arrays are value-domain objects: clone, equality and fingerprint are deep.
/// Implementations must be stateless and safe for concurrent use.
/// </summary>
/// <typeparam name="T">Logical value produced by a terminal/node.</typeparam>
public interface IValueSemantics<T>
{
    /// <summary>Stable identifier of the value domain, persisted in genome/checkpoint JSON.</summary>
    string ValueTypeId { get; }

    /// <summary>Returns a deep copy of <paramref name="value"/>.</summary>
    T Clone(T value);

    /// <summary>Deep, content-based equality.</summary>
    bool ValueEquals(T left, T right);

    /// <summary>Appends the canonical byte encoding of the value to a fingerprint hash.</summary>
    void AppendFingerprint(T value, System.Security.Cryptography.IncrementalHash hash);
}
