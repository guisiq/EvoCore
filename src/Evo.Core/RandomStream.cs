namespace Evo.Core;

/// <summary>
/// Deterministic RNG stream handle. The algorithm, stream derivation and the
/// reference vectors are normative in 02-dag-e-evolucao.md section 5 and are
/// implemented by EVO-003; EVO-001 only freezes the public type seam so that
/// <see cref="IParameterGeneStrategy{T}"/> can take it by reference.
/// </summary>
/// <remarks>
/// The type is a mutable value type passed by <c>ref</c>: advancing a stream
/// mutates the caller's state. It is not thread-safe; each worker owns its own
/// stream instance.
/// </remarks>
public struct RandomStream
{
    /// <summary>
    /// Frozen RNG algorithm/version identifier. Changing it breaks reproducibility
    /// and requires a new checkpoint schema version.
    /// </summary>
    public const string Version = "rng/xoshiro256ss-sha256-key/v1";
}
