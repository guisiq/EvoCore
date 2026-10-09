using System.Buffers.Binary;
using System.Security.Cryptography;

namespace Evo.Core;

/// <summary>
/// Deterministic RNG stream handle implementing <c>rng/xoshiro256ss-sha256-key/v1</c>
/// (02-dag-e-evolucao.md section 5). The stream state is derived from a SHA-256
/// digest of the logical key <c>(rootSeed, generation, individualIndex, streamId)</c>.
/// </summary>
/// <remarks>
/// The type is a mutable value type passed by <c>ref</c>: advancing a stream
/// mutates the caller's state. It is not thread-safe; each worker owns its own
/// stream instance. There is no global RNG state.
/// </remarks>
public struct RandomStream
{
    /// <summary>
    /// Frozen RNG algorithm/version identifier. Changing it breaks reproducibility
    /// and requires a new checkpoint schema version.
    /// </summary>
    public const string Version = "rng/xoshiro256ss-sha256-key/v1";

    private ulong _s0;
    private ulong _s1;
    private ulong _s2;
    private ulong _s3;

    private RandomStream(ulong s0, ulong s1, ulong s2, ulong s3)
    {
        _s0 = s0;
        _s1 = s1;
        _s2 = s2;
        _s3 = s3;
    }

    /// <summary>
    /// Creates a stream for the logical key. The key is encoded as four
    /// little-endian <c>ulong</c> values (32 bytes); its SHA-256 digest is read as
    /// four little-endian <c>ulong</c> state words.
    /// </summary>
    public static RandomStream Create(ulong rootSeed, ulong generation, ulong individualIndex, ulong streamId)
    {
        Span<byte> key = stackalloc byte[32];
        BinaryPrimitives.WriteUInt64LittleEndian(key[0..8], rootSeed);
        BinaryPrimitives.WriteUInt64LittleEndian(key[8..16], generation);
        BinaryPrimitives.WriteUInt64LittleEndian(key[16..24], individualIndex);
        BinaryPrimitives.WriteUInt64LittleEndian(key[24..32], streamId);

        Span<byte> digest = stackalloc byte[SHA256.HashSizeInBytes];
        SHA256.HashData(key, digest);

        return new RandomStream(
            BinaryPrimitives.ReadUInt64LittleEndian(digest[0..8]),
            BinaryPrimitives.ReadUInt64LittleEndian(digest[8..16]),
            BinaryPrimitives.ReadUInt64LittleEndian(digest[16..24]),
            BinaryPrimitives.ReadUInt64LittleEndian(digest[24..32]));
    }

    /// <summary>Advances the stream and returns the next 64-bit output (xoshiro256**).</summary>
    public ulong NextUInt64()
    {
        ulong result = RotateLeft(_s1 * 5, 7) * 9;
        ulong t = _s1 << 17;

        _s2 ^= _s0;
        _s3 ^= _s1;
        _s1 ^= _s2;
        _s0 ^= _s3;
        _s2 ^= t;
        _s3 = RotateLeft(_s3, 45);

        return result;
    }

    /// <summary>Returns a <c>double</c> in <c>[0,1)</c> built from 53 random bits.</summary>
    public double NextDouble()
        => (NextUInt64() >> 11) * (1.0 / 9007199254740992.0);

    /// <summary>Returns a <c>float</c> in <c>[0,1)</c> built from 24 random bits.</summary>
    public float NextSingle()
        => (NextUInt64() >> 40) * (1.0f / 16777216.0f);

    /// <summary>
    /// Returns an unbiased <c>int</c> in <c>[minInclusive, maxExclusive)</c> using
    /// rejection sampling.
    /// </summary>
    public int NextInt(int minInclusive, int maxExclusive)
    {
        if (minInclusive >= maxExclusive)
        {
            throw new ArgumentOutOfRangeException(nameof(maxExclusive), "maxExclusive must be greater than minInclusive.");
        }

        ulong range = (ulong)((long)maxExclusive - minInclusive);
        // Raw outputs below this threshold would bias `x % range`; they are rejected.
        ulong threshold = (0UL - range) % range;

        while (true)
        {
            ulong x = NextUInt64();
            if (x >= threshold)
            {
                return (int)(minInclusive + (long)(x % range));
            }
        }
    }

    /// <summary>Bernoulli trial: returns <c>true</c> with probability <paramref name="p"/>.</summary>
    public bool NextBernoulli(double p)
        => NextDouble() < p;

    /// <summary>Shuffles the span in place with Fisher-Yates driven by <see cref="NextInt"/>.</summary>
    public void Shuffle<T>(Span<T> values)
    {
        for (int i = values.Length - 1; i > 0; i--)
        {
            int j = NextInt(0, i + 1);
            (values[i], values[j]) = (values[j], values[i]);
        }
    }

    private static ulong RotateLeft(ulong value, int offset)
        => (value << offset) | (value >> (64 - offset));
}
