using System.Buffers.Binary;
using System.Security.Cryptography;

namespace Evo.Core.Values;

/// <summary>
/// Canonical little-endian encoders used by the built-in value semantics
/// (02-dag-e-evolucao.md section 3).
/// </summary>
internal static class FingerprintEncoding
{
    internal static void AppendInt32(IncrementalHash hash, int value)
    {
        Span<byte> buffer = stackalloc byte[sizeof(int)];
        BinaryPrimitives.WriteInt32LittleEndian(buffer, value);
        hash.AppendData(buffer);
    }

    internal static void AppendDouble(IncrementalHash hash, double value)
    {
        Span<byte> buffer = stackalloc byte[sizeof(long)];
        BinaryPrimitives.WriteInt64LittleEndian(buffer, BitConverter.DoubleToInt64Bits(value));
        hash.AppendData(buffer);
    }
}

/// <summary>Value semantics for <see cref="double"/>.</summary>
public sealed class DoubleValueSemantics : IValueSemantics<double>
{
    /// <summary>Shared stateless instance.</summary>
    public static readonly DoubleValueSemantics Instance = new();

    /// <inheritdoc />
    public string ValueTypeId => "double";

    /// <inheritdoc />
    public double Clone(double value) => value;

    /// <inheritdoc />
    public bool ValueEquals(double left, double right) =>
        BitConverter.DoubleToInt64Bits(left) == BitConverter.DoubleToInt64Bits(right);

    /// <inheritdoc />
    public void AppendFingerprint(double value, IncrementalHash hash)
    {
        ArgumentNullException.ThrowIfNull(hash);
        FingerprintEncoding.AppendDouble(hash, value);
    }
}

/// <summary>Value semantics for <see cref="int"/>.</summary>
public sealed class Int32ValueSemantics : IValueSemantics<int>
{
    /// <summary>Shared stateless instance.</summary>
    public static readonly Int32ValueSemantics Instance = new();

    /// <inheritdoc />
    public string ValueTypeId => "int32";

    /// <inheritdoc />
    public int Clone(int value) => value;

    /// <inheritdoc />
    public bool ValueEquals(int left, int right) => left == right;

    /// <inheritdoc />
    public void AppendFingerprint(int value, IncrementalHash hash)
    {
        ArgumentNullException.ThrowIfNull(hash);
        FingerprintEncoding.AppendInt32(hash, value);
    }
}

/// <summary>Deep value semantics for <c>double[]</c>.</summary>
public sealed class DoubleArrayValueSemantics : IValueSemantics<double[]>
{
    /// <summary>Shared stateless instance.</summary>
    public static readonly DoubleArrayValueSemantics Instance = new();

    /// <inheritdoc />
    public string ValueTypeId => "double[]";

    /// <inheritdoc />
    public double[] Clone(double[] value)
    {
        ArgumentNullException.ThrowIfNull(value);
        return (double[])value.Clone();
    }

    /// <inheritdoc />
    public bool ValueEquals(double[] left, double[] right)
    {
        ArgumentNullException.ThrowIfNull(left);
        ArgumentNullException.ThrowIfNull(right);
        if (left.Length != right.Length)
        {
            return false;
        }

        for (var i = 0; i < left.Length; i++)
        {
            if (BitConverter.DoubleToInt64Bits(left[i]) != BitConverter.DoubleToInt64Bits(right[i]))
            {
                return false;
            }
        }

        return true;
    }

    /// <inheritdoc />
    public void AppendFingerprint(double[] value, IncrementalHash hash)
    {
        ArgumentNullException.ThrowIfNull(value);
        ArgumentNullException.ThrowIfNull(hash);

        FingerprintEncoding.AppendInt32(hash, value.Length);
        foreach (var element in value)
        {
            FingerprintEncoding.AppendDouble(hash, element);
        }
    }
}

/// <summary>Deep value semantics for <c>int[]</c>.</summary>
public sealed class Int32ArrayValueSemantics : IValueSemantics<int[]>
{
    /// <summary>Shared stateless instance.</summary>
    public static readonly Int32ArrayValueSemantics Instance = new();

    /// <inheritdoc />
    public string ValueTypeId => "int32[]";

    /// <inheritdoc />
    public int[] Clone(int[] value)
    {
        ArgumentNullException.ThrowIfNull(value);
        return (int[])value.Clone();
    }

    /// <inheritdoc />
    public bool ValueEquals(int[] left, int[] right)
    {
        ArgumentNullException.ThrowIfNull(left);
        ArgumentNullException.ThrowIfNull(right);
        return left.AsSpan().SequenceEqual(right);
    }

    /// <inheritdoc />
    public void AppendFingerprint(int[] value, IncrementalHash hash)
    {
        ArgumentNullException.ThrowIfNull(value);
        ArgumentNullException.ThrowIfNull(hash);

        FingerprintEncoding.AppendInt32(hash, value.Length);
        foreach (var element in value)
        {
            FingerprintEncoding.AppendInt32(hash, element);
        }
    }
}
