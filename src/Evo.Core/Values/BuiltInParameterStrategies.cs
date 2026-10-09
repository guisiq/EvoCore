namespace Evo.Core.Values;

/// <summary>
/// Built-in <see cref="double"/> parameter strategy (02-dag-e-evolucao.md section 6).
/// Initial value is uniform in <c>[-5,5)</c>; mutation adds <c>Normal(0, 0.5)</c>
/// (Box-Muller) and clamps to <c>[-1e6,1e6]</c>.
/// </summary>
public sealed class DoubleParameterStrategy : IParameterGeneStrategy<double>
{
    /// <summary>Shared stateless instance.</summary>
    public static readonly DoubleParameterStrategy Instance = new();

    private const double InitialMin = -5.0;
    private const double InitialMax = 5.0;
    private const double MutationStdDev = 0.5;
    private const double Limit = 1e6;
    private const int MaxMutationAttempts = 64;

    /// <inheritdoc />
    public double Create(ref RandomStream random)
        => InitialMin + (InitialMax - InitialMin) * random.NextDouble();

    /// <inheritdoc />
    public double Mutate(double current, ref RandomStream random)
    {
        for (int attempt = 0; attempt < MaxMutationAttempts; attempt++)
        {
            double next = Math.Clamp(current + MutationStdDev * NextStandardNormal(ref random), -Limit, Limit);
            if (next != current)
            {
                return next;
            }
        }

        // Fallback when clamping or rounding keeps the value fixed: step one ulp inward.
        return current >= Limit ? Math.BitDecrement(current) : Math.BitIncrement(current);
    }

    private static double NextStandardNormal(ref RandomStream random)
    {
        // u1 in (0,1] avoids log(0).
        double u1 = 1.0 - random.NextDouble();
        double u2 = random.NextDouble();
        return Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
    }
}

/// <summary>
/// Built-in <see cref="int"/> parameter strategy (02-dag-e-evolucao.md section 6).
/// Initial value is uniform in <c>[-10,11)</c>; mutation adds a non-zero delta in
/// <c>[-3,4)</c> and clamps to the <see cref="int"/> range.
/// </summary>
public sealed class IntParameterStrategy : IParameterGeneStrategy<int>
{
    /// <summary>Shared stateless instance.</summary>
    public static readonly IntParameterStrategy Instance = new();

    /// <inheritdoc />
    public int Create(ref RandomStream random)
        => random.NextInt(-10, 11);

    /// <inheritdoc />
    public int Mutate(int current, ref RandomStream random)
    {
        int delta = NextNonZeroDelta(ref random);
        int next = Clamp(current, delta);
        if (next == current)
        {
            // Only reachable at int bounds: the opposite sign always moves the value.
            next = Clamp(current, -delta);
        }

        return next;
    }

    internal static int NextNonZeroDelta(ref RandomStream random)
    {
        int delta;
        do
        {
            delta = random.NextInt(-3, 4);
        }
        while (delta == 0);

        return delta;
    }

    private static int Clamp(int current, int delta)
    {
        long sum = (long)current + delta;
        return (int)Math.Clamp(sum, int.MinValue, int.MaxValue);
    }
}

/// <summary>
/// Built-in <see cref="double"/>[] parameter strategy with fixed length. Creation applies
/// the scalar rule to every element; mutation applies it to one uniformly chosen index
/// and always returns a new array.
/// </summary>
public sealed class DoubleArrayParameterStrategy : IParameterGeneStrategy<double[]>
{
    private readonly int _length;

    /// <summary>Creates a strategy for arrays of fixed length <paramref name="length"/>.</summary>
    public DoubleArrayParameterStrategy(int length)
    {
        if (length < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(length), "Length must be at least 1.");
        }

        _length = length;
    }

    /// <summary>Fixed length produced by this strategy.</summary>
    public int Length => _length;

    /// <inheritdoc />
    public double[] Create(ref RandomStream random)
    {
        var values = new double[_length];
        for (int i = 0; i < values.Length; i++)
        {
            values[i] = DoubleParameterStrategy.Instance.Create(ref random);
        }

        return values;
    }

    /// <inheritdoc />
    public double[] Mutate(double[] current, ref RandomStream random)
    {
        ArgumentNullException.ThrowIfNull(current);
        if (current.Length != _length)
        {
            throw new ArgumentException($"Expected array length {_length} but received {current.Length}.", nameof(current));
        }

        var next = (double[])current.Clone();
        int index = random.NextInt(0, next.Length);
        next[index] = DoubleParameterStrategy.Instance.Mutate(next[index], ref random);
        return next;
    }
}

/// <summary>
/// Built-in <see cref="int"/>[] parameter strategy with fixed length. Creation applies
/// the scalar rule to every element; mutation applies it to one uniformly chosen index
/// and always returns a new array.
/// </summary>
public sealed class IntArrayParameterStrategy : IParameterGeneStrategy<int[]>
{
    private readonly int _length;

    /// <summary>Creates a strategy for arrays of fixed length <paramref name="length"/>.</summary>
    public IntArrayParameterStrategy(int length)
    {
        if (length < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(length), "Length must be at least 1.");
        }

        _length = length;
    }

    /// <summary>Fixed length produced by this strategy.</summary>
    public int Length => _length;

    /// <inheritdoc />
    public int[] Create(ref RandomStream random)
    {
        var values = new int[_length];
        for (int i = 0; i < values.Length; i++)
        {
            values[i] = IntParameterStrategy.Instance.Create(ref random);
        }

        return values;
    }

    /// <inheritdoc />
    public int[] Mutate(int[] current, ref RandomStream random)
    {
        ArgumentNullException.ThrowIfNull(current);
        if (current.Length != _length)
        {
            throw new ArgumentException($"Expected array length {_length} but received {current.Length}.", nameof(current));
        }

        var next = (int[])current.Clone();
        int index = random.NextInt(0, next.Length);
        next[index] = IntParameterStrategy.Instance.Mutate(next[index], ref random);
        return next;
    }
}
