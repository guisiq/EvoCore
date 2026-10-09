using Evo.Core.Values;
using Xunit;

namespace Evo.Core.Tests;

public sealed class BuiltInParameterStrategyTests
{
    private const int Samples = 2000;

    [Fact]
    public void ParameterStrategy_double_create_stays_in_initial_range_and_is_deterministic()
    {
        var a = RandomStream.Create(42, 0, 0, 1);
        var b = RandomStream.Create(42, 0, 0, 1);
        for (int i = 0; i < Samples; i++)
        {
            double va = DoubleParameterStrategy.Instance.Create(ref a);
            double vb = DoubleParameterStrategy.Instance.Create(ref b);
            Assert.InRange(va, -5.0, 5.0);
            Assert.Equal(va, vb);
        }
    }

    [Fact]
    public void ParameterStrategy_double_mutate_is_clamped_and_differs_from_current()
    {
        var rng = RandomStream.Create(7, 1, 2, 3);
        foreach (double current in new[] { 0.0, 1e6, -1e6, 999999.9999 })
        {
            for (int i = 0; i < 200; i++)
            {
                double next = DoubleParameterStrategy.Instance.Mutate(current, ref rng);
                Assert.InRange(next, -1e6, 1e6);
                Assert.NotEqual(current, next);
            }
        }
    }

    [Fact]
    public void ParameterStrategy_int_create_stays_in_range_and_is_deterministic()
    {
        var a = RandomStream.Create(42, 0, 0, 2);
        var b = RandomStream.Create(42, 0, 0, 2);
        for (int i = 0; i < Samples; i++)
        {
            int va = IntParameterStrategy.Instance.Create(ref a);
            int vb = IntParameterStrategy.Instance.Create(ref b);
            Assert.InRange(va, -10, 10);
            Assert.Equal(va, vb);
        }
    }

    [Fact]
    public void ParameterStrategy_int_mutate_uses_non_zero_delta_in_range_and_differs_from_current()
    {
        var rng = RandomStream.Create(7, 1, 2, 4);
        for (int i = 0; i < Samples; i++)
        {
            int current = 0;
            int next = IntParameterStrategy.Instance.Mutate(current, ref rng);
            long delta = (long)next - current;
            Assert.NotEqual(0, delta);
            Assert.InRange(delta, -3, 3);
            Assert.NotEqual(current, next);
        }
    }

    [Fact]
    public void ParameterStrategy_int_mutate_clamps_at_int_bounds_and_still_differs()
    {
        var rng = RandomStream.Create(7, 1, 2, 5);
        for (int i = 0; i < Samples; i++)
        {
            int high = IntParameterStrategy.Instance.Mutate(int.MaxValue, ref rng);
            int low = IntParameterStrategy.Instance.Mutate(int.MinValue, ref rng);
            Assert.NotEqual(int.MaxValue, high);
            Assert.NotEqual(int.MinValue, low);
        }
    }

    [Fact]
    public void ParameterStrategy_double_array_create_has_fixed_length_and_is_deterministic()
    {
        var strategy = new DoubleArrayParameterStrategy(4);
        var a = RandomStream.Create(42, 0, 0, 6);
        var b = RandomStream.Create(42, 0, 0, 6);
        double[] va = strategy.Create(ref a);
        double[] vb = strategy.Create(ref b);
        Assert.Equal(4, va.Length);
        Assert.Equal(va, vb);
        foreach (double v in va)
        {
            Assert.InRange(v, -5.0, 5.0);
        }
    }

    [Fact]
    public void ParameterStrategy_double_array_mutate_returns_new_array_without_touching_input()
    {
        var strategy = new DoubleArrayParameterStrategy(3);
        var rng = RandomStream.Create(7, 1, 2, 7);
        double[] input = [1.0, 2.0, 3.0];
        double[] snapshot = (double[])input.Clone();

        double[] next = strategy.Mutate(input, ref rng);

        Assert.NotSame(input, next);
        Assert.Equal(snapshot, input);
        Assert.Equal(3, next.Length);
        Assert.NotEqual(input, next);
    }

    [Fact]
    public void ParameterStrategy_int_array_create_has_fixed_length_and_is_deterministic()
    {
        var strategy = new IntArrayParameterStrategy(5);
        var a = RandomStream.Create(42, 0, 0, 8);
        var b = RandomStream.Create(42, 0, 0, 8);
        int[] va = strategy.Create(ref a);
        int[] vb = strategy.Create(ref b);
        Assert.Equal(5, va.Length);
        Assert.Equal(va, vb);
        foreach (int v in va)
        {
            Assert.InRange(v, -10, 10);
        }
    }

    [Fact]
    public void ParameterStrategy_int_array_mutate_returns_new_array_without_touching_input()
    {
        var strategy = new IntArrayParameterStrategy(4);
        var rng = RandomStream.Create(7, 1, 2, 9);
        int[] input = [1, 2, 3, 4];
        int[] snapshot = (int[])input.Clone();

        int[] next = strategy.Mutate(input, ref rng);

        Assert.NotSame(input, next);
        Assert.Equal(snapshot, input);
        Assert.Equal(4, next.Length);
        Assert.NotEqual(input, next);
    }

    [Fact]
    public void ParameterStrategy_array_mutate_rejects_wrong_length()
    {
        var rng = RandomStream.Create(7, 1, 2, 10);
        Assert.Throws<ArgumentException>(() => new IntArrayParameterStrategy(2).Mutate([1, 2, 3], ref rng));
        Assert.Throws<ArgumentException>(() => new DoubleArrayParameterStrategy(2).Mutate([1.0], ref rng));
    }
}
