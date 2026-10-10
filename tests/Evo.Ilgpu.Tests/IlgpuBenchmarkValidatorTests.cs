using Evo.Benchmarks;
using Xunit;

namespace Evo.Ilgpu.Tests;

public sealed class IlgpuBenchmarkValidatorTests
{
    [Fact]
    public void Benchmark_scalar_validator_rejects_nan_vs_infinity()
    {
        var error = Assert.Throws<InvalidOperationException>(() => IlgpuBenchmarkProgram.AssertDouble(double.NaN, double.PositiveInfinity));
        Assert.Contains("paridade double falhou", error.Message);
    }

    [Fact]
    public void Benchmark_scalar_validator_rejects_opposite_signed_infinities()
    {
        var error = Assert.Throws<InvalidOperationException>(() => IlgpuBenchmarkProgram.AssertDouble(double.PositiveInfinity, double.NegativeInfinity));
        Assert.Contains("paridade double falhou", error.Message);
    }

    [Fact]
    public void Benchmark_array_validator_rejects_nan_vs_infinity_and_opposite_signed_infinities()
    {
        Assert.Throws<InvalidOperationException>(() =>
            IlgpuBenchmarkProgram.AssertDoubleArray(
                [double.NaN, double.PositiveInfinity],
                [double.PositiveInfinity, double.NegativeInfinity]));
    }
}
