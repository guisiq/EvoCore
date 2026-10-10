using Xunit;

namespace Evo.Ilgpu.Tests;

/// <summary>
/// Regressões diretas dos validadores de paridade usados pelos testes e pelo benchmark ILGPU.
/// Comprovam que NaN e Infinity, e infinitos de sinais opostos, não são aceitos como equivalentes.
/// </summary>
public sealed class IlgpuTestHelpersValidatorTests
{
    [Fact]
    public void AssertDoubleClose_rejects_nan_versus_positive_infinity()
    {
        var ex = Record.Exception(() => IlgpuTestHelpers.AssertDoubleClose(double.NaN, double.PositiveInfinity));
        Assert.NotNull(ex);
    }

    [Fact]
    public void AssertDoubleClose_rejects_positive_versus_negative_infinity()
    {
        var ex = Record.Exception(() => IlgpuTestHelpers.AssertDoubleClose(double.PositiveInfinity, double.NegativeInfinity));
        Assert.NotNull(ex);
    }

    [Fact]
    public void AssertDoubleClose_accepts_nan_versus_nan()
    {
        IlgpuTestHelpers.AssertDoubleClose(double.NaN, double.NaN);
    }

    [Fact]
    public void AssertDoubleClose_accepts_same_signed_infinity()
    {
        IlgpuTestHelpers.AssertDoubleClose(double.PositiveInfinity, double.PositiveInfinity);
        IlgpuTestHelpers.AssertDoubleClose(double.NegativeInfinity, double.NegativeInfinity);
    }

    [Fact]
    public void AssertDoubleClose_still_accepts_finite_values_within_tolerance()
    {
        IlgpuTestHelpers.AssertDoubleClose(1.0, 1.0 + 1e-13);
    }

    [Fact]
    public void AssertDoubleArrayClose_rejects_nan_versus_infinity_element_wise()
    {
        var ex = Record.Exception(() => IlgpuTestHelpers.AssertDoubleArrayClose(
            new[] { 1.0, double.NaN },
            new[] { 1.0, double.PositiveInfinity },
            "array"));
        Assert.NotNull(ex);
    }

    [Fact]
    public void AssertDoubleArrayClose_rejects_opposite_signed_infinity_element_wise()
    {
        var ex = Record.Exception(() => IlgpuTestHelpers.AssertDoubleArrayClose(
            new[] { double.PositiveInfinity, 2.0 },
            new[] { double.NegativeInfinity, 2.0 },
            "array"));
        Assert.NotNull(ex);
    }
}
