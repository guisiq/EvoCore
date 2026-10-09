using System.Security.Cryptography;
using Evo.Core;
using Evo.Core.Values;
using Xunit;

namespace Evo.Core.Tests;

/// <summary>
/// Smoke tests of the four official v1 value domains: <c>double</c>, <c>int</c>,
/// <c>double[]</c> and <c>int[]</c>.
/// </summary>
public sealed class ValueDomainSmokeTests
{
    private static byte[] Fingerprint<T>(IValueSemantics<T> semantics, T value)
    {
        using var hash = IncrementalHash.CreateHash(HashAlgorithmName.SHA256);
        semantics.AppendFingerprint(value, hash);
        return hash.GetHashAndReset();
    }

    private static void AssertContract<T>(IValueSemantics<T> semantics, T a, T b, string valueTypeId)
    {
        Assert.Equal(valueTypeId, semantics.ValueTypeId);

        var clone = semantics.Clone(a);
        Assert.True(semantics.ValueEquals(a, clone));
        Assert.False(semantics.ValueEquals(a, b));
        Assert.Equal(Fingerprint(semantics, a), Fingerprint(semantics, clone));
        Assert.NotEqual(Fingerprint(semantics, a), Fingerprint(semantics, b));
    }

    [Fact]
    public void DoubleSemanticsAreValueBased() =>
        AssertContract(DoubleValueSemantics.Instance, 1.5, -2.25, "double");

    [Fact]
    public void Int32SemanticsAreValueBased() =>
        AssertContract(Int32ValueSemantics.Instance, 7, -7, "int32");

    [Fact]
    public void DoubleArraySemanticsAreDeep()
    {
        var semantics = DoubleArrayValueSemantics.Instance;
        var a = new[] { 1.0, 2.0, 3.0 };
        AssertContract(semantics, a, new[] { 1.0, 2.0, 4.0 }, "double[]");

        var clone = semantics.Clone(a);
        Assert.NotSame(a, clone);
        clone[0] = 99.0;
        Assert.Equal(1.0, a[0]);

        Assert.False(semantics.ValueEquals(a, new[] { 1.0, 2.0 }));
        Assert.True(semantics.ValueEquals(a, new[] { 1.0, 2.0, 3.0 }));
    }

    [Fact]
    public void Int32ArraySemanticsAreDeep()
    {
        var semantics = Int32ArrayValueSemantics.Instance;
        var a = new[] { 1, 2, 3 };
        AssertContract(semantics, a, new[] { 1, 2, 4 }, "int32[]");

        var clone = semantics.Clone(a);
        Assert.NotSame(a, clone);
        clone[0] = 99;
        Assert.Equal(1, a[0]);

        Assert.False(semantics.ValueEquals(a, new[] { 1, 2 }));
        Assert.True(semantics.ValueEquals(a, new[] { 1, 2, 3 }));
    }

    [Fact]
    public void ArrayFingerprintIncludesLength()
    {
        var semantics = Int32ArrayValueSemantics.Instance;
        Assert.NotEqual(
            Fingerprint(semantics, new[] { 1, 2, 3 }),
            Fingerprint(semantics, new[] { 1, 2, 3, 0 }));
    }

    [Fact]
    public void DoubleEqualityUsesBitPattern()
    {
        var semantics = DoubleValueSemantics.Instance;
        Assert.False(semantics.ValueEquals(0.0, -0.0));
        Assert.True(semantics.ValueEquals(double.NaN, double.NaN));
    }

    [Theory]
    [InlineData(typeof(double))]
    [InlineData(typeof(int))]
    [InlineData(typeof(double[]))]
    [InlineData(typeof(int[]))]
    public void GenomeAndContractsCloseOverEveryOfficialValueDomain(Type valueType)
    {
        var genomeType = typeof(Genome<>).MakeGenericType(valueType);
        var shape = new GenomeShape(2, 8, 1, 3, 8);
        var genome = Activator.CreateInstance(genomeType, shape);

        Assert.NotNull(genome);
        Assert.Equal(shape, genomeType.GetProperty(nameof(Genome<int>.Shape))!.GetValue(genome));
        Assert.NotNull(typeof(IGeneticProblem<>).MakeGenericType(valueType));
        Assert.NotNull(typeof(IEvolutionEngine<>).MakeGenericType(valueType));
        Assert.NotNull(typeof(IGenomeEvaluator<>).MakeGenericType(valueType));
        Assert.NotNull(typeof(IFitnessEvaluator<>).MakeGenericType(valueType));
    }
}
