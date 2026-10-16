using Evo.Core;
using Xunit;

namespace Evo.Core.Tests;

public sealed class PartitionAllocatorTests
{
    [Fact]
    public void LargestRemainder_SumsExactlyAndBreaksTiesByIndex()
    {
        // Pesos 1/1/1 com total 3 -> cotas exatas 1,1,1 (sem resto).
        Assert.Equal(new[] { 1, 1, 1 }, PartitionAllocator.LargestRemainder(new[] { 1.0, 1.0, 1.0 }, 3));

        // Pesos 1/1/2 com total 3 -> cotas 0,75 / 0,75 / 1,5 -> piso 0,0,1; restam 2 posições para os
        // maiores restos (0,75 e 0,75 empatam; desempate pelo menor índice -> 0 e 1).
        var allocation = PartitionAllocator.LargestRemainder(new[] { 1.0, 1.0, 2.0 }, 3);
        Assert.Equal(new[] { 1, 1, 1 }, allocation);
        Assert.Equal(3, allocation.Sum());
    }

    [Fact]
    public void LargestRemainder_SmallTotalsAndFractionalWeights()
    {
        Assert.Equal(new[] { 1, 0 }, PartitionAllocator.LargestRemainder(new[] { 0.7, 0.3 }, 1));
        Assert.Equal(new[] { 2, 1, 1 }, PartitionAllocator.LargestRemainder(new[] { 1.0, 1.0, 1.0 }, 4));
        Assert.Equal(new[] { 0, 0 }, PartitionAllocator.LargestRemainder(new[] { 1.0, 1.0 }, 0));
    }

    [Theory]
    [InlineData(0.0)]
    [InlineData(-1.0)]
    [InlineData(double.NaN)]
    [InlineData(double.PositiveInfinity)]
    public void LargestRemainder_RejectsInvalidWeights(double weight)
    {
        Assert.Throws<ArgumentException>(() => PartitionAllocator.LargestRemainder(new[] { 1.0, weight }, 4));
    }

    [Fact]
    public void LargestRemainder_RejectsEmptyAndNegativeTotal()
    {
        Assert.Throws<ArgumentException>(() => PartitionAllocator.LargestRemainder(Array.Empty<double>(), 1));
        Assert.Throws<ArgumentOutOfRangeException>(() => PartitionAllocator.LargestRemainder(new[] { 1.0 }, -1));
    }
}
