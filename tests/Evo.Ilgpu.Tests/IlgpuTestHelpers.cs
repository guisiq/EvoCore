using Evo.Core;
using Evo.Core.Functions;
using Evo.Ilgpu;
using Xunit;

namespace Evo.Ilgpu.Tests;

internal static class IlgpuTestHelpers
{
    public static int FindFunctionId<T>(IFunctionSet<T> functions, string part)
    {
        for (var i = 0; i < functions.Count; i++)
        {
            if (functions.Get(i).Id.Contains(part, StringComparison.OrdinalIgnoreCase))
            {
                return i;
            }
        }

        throw new InvalidOperationException($"função '{part}' não encontrada");
    }

    public static void AssertDoubleClose(double expected, double actual, string? because = null)
    {
        if (double.IsNaN(expected) || double.IsNaN(actual))
        {
            Assert.True(
                double.IsNaN(expected) && double.IsNaN(actual),
                because ?? $"esperado {expected}, atual {actual}");
            return;
        }

        if (double.IsInfinity(expected) || double.IsInfinity(actual))
        {
            Assert.True(expected == actual, because ?? $"esperado {expected}, atual {actual}");
            return;
        }

        var diff = Math.Abs(expected - actual);
        Assert.True(
            diff <= 1e-12 || diff <= 1e-12 * Math.Abs(expected),
            because ?? $"esperado {expected}, atual {actual}");
    }

    public static void AssertDoubleArrayClose(double[] expected, double[] actual, string prefix)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (var i = 0; i < expected.Length; i++)
        {
            AssertDoubleClose(expected[i], actual[i], $"{prefix} lane {i}: esperado {expected[i]}, atual {actual[i]}");
        }
    }

    /// <summary>
    /// R9-2: comparação bit a bit (sem tolerância) usada para provar ausência de mutação entre
    /// chamadas sucessivas de snapshots já retidos de reference/compiled/device.
    /// </summary>
    public static void AssertDoubleArrayBitExact(double[] expected, double[] actual, string prefix)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (var i = 0; i < expected.Length; i++)
        {
            Assert.True(
                BitConverter.DoubleToInt64Bits(expected[i]) == BitConverter.DoubleToInt64Bits(actual[i]),
                $"{prefix} lane {i}: bits divergentes, esperado {expected[i]:R}, atual {actual[i]:R}");
        }
    }

    public static void AssertRealKernelExecution<T>(IlgpuGenomeEvaluator<T> evaluator, Genome<T> genome, IFunctionSet<T> functions)
    {
        Assert.True(evaluator.LastEvaluationUsedDevice);
        Assert.StartsWith("ILGPU", evaluator.Backend);
        Assert.NotEqual("nenhum", evaluator.DeviceName);
        var counts = evaluator.LastDeviceExecutionCounts;
        Assert.NotNull(counts);

        var active = genome.Analyze(functions).ActiveNodes.ToHashSet();
        var inputCount = genome.Shape.InputCount;
        for (var input = 0; input < inputCount; input++)
        {
            Assert.Equal(0, counts![input]);
        }

        for (var node = 0; node < genome.Shape.NodeCount; node++)
        {
            var expected = active.Contains(node) ? 1 : 0;
            Assert.Equal(expected, counts![inputCount + node]);
        }
    }
}
