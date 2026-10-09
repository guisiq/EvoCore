using Evo.Core.Compiled;
using Evo.Core.Functions;
using Evo.Core.Values;
using Xunit;

namespace Evo.Core.Tests;

public sealed class CompiledSimdTests
{
    private static readonly GenomeShape Shape = new(InputCount: 3, NodeCount: 12, OutputCount: 2, MaxArity: 3, LevelsBack: 12);

    private static Genome<double> RandomDoubleGenome(ulong seed, IFunctionSet<double> functions)
    {
        var rng = RandomStream.Create(seed, 0, 0, 0);
        var fg = new int[Shape.NodeCount];
        var cg = new int[Shape.NodeCount * Shape.MaxArity];
        var pg = new double[Shape.NodeCount];
        for (var n = 0; n < Shape.NodeCount; n++)
        {
            fg[n] = rng.NextInt(0, functions.Count);
            for (var a = 0; a < Shape.MaxArity; a++)
            {
                cg[n * Shape.MaxArity + a] = rng.NextInt(0, Shape.InputCount + n);
            }

            pg[n] = rng.NextDouble() * 4 - 2;
        }

        var og = new[] { rng.NextInt(0, Shape.InputCount + Shape.NodeCount), rng.NextInt(0, Shape.InputCount + Shape.NodeCount) };
        return Genome<double>.FromGenes(Shape, fg, cg, pg, og, DoubleValueSemantics.Instance, functions);
    }

    private static Genome<int> RandomInt32Genome(ulong seed, IFunctionSet<int> functions)
    {
        var rng = RandomStream.Create(seed, 0, 0, 0);
        var fg = new int[Shape.NodeCount];
        var cg = new int[Shape.NodeCount * Shape.MaxArity];
        var pg = new int[Shape.NodeCount];
        for (var n = 0; n < Shape.NodeCount; n++)
        {
            fg[n] = rng.NextInt(0, functions.Count);
            for (var a = 0; a < Shape.MaxArity; a++)
            {
                cg[n * Shape.MaxArity + a] = rng.NextInt(0, Shape.InputCount + n);
            }

            pg[n] = rng.NextInt(-20, 20);
        }

        var og = new[] { rng.NextInt(0, Shape.InputCount + Shape.NodeCount), rng.NextInt(0, Shape.InputCount + Shape.NodeCount) };
        return Genome<int>.FromGenes(Shape, fg, cg, pg, og, Int32ValueSemantics.Instance, functions);
    }

    private static double[] RandomDoubleCases(ulong seed, int caseCount)
    {
        var rng = RandomStream.Create(seed, 1, 0, 0);
        var data = new double[caseCount * Shape.InputCount];
        for (var i = 0; i < data.Length; i++)
        {
            data[i] = rng.NextDouble() * 6 - 3;
        }

        return data;
    }

    private static int[] RandomIntCases(ulong seed, int caseCount)
    {
        var rng = RandomStream.Create(seed, 1, 0, 0);
        var data = new int[caseCount * Shape.InputCount];
        for (var i = 0; i < data.Length; i++)
        {
            data[i] = rng.NextInt(-50, 50);
        }

        return data;
    }

    [Fact]
    public void CompiledDouble_matches_reference_bitwise_on_random_genomes()
    {
        var functions = StandardDoubleFunctionSet.Instance;
        var reference = new ReferenceGenomeEvaluator<double>(functions);
        var compiled = new CompiledGenomeEvaluator<double>(functions, DoubleValueSemantics.Instance);
        var inputs = RandomDoubleCases(7, 1);
        for (var seed = 0UL; seed < 40; seed++)
        {
            var genome = RandomDoubleGenome(seed, functions);
            var expected = new double[Shape.OutputCount];
            var actual = new double[Shape.OutputCount];
            reference.Evaluate(genome, inputs.AsSpan(0, Shape.InputCount), expected);
            compiled.Evaluate(genome, inputs.AsSpan(0, Shape.InputCount), actual);
            for (var o = 0; o < expected.Length; o++)
            {
                Assert.Equal(BitConverter.DoubleToInt64Bits(expected[o]), BitConverter.DoubleToInt64Bits(actual[o]));
            }
        }
    }

    [Fact]
    public void CompiledInt32_matches_reference_on_random_genomes()
    {
        var functions = StandardInt32FunctionSet.Instance;
        var reference = new ReferenceGenomeEvaluator<int>(functions);
        var compiled = new CompiledGenomeEvaluator<int>(functions, Int32ValueSemantics.Instance);
        var inputs = RandomIntCases(11, 1);
        for (var seed = 0UL; seed < 40; seed++)
        {
            var genome = RandomInt32Genome(seed, functions);
            var expected = new int[Shape.OutputCount];
            var actual = new int[Shape.OutputCount];
            reference.Evaluate(genome, inputs.AsSpan(0, Shape.InputCount), expected);
            compiled.Evaluate(genome, inputs.AsSpan(0, Shape.InputCount), actual);
            Assert.Equal(expected, actual);
        }
    }

    [Fact]
    public void CompiledCache_hits_on_same_genome_and_stays_bounded()
    {
        var functions = StandardDoubleFunctionSet.Instance;
        var compiled = new CompiledGenomeEvaluator<double>(functions, DoubleValueSemantics.Instance, cacheCapacity: 4);
        var inputs = RandomDoubleCases(3, 1).AsSpan(0, Shape.InputCount);
        var outputs = new double[Shape.OutputCount];
        var genome = RandomDoubleGenome(5, functions);
        compiled.Evaluate(genome, inputs, outputs);
        compiled.Evaluate(genome, inputs, outputs);
        Assert.Equal(1, compiled.CacheMisses);
        Assert.Equal(1, compiled.CacheHits);
        for (var seed = 100UL; seed < 120; seed++)
        {
            compiled.Evaluate(RandomDoubleGenome(seed, functions), inputs, outputs);
        }

        Assert.True(compiled.CacheCount <= 4);
    }

    [Fact]
    public void SimdDouble_matches_scalar_for_block_and_tail_case_counts()
    {
        var functions = StandardDoubleFunctionSet.Instance;
        var reference = new ReferenceGenomeEvaluator<double>(functions);
        var width = System.Numerics.Vector<double>.Count;
        var counts = new[] { 1, width - 1, width, width + 1, 2 * width, 2 * width + 3 }.Where(c => c > 0).Distinct();
        foreach (var caseCount in counts)
        {
            var simd = new SimdGenomeEvaluator<double>(functions, DoubleValueSemantics.Instance);
            var inputs = RandomDoubleCases(13, caseCount);
            for (var seed = 0UL; seed < 10; seed++)
            {
                var genome = RandomDoubleGenome(seed, functions);
                var expected = new double[caseCount * Shape.OutputCount];
                var actual = new double[caseCount * Shape.OutputCount];
                for (var c = 0; c < caseCount; c++)
                {
                    reference.Evaluate(genome, inputs.AsSpan(c * Shape.InputCount, Shape.InputCount), expected.AsSpan(c * Shape.OutputCount, Shape.OutputCount));
                }

                var report = simd.Evaluate(genome, inputs, caseCount, actual);
                for (var i = 0; i < expected.Length; i++)
                {
                    var tol = 1e-12 * Math.Max(1, Math.Abs(expected[i]));
                    Assert.True(Math.Abs(expected[i] - actual[i]) <= tol || (double.IsNaN(expected[i]) && double.IsNaN(actual[i])), $"case {caseCount} seed {seed} index {i}");
                }

                Assert.Equal(caseCount, report.CaseCount);
                if (report.Backend == EvaluationBackend.Simd)
                {
                    Assert.Equal(caseCount / width, report.VectorBlocks);
                    Assert.Equal(caseCount % width, report.TailCases);
                }
                else
                {
                    Assert.Equal(0, report.VectorBlocks);
                    Assert.Equal(caseCount, report.TailCases);
                }
            }
        }
    }

    [Fact]
    public void SimdInt32_matches_scalar_including_abs_of_int_min()
    {
        var functions = StandardInt32FunctionSet.Instance;
        var reference = new ReferenceGenomeEvaluator<int>(functions);
        var simd = new SimdGenomeEvaluator<int>(functions, Int32ValueSemantics.Instance);
        var width = System.Numerics.Vector<int>.Count;
        var caseCount = 2 * width + 3;
        var inputs = RandomIntCases(17, caseCount);
        inputs[0] = int.MinValue;
        for (var seed = 0UL; seed < 10; seed++)
        {
            var genome = RandomInt32Genome(seed, functions);
            var expected = new int[caseCount * Shape.OutputCount];
            var actual = new int[caseCount * Shape.OutputCount];
            for (var c = 0; c < caseCount; c++)
            {
                reference.Evaluate(genome, inputs.AsSpan(c * Shape.InputCount, Shape.InputCount), expected.AsSpan(c * Shape.OutputCount, Shape.OutputCount));
            }

            simd.Evaluate(genome, inputs, caseCount, actual);
            Assert.Equal(expected, actual);
        }
    }

    [Fact]
    public void SimdFallback_is_explicit_when_function_has_no_kernel()
    {
        var functions = StandardDoubleFunctionSet.Instance;
        var simd = new SimdGenomeEvaluator<double>(functions, DoubleValueSemantics.Instance);
        var width = System.Numerics.Vector<double>.Count;
        var caseCount = 2 * width;
        var inputs = RandomDoubleCases(19, caseCount);
        var found = false;
        for (var seed = 0UL; seed < 200 && !found; seed++)
        {
            var genome = RandomDoubleGenome(seed, functions);
            var output = new double[caseCount * Shape.OutputCount];
            var report = simd.Evaluate(genome, inputs, caseCount, output);
            if (report.Backend == EvaluationBackend.CompiledScalar)
            {
                found = true;
                Assert.False(string.IsNullOrEmpty(report.FallbackReason));
                Assert.Equal(0, report.VectorBlocks);
            }
        }

        Assert.True(found, "Expected at least one genome with a non-kernel active function (ProtectedDivide/ProtectedLog/etc.).");
    }
}
