using Evo.Core;
using Evo.Core.Functions;
using Evo.Core.Values;
using Evo.Ilgpu;
using Xunit;

namespace Evo.Ilgpu.Tests;

public sealed class IlgpuWorkerCountTests
{
    private static int AddId(IFunctionSet<double> functions)
    {
        for (var i = 0; i < functions.Count; i++)
        {
            if (functions.Get(i).Id.Contains("add", StringComparison.OrdinalIgnoreCase))
            {
                return i;
            }
        }

        throw new InvalidOperationException("função add não encontrada");
    }

    [Theory]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(4)]
    public void Results_are_identical_across_isolated_workers(int workerCount)
    {
        var functions = StandardDoubleFunctionSet.Instance;
        var shape = new GenomeShape(InputCount: 2, NodeCount: 1, OutputCount: 1, MaxArity: 2, LevelsBack: 1);
        var genome = Genome<double>.FromGenes(shape, new[] { AddId(functions) }, new[] { 0, 1 },
            new[] { 0.0 }, new[] { 2 }, DoubleValueSemantics.Instance, functions);
        var inputs = new[] { 1.5, -2.25 };
        var expected = new double[1];
        new ReferenceGenomeEvaluator<double>(functions).Evaluate(genome, inputs, expected);

        var results = new double[workerCount][];
        Parallel.For(0, workerCount, w =>
        {
            using var evaluator = new IlgpuGenomeEvaluator<double>(functions);
            var output = new double[1];
            evaluator.Evaluate(genome, inputs, output);
            results[w] = output;
        });

        foreach (var result in results)
        {
            Assert.Equal(expected[0], result[0], 12);
        }
    }
}
