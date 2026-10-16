using Evo.ExternalConsumer;
using Evo.Core.Functions;
using Evo.Core.Values;
using Xunit;

namespace Evo.Core.Tests;

public sealed class ExternalConsumerTests
{
    private static readonly GenomeShape Shape = new(1, 8, 1, 2, 3);

    [Fact]
    public void External_selector_and_operator_run_for_double()
    {
        AssertConsumerRuns(new EvolutionTestProblem<double>(
            Shape, StandardDoubleFunctionSet.Instance, DoubleValueSemantics.Instance, DoubleParameterStrategy.Instance,
            () => throw new NotSupportedException()));
    }

    [Fact]
    public void External_selector_and_operator_run_for_int()
    {
        AssertConsumerRuns(new EvolutionTestProblem<int>(
            Shape, StandardInt32FunctionSet.Instance, Int32ValueSemantics.Instance, IntParameterStrategy.Instance,
            () => throw new NotSupportedException()));
    }

    [Fact]
    public void External_selector_and_operator_run_for_double_array()
    {
        AssertConsumerRuns(new EvolutionTestProblem<double[]>(
            Shape, StandardDoubleArrayFunctionSet.Instance, DoubleArrayValueSemantics.Instance, new DoubleArrayParameterStrategy(2),
            () => throw new NotSupportedException()));
    }

    [Fact]
    public void External_selector_and_operator_run_for_int_array()
    {
        AssertConsumerRuns(new EvolutionTestProblem<int[]>(
            Shape, StandardInt32ArrayFunctionSet.Instance, Int32ArrayValueSemantics.Instance, new IntArrayParameterStrategy(2),
            () => throw new NotSupportedException()));
    }

    private static void AssertConsumerRuns<T>(IGeneticProblem<T> problem)
    {
        var operators = new EvolutionOperators<T>(problem);
        const int count = 6;
        var population = new Genome<T>[count];
        var fitness = new FitnessResult[count];
        for (var i = 0; i < count; i++)
        {
            var init = RandomStream.Create(11, 0, (ulong)i, 1);
            population[i] = operators.CreateInitial(ref init);
            fitness[i] = new FitnessResult(i % 3, true);
        }

        var options = new EvolutionOptions { TournamentSize = 3, CrossoverRate = 0.5, Seed = 42 };
        var context = new VariationContext<T>(population, fitness, options);
        var selector = new ConsumerParentSelector<T>();
        IVariationOperator<T> variation = new ConsumerVariationOperator<T>(problem, selector);
        var rng = RandomStream.Create(5, 0, 0, 1);

        for (var i = 0; i < 20; i++)
        {
            var index = selector.SelectParentIndex(new ParentSelectionContext<T>(population, fitness), ref rng);
            Assert.InRange(index, 0, count - 1);
            Assert.NotNull(variation.CreateOffspring(context, ref rng));
        }
    }
}
