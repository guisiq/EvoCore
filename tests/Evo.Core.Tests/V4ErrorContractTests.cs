using Evo.Core.Functions;
using Evo.Core.Values;
using Xunit;

namespace Evo.Core.Tests;

public sealed class V4ErrorContractTests
{
    private static readonly GenomeShape Shape = new(1, 8, 1, 2, 3);
    private static readonly GenomeShape OtherShape = new(1, 6, 1, 2, 3);

    private static EvolutionTestProblem<double> Problem(GenomeShape shape) => new(
        shape,
        StandardDoubleFunctionSet.Instance,
        DoubleValueSemantics.Instance,
        DoubleParameterStrategy.Instance,
        () => throw new NotSupportedException());

    [Fact]
    public void Context_rejects_empty_population_and_count_mismatch()
    {
        Assert.Throws<ArgumentException>(() => new ParentSelectionContext<double>(Array.Empty<Genome<double>>(), Array.Empty<FitnessResult>()));

        var operators = new EvolutionOperators<double>(Problem(Shape));
        var rng = RandomStream.Create(1, 0, 0, 1);
        var population = new[] { operators.CreateInitial(ref rng), operators.CreateInitial(ref rng) };
        Assert.Throws<ArgumentException>(() => new ParentSelectionContext<double>(population, new[] { new FitnessResult(0, true) }));
    }

    [Fact]
    public void Context_rejects_genomes_with_different_shapes_before_selection()
    {
        var rng = RandomStream.Create(2, 0, 0, 1);
        var first = new EvolutionOperators<double>(Problem(Shape)).CreateInitial(ref rng);
        var other = new EvolutionOperators<double>(Problem(OtherShape)).CreateInitial(ref rng);
        var fitness = new[] { new FitnessResult(0, true), new FitnessResult(1, true) };

        Assert.Throws<ArgumentException>(() => new ParentSelectionContext<double>(new[] { first, other }, fitness));
    }

    [Fact]
    public void Breed_with_out_of_range_selector_fails_before_building_offspring()
    {
        var operators = new EvolutionOperators<double>(Problem(Shape));
        var rng = RandomStream.Create(3, 0, 0, 1);
        var population = new[] { operators.CreateInitial(ref rng), operators.CreateInitial(ref rng) };
        var fitness = new[] { new FitnessResult(0, true), new FitnessResult(1, true) };
        var options = new EvolutionOptions { CrossoverRate = 0.0, Seed = 3 };

        Assert.Throws<InvalidOperationException>(() => operators.BreedWith(
            new OutOfRangeSelector(), population, fitness, options, ref rng));
    }

    [Fact]
    public void Breed_with_cancelled_token_throws_before_selection()
    {
        var operators = new EvolutionOperators<double>(Problem(Shape));
        var rng = RandomStream.Create(4, 0, 0, 1);
        var population = new[] { operators.CreateInitial(ref rng), operators.CreateInitial(ref rng) };
        var fitness = new[] { new FitnessResult(0, true), new FitnessResult(1, true) };
        var options = new EvolutionOptions { Seed = 4 };
        var before = rng;
        using var cts = new CancellationTokenSource();
        cts.Cancel();

        Assert.Throws<OperationCanceledException>(() => operators.BreedWith(
            new TournamentParentSelector<double>(2), population, fitness, options, ref rng, cts.Token));
        Assert.Equal(before, rng);
    }

    [Fact]
    public void Standard_variation_operator_honors_cancellation()
    {
        var problem = Problem(Shape);
        var operators = new EvolutionOperators<double>(problem);
        var rng = RandomStream.Create(5, 0, 0, 1);
        var population = new[] { operators.CreateInitial(ref rng), operators.CreateInitial(ref rng) };
        var fitness = new[] { new FitnessResult(0, true), new FitnessResult(1, true) };
        using var cts = new CancellationTokenSource();
        cts.Cancel();
        var context = new VariationContext<double>(population, fitness, new EvolutionOptions { Seed = 5 }, cts.Token);

        Assert.Throws<OperationCanceledException>(() => new StandardVariationOperator<double>(problem).CreateOffspring(context, ref rng));
    }

    private sealed class OutOfRangeSelector : IParentSelector<double>
    {
        public int SelectParentIndex(ParentSelectionContext<double> context, ref RandomStream random) => context.Count;
    }
}
