using Evo.Core.Functions;
using Evo.Core.Values;
using Xunit;

namespace Evo.Core.Tests;

public sealed class V4ExtensionContractTests
{
    private static readonly GenomeShape Shape = new(1, 8, 1, 2, 3);

    private static EvolutionTestProblem<double> Problem() => new(
        Shape,
        StandardDoubleFunctionSet.Instance,
        DoubleValueSemantics.Instance,
        DoubleParameterStrategy.Instance,
        () => new QuadraticFitness());

    private static (Genome<double>[] Population, FitnessResult[] Fitness) Population(EvolutionOperators<double> operators, int count)
    {
        var population = new Genome<double>[count];
        var fitness = new FitnessResult[count];
        for (var i = 0; i < count; i++)
        {
            var rng = RandomStream.Create(7, 0, (ulong)i, 1);
            population[i] = operators.CreateInitial(ref rng);
            fitness[i] = new FitnessResult(i % 3, true);
        }

        return (population, fitness);
    }

    [Fact]
    public void Best_selector_returns_lowest_index_on_tie_and_consumes_no_rng()
    {
        var problem = Problem();
        var operators = new EvolutionOperators<double>(problem);
        var (population, fitness) = Population(operators, 6);
        var context = new ParentSelectionContext<double>(population, fitness);
        var rng = RandomStream.Create(1, 0, 0, 1);
        var before = rng;

        var index = new BestParentSelector<double>().SelectParentIndex(context, ref rng);

        Assert.Equal(2, index);
        Assert.Equal(before, rng);
    }

    [Fact]
    public void Uniform_and_tournament_selectors_return_valid_indices()
    {
        var operators = new EvolutionOperators<double>(Problem());
        var (population, fitness) = Population(operators, 5);
        var context = new ParentSelectionContext<double>(population, fitness);
        var rng = RandomStream.Create(3, 0, 0, 1);

        for (var i = 0; i < 200; i++)
        {
            var uniform = new UniformParentSelector<double>().SelectParentIndex(context, ref rng);
            var tournament = new TournamentParentSelector<double>(3).SelectParentIndex(context, ref rng);
            Assert.InRange(uniform, 0, 4);
            Assert.InRange(tournament, 0, 4);
        }
    }

    [Fact]
    public void Context_and_selector_reject_invalid_arguments()
    {
        var operators = new EvolutionOperators<double>(Problem());
        var (population, fitness) = Population(operators, 3);

        Assert.Throws<ArgumentException>(() => new ParentSelectionContext<double>(population, fitness[..2]));
        Assert.Throws<ArgumentException>(() => new ParentSelectionContext<double>([], []));
        Assert.Throws<ArgumentOutOfRangeException>(() => new TournamentParentSelector<double>(0));
    }

    [Fact]
    public void Out_of_range_selector_index_is_rejected_before_offspring_is_built()
    {
        var operators = new EvolutionOperators<double>(Problem());
        var (population, fitness) = Population(operators, 4);
        var rng = RandomStream.Create(5, 0, 0, 1);

        Assert.Throws<InvalidOperationException>(() => operators.BreedWith(
            new BadSelector(), population, fitness, new EvolutionOptions(), ref rng));
    }

    [Fact]
    public void Standard_variation_matches_legacy_breed_stream_for_same_seeds()
    {
        var problem = Problem();
        var operators = new EvolutionOperators<double>(problem);
        var (population, fitness) = Population(operators, 8);
        var options = new EvolutionOptions { TournamentSize = 3, CrossoverRate = 0.5, Seed = 42 };
        var standard = new StandardVariationOperator<double>(problem);
        var context = new VariationContext<double>(population, fitness, options);

        for (ulong i = 0; i < 50; i++)
        {
            var legacyRng = RandomStream.Create(42, 1, i, 2);
            var contractRng = RandomStream.Create(42, 1, i, 2);
            var legacy = operators.Breed(population, fitness, options, ref legacyRng);
            var contract = standard.CreateOffspring(context, ref contractRng);

            Assert.Equal(
                legacy.ComputePhenotypeFingerprint(problem.Functions, problem.Values),
                contract.ComputePhenotypeFingerprint(problem.Functions, problem.Values));
            Assert.Equal(legacyRng, contractRng);
        }
    }

    [Fact]
    public void External_selector_and_variation_are_usable_through_public_contracts()
    {
        var problem = Problem();
        var operators = new EvolutionOperators<double>(problem);
        var (population, fitness) = Population(operators, 4);
        var options = new EvolutionOptions { TournamentSize = 2, CrossoverRate = 0.0, Seed = 42 };
        IVariationOperator<double> external = new StandardVariationOperator<double>(problem, new BestParentSelector<double>());
        var rng = RandomStream.Create(11, 0, 0, 1);

        var child = external.CreateOffspring(new VariationContext<double>(population, fitness, options), ref rng);

        Assert.NotNull(child);
    }

    private sealed class BadSelector : IParentSelector<double>
    {
        public int SelectParentIndex(ParentSelectionContext<double> context, ref RandomStream random) => context.Count;
    }
}
