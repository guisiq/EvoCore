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

    [Fact]
    public void Cancelled_token_stops_selection_before_selector_is_called()
    {
        var problem = Problem();
        var operators = new EvolutionOperators<double>(problem);
        var (population, fitness) = Population(operators, 4);
        using var cts = new CancellationTokenSource();
        cts.Cancel();
        var selector = new CountingSelector();
        var rng = RandomStream.Create(9, 0, 0, 1);

        Assert.Throws<OperationCanceledException>(() => operators.BreedWith(
            selector, population, fitness, new EvolutionOptions(), ref rng, cts.Token));
        Assert.Equal(0, selector.Calls);

        var context = new VariationContext<double>(population, fitness, new EvolutionOptions(), cts.Token);
        var variation = new StandardVariationOperator<double>(problem, selector);
        Assert.Throws<OperationCanceledException>(() => variation.CreateOffspring(context, ref rng));
        Assert.Equal(0, selector.Calls);
    }

    [Fact]
    public void Shape_null_and_empty_population_errors_are_explicit()
    {
        var operators = new EvolutionOperators<double>(Problem());
        var (population, fitness) = Population(operators, 3);

        var otherProblem = new EvolutionTestProblem<double>(
            new GenomeShape(1, 4, 1, 2, 3),
            StandardDoubleFunctionSet.Instance,
            DoubleValueSemantics.Instance,
            DoubleParameterStrategy.Instance,
            () => new QuadraticFitness());
        var otherRng = RandomStream.Create(7, 0, 99, 1);
        var otherGenome = new EvolutionOperators<double>(otherProblem).CreateInitial(ref otherRng);

        var mixed = (Genome<double>[])population.Clone();
        mixed[1] = otherGenome;
        var withNull = (Genome<double>[])population.Clone();
        withNull[2] = null!;

        Assert.Throws<ArgumentException>(() => new ParentSelectionContext<double>(mixed, fitness));
        Assert.Throws<ArgumentException>(() => new ParentSelectionContext<double>(withNull, fitness));
        Assert.Throws<ArgumentException>(() => new VariationContext<double>(withNull, fitness, new EvolutionOptions()));
        Assert.Throws<ArgumentException>(() => new VariationContext<double>([], [], new EvolutionOptions()));
    }

    [Fact]
    public void Negative_selector_index_is_rejected_before_offspring_is_built()
    {
        var operators = new EvolutionOperators<double>(Problem());
        var (population, fitness) = Population(operators, 4);
        var rng = RandomStream.Create(5, 0, 0, 1);

        Assert.Throws<InvalidOperationException>(() => operators.BreedWith(
            new NegativeSelector(), population, fitness, new EvolutionOptions(), ref rng));
    }

    [Fact]
    public void Standard_variation_returns_deep_copies_of_array_genes()
    {
        var problem = Problem();
        var operators = new EvolutionOperators<double>(problem);
        var (population, fitness) = Population(operators, 3);
        var options = new EvolutionOptions { CrossoverRate = 0.0, Seed = 42 };
        var variation = new StandardVariationOperator<double>(problem, new BestParentSelector<double>());
        var context = new VariationContext<double>(population, fitness, options);
        var parentBefore = population[2].ComputePhenotypeFingerprint(problem.Functions, problem.Values);
        var rng = RandomStream.Create(13, 0, 0, 1);

        var child = variation.CreateOffspring(context, ref rng);
        var childFunctions = child.FunctionGenes.ToArray();
        childFunctions[0] = -1;

        Assert.NotSame(population[2], child);
        Assert.Equal(parentBefore, population[2].ComputePhenotypeFingerprint(problem.Functions, problem.Values));
        Assert.NotEqual(-1, child.FunctionGenes[0]);
    }

    private sealed class CountingSelector : IParentSelector<double>
    {
        public int Calls { get; private set; }

        public int SelectParentIndex(ParentSelectionContext<double> context, ref RandomStream random)
        {
            Calls++;
            return 0;
        }
    }

    private sealed class NegativeSelector : IParentSelector<double>
    {
        public int SelectParentIndex(ParentSelectionContext<double> context, ref RandomStream random) => -1;
    }

    private sealed class BadSelector : IParentSelector<double>
    {
        public int SelectParentIndex(ParentSelectionContext<double> context, ref RandomStream random) => context.Count;
    }
}
