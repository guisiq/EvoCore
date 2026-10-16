using Evo.Core.Functions;
using Evo.Core.Values;
using Xunit;

namespace Evo.Core.Tests;

public sealed class CompositeVariationTests
{
    private static readonly GenomeShape Shape = new(1, 8, 1, 2, 3);

    private static EvolutionTestProblem<double> Problem() => new(
        Shape,
        StandardDoubleFunctionSet.Instance,
        DoubleValueSemantics.Instance,
        DoubleParameterStrategy.Instance,
        () => new QuadraticFitness());

    private static EvolutionOptions Options(int workers) => new()
    {
        PopulationSize = 12,
        MaxGenerations = 3,
        EliteCount = 1,
        TournamentSize = 3,
        Seed = 7,
        WorkerCount = workers,
    };

    [Fact]
    public void Partitioned_allocation_follows_largest_remainder_with_index_tie_break()
    {
        var composite = new CompositeVariationOperator<double>(
            CompositionMode.Partitioned,
            new IVariationOperator<double>[] { new StubOperator(), new StubOperator(), new StubOperator() },
            new[] { 1.0, 1.0, 1.0 });

        // 10 posições com pesos iguais: 3,3,3 + restos iguais -> primeiros índices recebem a sobra.
        var branches = Enumerable.Range(0, 10).Select(p => composite.BranchFor(p, 10)).ToArray();
        Assert.Equal(new[] { 0, 0, 0, 0, 1, 1, 1, 2, 2, 2 }, branches);
    }

    [Fact]
    public void Partitioned_rejects_missing_or_invalid_weights_before_generation()
    {
        var stages = new IVariationOperator<double>[] { new StubOperator(), new StubOperator() };
        Assert.Throws<ArgumentException>(() => new CompositeVariationOperator<double>(CompositionMode.Partitioned, stages));
        Assert.Throws<ArgumentException>(() => new CompositeVariationOperator<double>(CompositionMode.Partitioned, stages, new[] { 1.0, 0.0 }));
        Assert.Throws<ArgumentException>(() => new CompositeVariationOperator<double>(CompositionMode.Sequential, stages, new[] { 1.0, 1.0 }));
        Assert.Throws<ArgumentException>(() => new CompositeVariationOperator<double>(CompositionMode.Sequential, Array.Empty<IVariationOperator<double>>()));
    }

    [Fact]
    public void Engine_with_composite_variation_is_worker_count_equivalent()
    {
        var composite = new CompositeVariationOperator<double>(
            CompositionMode.Partitioned,
            new IVariationOperator<double>[] { new StandardVariationOperator<double>(Problem()), new StandardVariationOperator<double>(Problem()) },
            new[] { 2.0, 1.0 });

        var one = new EvolutionEngine<double>(composite).Run(Problem(), Options(1));
        var two = new EvolutionEngine<double>(composite).Run(Problem(), Options(2));

        Assert.Equal(one.BestFitness, two.BestFitness);
    }

    [Fact]
    public void Sequential_composition_applies_stages_in_order_on_single_offspring()
    {
        var log = new List<int>();
        var composite = new CompositeVariationOperator<double>(
            CompositionMode.Sequential,
            new IVariationOperator<double>[] { new TagOperator(log, 1), new TagOperator(log, 2) });

        var context = new VariationContext<double>(
            new[] { SeedGenome() },
            new[] { FitnessResult.Valid(0) },
            Options(1));

        var rng = RandomStreamFor(1);
        composite.CreateOffspring(context, ref rng);
        Assert.Equal(new[] { 1, 2 }, log);
    }

    private static RandomStream RandomStreamFor(ulong index) => RandomStream.Create(7, 1, index, 2);

    private static Genome<double> SeedGenome()
    {
        var rng = RandomStreamFor(0);
        return new EvolutionOperators<double>(Problem()).CreateInitial(ref rng);
    }

    private sealed class StubOperator : IVariationOperator<double>
    {
        public Genome<double> CreateOffspring(VariationContext<double> context, ref RandomStream random) => context.Population[0];
    }

    private sealed class TagOperator : IVariationOperator<double>
    {
        private readonly List<int> _log;
        private readonly int _tag;

        public TagOperator(List<int> log, int tag)
        {
            _log = log;
            _tag = tag;
        }

        public Genome<double> CreateOffspring(VariationContext<double> context, ref RandomStream random)
        {
            _log.Add(_tag);
            return context.Population[0];
        }
    }
}
