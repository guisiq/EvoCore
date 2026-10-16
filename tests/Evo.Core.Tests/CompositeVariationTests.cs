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
    public void Composition_rejects_duplicate_instance_and_self_reference_cycle_before_generation()
    {
        var shared = new StubOperator();
        Assert.Throws<ArgumentException>(() => new CompositeVariationOperator<double>(
            CompositionMode.Sequential,
            new IVariationOperator<double>[] { shared, shared }));

        var inner = new CompositeVariationOperator<double>(
            CompositionMode.Sequential,
            new IVariationOperator<double>[] { new StubOperator() });
        Assert.Throws<ArgumentException>(() => new CompositeVariationOperator<double>(
            CompositionMode.Partitioned,
            new IVariationOperator<double>[] { inner, inner },
            new[] { 1.0, 1.0 }));
    }

    [Fact]
    public void Nested_composition_is_allowed_when_graph_has_no_repeated_instance()
    {
        var innerA = new CompositeVariationOperator<double>(
            CompositionMode.Sequential,
            new IVariationOperator<double>[] { new StubOperator(), new StubOperator() });
        var innerB = new CompositeVariationOperator<double>(
            CompositionMode.Partitioned,
            new IVariationOperator<double>[] { new StubOperator(), new StubOperator() },
            new[] { 1.0, 1.0 });
        var outer = new CompositeVariationOperator<double>(
            CompositionMode.Partitioned,
            new IVariationOperator<double>[] { innerA, innerB },
            new[] { 1.0, 1.0 });

        var context = new VariationContext<double>(
            new[] { SeedGenome() },
            new[] { FitnessResult.Valid(0) },
            Options(1));
        var rng = RandomStreamFor(1);
        Assert.NotNull(outer.CreatePositionOffspring(context, 0, 4, ref rng));
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

    private static EvolutionTestProblem<double> BatchProblem(int preferredBatchSize, BatchCounter counter) => new(
        Shape,
        StandardDoubleFunctionSet.Instance,
        DoubleValueSemantics.Instance,
        DoubleParameterStrategy.Instance,
        () => new BatchQuadraticFitness(preferredBatchSize, counter));

    [Fact]
    public void Composite_variation_is_equivalent_between_individual_and_batch_evaluation()
    {
        CompositeVariationOperator<double> MakeComposite(EvolutionTestProblem<double> p) => new(
            CompositionMode.Sequential,
            new IVariationOperator<double>[]
            {
                new CompositeVariationOperator<double>(
                    CompositionMode.Partitioned,
                    new IVariationOperator<double>[] { new StandardVariationOperator<double>(p), new StandardVariationOperator<double>(p) },
                    new[] { 1.0, 2.0 }),
                new StandardVariationOperator<double>(p),
            });

        var individualProblem = Problem();
        var individual = new EvolutionEngine<double>(MakeComposite(individualProblem)).Run(individualProblem, Options(1));

        var counter = new BatchCounter();
        var batchProblem = BatchProblem(preferredBatchSize: 4, counter);
        var batch = new EvolutionEngine<double>(MakeComposite(batchProblem)).Run(batchProblem, Options(1));

        Assert.Equal(individual.BestFitness, batch.BestFitness);
        Assert.True(counter.Calls > 0);
    }

    [Theory]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(4)]
    public void Nested_composition_is_worker_count_equivalent_with_small_batches_and_zero_weight_stage(int workers)
    {
        CompositeVariationOperator<double> MakeComposite(EvolutionTestProblem<double> p) => new(
            CompositionMode.Partitioned,
            new IVariationOperator<double>[]
            {
                new CompositeVariationOperator<double>(
                    CompositionMode.Sequential,
                    new IVariationOperator<double>[] { new StandardVariationOperator<double>(p), new StandardVariationOperator<double>(p) }),
                new StandardVariationOperator<double>(p),
                new StandardVariationOperator<double>(p),
            },
            new[] { 1.0, 0.0001, 2.0 });

        var baseline = new EvolutionEngine<double>(MakeComposite(Problem())).Run(Problem(), Options(1) with { PopulationSize = 4 });
        var problem = Problem();
        var result = new EvolutionEngine<double>(MakeComposite(problem)).Run(problem, Options(workers) with { PopulationSize = 4 });

        Assert.Equal(baseline.BestFitness, result.BestFitness);
    }

    [Fact]
    public void Partitioned_zero_weight_stage_never_receives_a_position()
    {
        var composite = new CompositeVariationOperator<double>(
            CompositionMode.Partitioned,
            new IVariationOperator<double>[] { new StubOperator(), new StubOperator(), new StubOperator() },
            new[] { 1.0, 0.0001, 1.0 });

        var branches = Enumerable.Range(0, 7).Select(p => composite.BranchFor(p, 7)).ToArray();
        Assert.DoesNotContain(1, branches);
    }

    [Fact]
    public void Partitioned_composition_is_worker_count_equivalent_with_batch_smaller_than_stage_count()
    {
        var composite = new CompositeVariationOperator<double>(
            CompositionMode.Partitioned,
            new IVariationOperator<double>[]
            {
                new StandardVariationOperator<double>(Problem()),
                new StandardVariationOperator<double>(Problem()),
                new StandardVariationOperator<double>(Problem()),
            },
            new[] { 1.0, 1.0, 1.0 });

        // EliteCount=1, PopulationSize=2: apenas 1 descendente por geração, menos que os 3 estágios.
        var options = Options(1) with { PopulationSize = 2, EliteCount = 1 };
        var one = new EvolutionEngine<double>(composite).Run(Problem(), options);
        var two = new EvolutionEngine<double>(composite).Run(Problem(), options with { WorkerCount = 2 });

        Assert.Equal(one.BestFitness, two.BestFitness);
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
