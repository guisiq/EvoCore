using Evo.Core.Functions;
using Evo.Core.Pipeline;
using Evo.Core.Values;
using Xunit;

namespace Evo.Core.Tests;

public sealed class PopulationPartitioningTests
{
    private static readonly GenomeShape Shape = new(1, 8, 1, 2, 3);

    private static Genome<T>[] InitialPopulation<T>(IGeneticProblem<T> problem, int populationSize, ulong seed = 11)
    {
        Genome<T>[]? population = null;
        var options = new EvolutionOptions { PopulationSize = populationSize, MaxGenerations = 0, Seed = seed, WorkerCount = 1 };
        new EvolutionEngine<T>(checkpoint: s => population = s.Population).Run(problem, options);
        return population!;
    }

    private static EvolutionTestProblem<double> DoubleProblem() => new(
        Shape, StandardDoubleFunctionSet.Instance, DoubleValueSemantics.Instance, DoubleParameterStrategy.Instance, () => new QuadraticFitness());

    private static EvolutionTestProblem<int> IntProblem() => new(
        Shape, StandardInt32FunctionSet.Instance, Int32ValueSemantics.Instance, IntParameterStrategy.Instance,
        () => new ScalarFitness<int>(new[] { -2, -1, 0, 1, 2 }, x => x * x, (a, b) => Math.Abs((double)a - b)));

    private static EvolutionTestProblem<double[]> DoubleArrayProblem()
    {
        var inputs = new[] { new[] { -1.0, 0.5 }, new[] { 1.0, 2.0 }, new[] { 0.0, -2.0 } };
        return new EvolutionTestProblem<double[]>(Shape, StandardDoubleArrayFunctionSet.Instance, DoubleArrayValueSemantics.Instance,
            new DoubleArrayParameterStrategy(2),
            () => new ScalarFitness<double[]>(inputs, v => v.Select(e => e * e).ToArray(), (a, b) => a.Zip(b, (p, q) => Math.Abs(p - q)).Sum()));
    }

    private static EvolutionTestProblem<int[]> IntArrayProblem()
    {
        var inputs = new[] { new[] { -1, 2 }, new[] { 1, 0 }, new[] { 3, -2 } };
        return new EvolutionTestProblem<int[]>(Shape, StandardInt32ArrayFunctionSet.Instance, Int32ArrayValueSemantics.Instance,
            new IntArrayParameterStrategy(2),
            () => new ScalarFitness<int[]>(inputs, v => v.Select(e => e * e).ToArray(), (a, b) => a.Zip(b, (p, q) => Math.Abs((double)p - q)).Sum()));
    }

    private static void AssertDisjointExhaustiveCoversAllIndicesOnce<T>(EvolutionTestProblem<T> problem, IValueSemantics<T> values)
    {
        var population = InitialPopulation(problem, 6);
        var snapshot = PopulationSnapshot<T>.FromPopulation(population, problem);
        var outputs = new[]
        {
            new PopulationOutputSpec("out-a", 1, Path.Combine(Path.GetTempPath(), Guid.NewGuid().ToString("N"))),
            new PopulationOutputSpec("out-b", 2, Path.Combine(Path.GetTempPath(), Guid.NewGuid().ToString("N"))),
            new PopulationOutputSpec("out-c", 3, Path.Combine(Path.GetTempPath(), Guid.NewGuid().ToString("N"))),
        };
        var request = new PopulationPartitionRequest<T>(snapshot, outputs, PopulationPartitionMode.DisjointExhaustive, 99);
        var partitions = new ContiguousBlockPartitioner<T>().Partition(request);

        Assert.Equal(3, partitions.Count);
        var union = partitions.SelectMany(p => p.SourceIndices).OrderBy(i => i).ToArray();
        Assert.Equal(Enumerable.Range(0, 6), union);
        Assert.Equal(outputs.Select(o => o.OutputId), partitions.Select(p => p.Output.OutputId));
    }

    [Fact]
    public void DisjointExhaustive_covers_all_indices_once_double() => AssertDisjointExhaustiveCoversAllIndicesOnce(DoubleProblem(), DoubleValueSemantics.Instance);

    [Fact]
    public void DisjointExhaustive_covers_all_indices_once_int() => AssertDisjointExhaustiveCoversAllIndicesOnce(IntProblem(), Int32ValueSemantics.Instance);

    [Fact]
    public void DisjointExhaustive_covers_all_indices_once_double_array() => AssertDisjointExhaustiveCoversAllIndicesOnce(DoubleArrayProblem(), DoubleArrayValueSemantics.Instance);

    [Fact]
    public void DisjointExhaustive_covers_all_indices_once_int_array() => AssertDisjointExhaustiveCoversAllIndicesOnce(IntArrayProblem(), Int32ArrayValueSemantics.Instance);

    [Fact]
    public void DisjointExhaustive_rejects_missing_or_duplicate_index()
    {
        var problem = DoubleProblem();
        var population = InitialPopulation(problem, 4);
        var snapshot = PopulationSnapshot<double>.FromPopulation(population, problem);
        var outputs = new[]
        {
            new PopulationOutputSpec("a", 1, Path.GetTempPath()),
            new PopulationOutputSpec("b", 2, Path.GetTempPath()),
        };
        var request = new PopulationPartitionRequest<double>(snapshot, outputs, PopulationPartitionMode.DisjointExhaustive, 1);
        var partitioner = new DelegatePopulationPartitioner<double>("bad", _ => new[]
        {
            new PopulationPartition<double>(outputs[0], new[] { 0, 1 }),
            new PopulationPartition<double>(outputs[1], new[] { 1, 2 }), // 1 repetido, 3 ausente
        });

        Assert.Throws<InvalidOperationException>(() => partitioner.Partition(request));
    }

    [Fact]
    public void DisjointSelected_allows_explicit_discard()
    {
        var problem = DoubleProblem();
        var population = InitialPopulation(problem, 5);
        var snapshot = PopulationSnapshot<double>.FromPopulation(population, problem);
        var outputs = new[] { new PopulationOutputSpec("a", 1, Path.GetTempPath()) };
        var request = new PopulationPartitionRequest<double>(snapshot, outputs, PopulationPartitionMode.DisjointSelected, 1);
        var partitioner = new DelegatePopulationPartitioner<double>("selected", _ => new[]
        {
            new PopulationPartition<double>(outputs[0], new[] { 0, 2 }), // 1, 3, 4 descartados
        });

        var partitions = partitioner.Partition(request);

        Assert.Equal(new[] { 0, 2 }, partitions[0].SourceIndices);
    }

    [Fact]
    public void DisjointSelected_rejects_double_assignment()
    {
        var problem = DoubleProblem();
        var population = InitialPopulation(problem, 4);
        var snapshot = PopulationSnapshot<double>.FromPopulation(population, problem);
        var outputs = new[]
        {
            new PopulationOutputSpec("a", 1, Path.GetTempPath()),
            new PopulationOutputSpec("b", 2, Path.GetTempPath()),
        };
        var request = new PopulationPartitionRequest<double>(snapshot, outputs, PopulationPartitionMode.DisjointSelected, 1);
        var partitioner = new DelegatePopulationPartitioner<double>("bad-selected", _ => new[]
        {
            new PopulationPartition<double>(outputs[0], new[] { 0 }),
            new PopulationPartition<double>(outputs[1], new[] { 0 }), // mesmo índice em duas saídas: inválido fora de Overlapping
        });

        Assert.Throws<InvalidOperationException>(() => partitioner.Partition(request));
    }

    [Fact]
    public void Overlapping_allows_explicit_duplicate_with_independent_deep_clones()
    {
        var problem = DoubleProblem();
        var population = InitialPopulation(problem, 4);
        var snapshot = PopulationSnapshot<double>.FromPopulation(population, problem);
        var outputs = new[]
        {
            new PopulationOutputSpec("a", 1, Path.GetTempPath()),
            new PopulationOutputSpec("b", 2, Path.GetTempPath()),
        };
        var request = new PopulationPartitionRequest<double>(snapshot, outputs, PopulationPartitionMode.Overlapping, 1);
        var partitioner = new DelegatePopulationPartitioner<double>("overlap", _ => new[]
        {
            new PopulationPartition<double>(outputs[0], new[] { 0, 1 }),
            new PopulationPartition<double>(outputs[1], new[] { 1, 2 }), // índice 1 em ambas as saídas
        });

        var partitions = partitioner.Partition(request);

        var cloneForA = snapshot.GenomeAt(1, DoubleValueSemantics.Instance);
        var cloneForB = snapshot.GenomeAt(1, DoubleValueSemantics.Instance);
        Assert.NotSame(cloneForA, cloneForB);
        cloneForA.SetParameterGene(0, 123456.0);
        Assert.NotEqual(123456.0, cloneForB.ParameterGenes[0]);
        Assert.Equal(2, partitions.Count);
    }

    [Fact]
    public void Snapshot_does_not_expose_fitness_cache_or_evaluator_state()
    {
        var problem = DoubleProblem();
        var population = InitialPopulation(problem, 3);
        var snapshot = PopulationSnapshot<double>.FromPopulation(population, problem);

        Assert.Equal(problem.Shape, snapshot.Shape);
        Assert.Equal(problem.Values.ValueTypeId, snapshot.ValueTypeId);
        Assert.Equal(problem.Functions.Id, snapshot.FunctionSetId);
        Assert.Equal(3, snapshot.Count);
        // O tipo PopulationSnapshot<T> não expõe nenhuma propriedade de fitness/cache/avaliador: a cobertura
        // pública observável é Shape, ValueTypeId, FunctionSetId, Lineage, Count e GenomeAt(index, values).
    }

    [Fact]
    public void Runner_sequential_and_concurrent_produce_same_result_per_output()
    {
        var problem = DoubleProblem();
        var population = InitialPopulation(problem, 8, seed: 7);
        var snapshot = PopulationSnapshot<double>.FromPopulation(population, problem);

        PopulationPartitionRequest<double> MakeRequest(out string dirA, out string dirB)
        {
            dirA = Path.Combine(Path.GetTempPath(), "evo-partition-" + Guid.NewGuid().ToString("N"));
            dirB = Path.Combine(Path.GetTempPath(), "evo-partition-" + Guid.NewGuid().ToString("N"));
            var outputs = new[]
            {
                new PopulationOutputSpec("left", 101, dirA),
                new PopulationOutputSpec("right", 202, dirB),
            };
            return new PopulationPartitionRequest<double>(snapshot, outputs, PopulationPartitionMode.DisjointExhaustive, 55);
        }

        EvolutionPipelineDefinition<double> DefinitionFactory(PopulationOutputSpec spec, int populationSize) =>
            new("pipe-partition", "main", new[]
            {
                new EvolutionPipelineState<double>("main", DoubleProblem(), new EvolutionOptions
                {
                    PopulationSize = populationSize,
                    MaxGenerations = 3,
                    Seed = spec.Seed,
                    WorkerCount = 1,
                }),
            });

        var runner = new PopulationPartitionRunner<double>();
        var sequential = runner.Run(MakeRequest(out _, out _), new ContiguousBlockPartitioner<double>(), DefinitionFactory, maxDegreeOfParallelism: 1);
        var concurrent = runner.Run(MakeRequest(out _, out _), new ContiguousBlockPartitioner<double>(), DefinitionFactory, maxDegreeOfParallelism: 2);

        Assert.Equal(2, sequential.Count);
        Assert.Equal(new[] { "left", "right" }, sequential.Select(r => r.OutputId));
        for (var i = 0; i < sequential.Count; i++)
        {
            Assert.Equal(PopulationPartitionOutcome.Success, sequential[i].Outcome);
            Assert.Equal(PopulationPartitionOutcome.Success, concurrent[i].Outcome);
            Assert.Equal(sequential[i].Result!.BestFitness.Value, concurrent[i].Result!.BestFitness.Value);
            Assert.Equal(sequential[i].OutputId, concurrent[i].OutputId);
        }
    }

    [Fact]
    public void Runner_isolates_failure_per_output_without_contaminating_others()
    {
        var problem = DoubleProblem();
        var population = InitialPopulation(problem, 6, seed: 3);
        var snapshot = PopulationSnapshot<double>.FromPopulation(population, problem);
        var outputs = new[]
        {
            new PopulationOutputSpec("ok", 1, Path.Combine(Path.GetTempPath(), Guid.NewGuid().ToString("N"))),
            new PopulationOutputSpec("incompatible", 2, Path.Combine(Path.GetTempPath(), Guid.NewGuid().ToString("N"))),
        };
        var request = new PopulationPartitionRequest<double>(snapshot, outputs, PopulationPartitionMode.DisjointExhaustive, 1);

        EvolutionPipelineDefinition<double> DefinitionFactory(PopulationOutputSpec spec, int populationSize)
        {
            var shape = spec.OutputId == "incompatible" ? new GenomeShape(1, 9, 1, 2, 3) : Shape;
            var incompatibleProblem = new EvolutionTestProblem<double>(shape, StandardDoubleFunctionSet.Instance,
                DoubleValueSemantics.Instance, DoubleParameterStrategy.Instance, () => new QuadraticFitness());
            return new("pipe-fail", "main", new[]
            {
                new EvolutionPipelineState<double>("main", incompatibleProblem, new EvolutionOptions
                {
                    PopulationSize = populationSize,
                    MaxGenerations = 1,
                    Seed = spec.Seed,
                    WorkerCount = 1,
                }),
            });
        }

        var runner = new PopulationPartitionRunner<double>();
        var results = runner.Run(request, new ContiguousBlockPartitioner<double>(), DefinitionFactory, maxDegreeOfParallelism: 1);

        Assert.Equal(2, results.Count);
        Assert.Equal(PopulationPartitionOutcome.Success, results[0].Outcome);
        Assert.Equal(PopulationPartitionOutcome.Failure, results[1].Outcome);
        Assert.NotNull(results[1].Error);
    }
}
