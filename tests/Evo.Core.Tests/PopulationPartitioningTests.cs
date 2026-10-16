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

    private static void AssertDisjointSelectedAllowsExplicitDiscard<T>(EvolutionTestProblem<T> problem)
    {
        var population = InitialPopulation(problem, 5);
        var snapshot = PopulationSnapshot<T>.FromPopulation(population, problem);
        var outputs = new[] { new PopulationOutputSpec("a", 1, Path.GetTempPath()) };
        var request = new PopulationPartitionRequest<T>(snapshot, outputs, PopulationPartitionMode.DisjointSelected, 1);
        var partitioner = new DelegatePopulationPartitioner<T>("selected", _ => new[]
        {
            new PopulationPartition<T>(outputs[0], new[] { 0, 2 }), // 1, 3, 4 descartados
        });

        var partitions = partitioner.Partition(request);

        Assert.Equal(2, partitions[0].SourceIndices.Count);
        Assert.Equal(new[] { 0, 2 }, partitions[0].SourceIndices);
    }

    [Fact]
    public void DisjointSelected_allows_explicit_discard_double() => AssertDisjointSelectedAllowsExplicitDiscard(DoubleProblem());

    [Fact]
    public void DisjointSelected_allows_explicit_discard_int() => AssertDisjointSelectedAllowsExplicitDiscard(IntProblem());

    [Fact]
    public void DisjointSelected_allows_explicit_discard_double_array() => AssertDisjointSelectedAllowsExplicitDiscard(DoubleArrayProblem());

    [Fact]
    public void DisjointSelected_allows_explicit_discard_int_array() => AssertDisjointSelectedAllowsExplicitDiscard(IntArrayProblem());

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

    private static void AssertOverlappingAllowsExplicitDuplicateWithIndependentDeepClones<T>(
        EvolutionTestProblem<T> problem, IValueSemantics<T> values, T marker, Func<T, T, bool> equal)
    {
        var population = InitialPopulation(problem, 4);
        var snapshot = PopulationSnapshot<T>.FromPopulation(population, problem);
        var outputs = new[]
        {
            new PopulationOutputSpec("a", 1, Path.GetTempPath()),
            new PopulationOutputSpec("b", 2, Path.GetTempPath()),
        };
        var request = new PopulationPartitionRequest<T>(snapshot, outputs, PopulationPartitionMode.Overlapping, 1);
        var partitioner = new DelegatePopulationPartitioner<T>("overlap", _ => new[]
        {
            new PopulationPartition<T>(outputs[0], new[] { 0, 1 }),
            new PopulationPartition<T>(outputs[1], new[] { 1, 2 }), // índice 1 em ambas as saídas
        });

        var partitions = partitioner.Partition(request);

        Assert.Equal(2, partitions.Count);
        Assert.Equal(new[] { 0, 1 }, partitions[0].SourceIndices);
        Assert.Equal(new[] { 1, 2 }, partitions[1].SourceIndices);

        var cloneForA = snapshot.GenomeAt(1, values);
        var cloneForB = snapshot.GenomeAt(1, values);
        Assert.NotSame(cloneForA, cloneForB);
        cloneForA.SetParameterGene(0, marker);
        Assert.False(equal(marker, cloneForB.ParameterGenes[0]));
    }

    [Fact]
    public void Overlapping_allows_explicit_duplicate_with_independent_deep_clones_double() =>
        AssertOverlappingAllowsExplicitDuplicateWithIndependentDeepClones(DoubleProblem(), DoubleValueSemantics.Instance, 123456.0, (a, b) => a.Equals(b));

    [Fact]
    public void Overlapping_allows_explicit_duplicate_with_independent_deep_clones_int() =>
        AssertOverlappingAllowsExplicitDuplicateWithIndependentDeepClones(IntProblem(), Int32ValueSemantics.Instance, 123456, (a, b) => a.Equals(b));

    [Fact]
    public void Overlapping_allows_explicit_duplicate_with_independent_deep_clones_double_array() =>
        AssertOverlappingAllowsExplicitDuplicateWithIndependentDeepClones(
            DoubleArrayProblem(), DoubleArrayValueSemantics.Instance, new[] { 123456.0, 123456.0 }, (a, b) => a.SequenceEqual(b));

    [Fact]
    public void Overlapping_allows_explicit_duplicate_with_independent_deep_clones_int_array() =>
        AssertOverlappingAllowsExplicitDuplicateWithIndependentDeepClones(
            IntArrayProblem(), Int32ArrayValueSemantics.Instance, new[] { 123456, 123456 }, (a, b) => a.SequenceEqual(b));

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

        PopulationPartitionRequest<double> MakeRequest()
        {
            var dirA = Path.Combine(Path.GetTempPath(), "evo-partition-" + Guid.NewGuid().ToString("N"));
            var dirB = Path.Combine(Path.GetTempPath(), "evo-partition-" + Guid.NewGuid().ToString("N"));
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
        var baseline = runner.Run(MakeRequest(), new ContiguousBlockPartitioner<double>(), DefinitionFactory, maxDegreeOfParallelism: 1);

        foreach (var workers in new[] { 1, 2, 4 })
        {
            var result = runner.Run(MakeRequest(), new ContiguousBlockPartitioner<double>(), DefinitionFactory, maxDegreeOfParallelism: workers);
            Assert.Equal(2, result.Count);
            Assert.Equal(new[] { "left", "right" }, result.Select(r => r.OutputId));
            for (var i = 0; i < baseline.Count; i++)
            {
                Assert.Equal(PopulationPartitionOutcome.Success, result[i].Outcome);
                Assert.Equal(baseline[i].OutputId, result[i].OutputId);
                Assert.Equal(baseline[i].Result!.BestFitness.Value, result[i].Result!.BestFitness.Value);
            }
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

        // Atomicidade: a falha isolada na saída "incompatible" não altera os artefatos nem o resultado da saída "ok".
        var okStore = new FileEvolutionEvidenceStore(outputs[0].EvidenceDirectory);
        var manifest = okStore.ReadManifest();
        Assert.Equal("ok", manifest.RunId);
        Assert.False(Directory.Exists(outputs[1].EvidenceDirectory) && File.Exists(Path.Combine(outputs[1].EvidenceDirectory, "manifest.json")));
    }

    [Fact]
    public void Runner_cooperative_cancellation_yields_cancelled_outcome_in_declared_order()
    {
        var problem = DoubleProblem();
        var population = InitialPopulation(problem, 6, seed: 5);
        var snapshot = PopulationSnapshot<double>.FromPopulation(population, problem);
        var outputs = new[]
        {
            new PopulationOutputSpec("first", 1, Path.Combine(Path.GetTempPath(), Guid.NewGuid().ToString("N"))),
            new PopulationOutputSpec("second", 2, Path.Combine(Path.GetTempPath(), Guid.NewGuid().ToString("N"))),
            new PopulationOutputSpec("third", 3, Path.Combine(Path.GetTempPath(), Guid.NewGuid().ToString("N"))),
        };
        var request = new PopulationPartitionRequest<double>(snapshot, outputs, PopulationPartitionMode.DisjointExhaustive, 1);

        EvolutionPipelineDefinition<double> DefinitionFactory(PopulationOutputSpec spec, int populationSize) =>
            new("pipe-cancel", "main", new[]
            {
                new EvolutionPipelineState<double>("main", DoubleProblem(), new EvolutionOptions
                {
                    PopulationSize = populationSize,
                    MaxGenerations = 2,
                    Seed = spec.Seed,
                    WorkerCount = 1,
                }),
            });

        using var cts = new CancellationTokenSource();
        cts.Cancel();
        var runner = new PopulationPartitionRunner<double>();
        var results = runner.Run(request, new ContiguousBlockPartitioner<double>(), DefinitionFactory, maxDegreeOfParallelism: 1, cts.Token);

        Assert.Equal(new[] { "first", "second", "third" }, results.Select(r => r.OutputId));
        Assert.All(results, r => Assert.Equal(PopulationPartitionOutcome.Cancelled, r.Outcome));
    }

    [Fact]
    public void Runner_cancellation_of_one_output_does_not_alter_artifacts_of_a_completed_output()
    {
        var problem = DoubleProblem();
        var population = InitialPopulation(problem, 6, seed: 9);
        var snapshot = PopulationSnapshot<double>.FromPopulation(population, problem);
        var outputs = new[]
        {
            new PopulationOutputSpec("ok", 1, Path.Combine(Path.GetTempPath(), Guid.NewGuid().ToString("N"))),
            new PopulationOutputSpec("to-cancel", 2, Path.Combine(Path.GetTempPath(), Guid.NewGuid().ToString("N"))),
        };
        var request = new PopulationPartitionRequest<double>(snapshot, outputs, PopulationPartitionMode.DisjointExhaustive, 1);

        using var cts = new CancellationTokenSource();

        EvolutionPipelineDefinition<double> DefinitionFactory(PopulationOutputSpec spec, int populationSize)
        {
            if (spec.OutputId == "to-cancel")
            {
                cts.Cancel(); // cancela somente depois que a saída "ok" já concluiu, em execução sequencial
            }

            return new("pipe-cancel-atomic", "main", new[]
            {
                new EvolutionPipelineState<double>("main", DoubleProblem(), new EvolutionOptions
                {
                    PopulationSize = populationSize,
                    MaxGenerations = 2,
                    Seed = spec.Seed,
                    WorkerCount = 1,
                }),
            });
        }

        var runner = new PopulationPartitionRunner<double>();
        var results = runner.Run(request, new ContiguousBlockPartitioner<double>(), DefinitionFactory, maxDegreeOfParallelism: 1, cts.Token);

        Assert.Equal(PopulationPartitionOutcome.Success, results[0].Outcome);
        Assert.Equal(PopulationPartitionOutcome.Cancelled, results[1].Outcome);

        var okStore = new FileEvolutionEvidenceStore(outputs[0].EvidenceDirectory);
        var manifest = okStore.ReadManifest();
        Assert.Equal("ok", manifest.RunId);
        var events = okStore.ReadEvents();
        Assert.NotEmpty(events);
        Assert.Equal(nameof(EvolutionPipelineDecisionKind.Stop), events[^1].DecisionKind);
    }

    [Fact]
    public void Runner_output_lineage_is_persisted_and_resume_rejects_other_output_identity()
    {
        var problem = DoubleProblem();
        var population = InitialPopulation(problem, 4, seed: 13);
        var snapshot = PopulationSnapshot<double>.FromPopulation(population, problem, new[] { "source-run" });
        var outputs = new[]
        {
            new PopulationOutputSpec("alpha", 1, Path.Combine(Path.GetTempPath(), Guid.NewGuid().ToString("N"))),
            new PopulationOutputSpec("beta", 2, Path.Combine(Path.GetTempPath(), Guid.NewGuid().ToString("N"))),
        };
        var request = new PopulationPartitionRequest<double>(snapshot, outputs, PopulationPartitionMode.DisjointExhaustive, 1);

        EvolutionPipelineDefinition<double> DefinitionFactory(PopulationOutputSpec spec, int populationSize) =>
            new("pipe-lineage", "main", new[]
            {
                new EvolutionPipelineState<double>("main", DoubleProblem(), new EvolutionOptions
                {
                    PopulationSize = populationSize,
                    MaxGenerations = 1,
                    Seed = spec.Seed,
                    WorkerCount = 1,
                }),
            });

        var runner = new PopulationPartitionRunner<double>();
        var results = runner.Run(request, new ContiguousBlockPartitioner<double>(), DefinitionFactory, maxDegreeOfParallelism: 1);
        Assert.All(results, r => Assert.Equal(PopulationPartitionOutcome.Success, r.Outcome));

        var alphaStore = new FileEvolutionEvidenceStore(outputs[0].EvidenceDirectory);
        var alphaManifest = alphaStore.ReadManifest();
        Assert.Equal(new[] { "source-run", "alpha" }, alphaManifest.Lineage);

        var pipelineRunner = new EvolutionPipelineRunner<double>();
        var definition = DefinitionFactory(outputs[0], 2);

        // Resume rejeita o uso da lineage declarada da OUTRA saída ("beta") sobre os artefatos de "alpha".
        var wrongLineage = new[] { "source-run", "beta" };
        var ex = Assert.Throws<InvalidDataException>(
            () => pipelineRunner.Resume(definition, "alpha", outputs[0].EvidenceDirectory, wrongLineage));
        Assert.Contains("Lineage", ex.Message);

        // Com a lineage correta da própria saída, a validação de lineage passa; a exceção aqui (se houver) é a de
        // "pipeline já concluído", não mais relacionada a identidade/lineage.
        var correctLineage = new[] { "source-run", "alpha" };
        var other = Record.Exception(() => pipelineRunner.Resume(definition, "alpha", outputs[0].EvidenceDirectory, correctLineage));
        Assert.IsNotType<InvalidDataException>(other);
    }
}
