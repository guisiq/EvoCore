using System.Collections.Generic;
using System.IO;
using System.Linq;
using Evo.Core.Functions;
using Evo.Core.Persistence;
using Evo.Core.Values;
using Xunit;

namespace Evo.Core.Tests;

public sealed class CheckpointPersistenceTests
{
    private static readonly GenomeShape Shape = new(1, 8, 1, 2, 3);

    private static EvolutionTestProblem<double> DoubleProblem() => new(
        Shape,
        StandardDoubleFunctionSet.Instance,
        new DoubleValueSemantics(),
        new DoubleParameterStrategy(),
        () => new QuadraticFitness());

    private static EvolutionOptions Options(int maxGenerations) => new()
    {
        PopulationSize = 24,
        MaxGenerations = maxGenerations,
        TournamentSize = 3,
        EliteCount = 2,
        Seed = 7,
        WorkerCount = 1,
    };

    private static string NewTempDir()
    {
        var dir = Path.Combine(Path.GetTempPath(), "evo-ckpt-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(dir);
        return dir;
    }

    private static bool DoubleBits(double a, double b) => BitConverter.DoubleToInt64Bits(a) == BitConverter.DoubleToInt64Bits(b);

    private static bool DoubleArrayBits(double[] a, double[] b) =>
        a.Length == b.Length && Enumerable.Range(0, a.Length).All(i => DoubleBits(a[i], b[i]));

    private static bool IntEq(int a, int b) => a == b;

    private static bool IntArrayEq(int[] a, int[] b) => a.AsSpan().SequenceEqual(b);

    private static void AssertSameGenome<T>(Genome<T> expected, Genome<T> actual, IGeneticProblem<T> problem, Func<T, T, bool> equal)
    {
        Assert.Equal(expected.Shape, actual.Shape);
        Assert.Equal(expected.FunctionGenes.ToArray(), actual.FunctionGenes.ToArray());
        Assert.Equal(expected.ConnectionGenes.ToArray(), actual.ConnectionGenes.ToArray());
        Assert.Equal(expected.OutputGenes.ToArray(), actual.OutputGenes.ToArray());
        var a = expected.ParameterGenes.ToArray();
        var b = actual.ParameterGenes.ToArray();
        Assert.Equal(a.Length, b.Length);
        for (var i = 0; i < a.Length; i++)
        {
            Assert.True(equal(a[i], b[i]), $"ParameterGene {i} diferente.");
        }

        Assert.Equal(
            expected.ComputePhenotypeFingerprint(problem.Functions, problem.Values),
            actual.ComputePhenotypeFingerprint(problem.Functions, problem.Values));
    }

    private static void AssertIndividualIdentity<T>(IGeneticProblem<T> problem, Func<T, T, bool> equal)
    {
        var options = Options(1);
        var operators = new EvolutionOperators<T>(problem);
        for (var i = 0; i < 20; i++)
        {
            var rng = RandomStream.Create(options.Seed, 0, (ulong)i, 1);
            var original = operators.CreateInitial(ref rng);
            var json = GenomeJson.Serialize(original, problem);
            var loaded = GenomeJson.Deserialize(json, problem, options);
            AssertSameGenome(original, loaded, problem, equal);
        }
    }

    [Fact]
    public void Checkpoint_individual_round_trip_is_exact_for_all_four_value_types()
    {
        AssertIndividualIdentity(DoubleProblem(), DoubleBits);
        AssertIndividualIdentity(
            new EvolutionTestProblem<int>(Shape, StandardInt32FunctionSet.Instance, new Int32ValueSemantics(), new IntParameterStrategy(), () => throw new NotSupportedException()),
            IntEq);
        AssertIndividualIdentity(
            new EvolutionTestProblem<double[]>(Shape, StandardDoubleArrayFunctionSet.Instance, new DoubleArrayValueSemantics(), new DoubleArrayParameterStrategy(3), () => throw new NotSupportedException()),
            DoubleArrayBits);
        AssertIndividualIdentity(
            new EvolutionTestProblem<int[]>(Shape, StandardInt32ArrayFunctionSet.Instance, new Int32ArrayValueSemantics(), new IntArrayParameterStrategy(3), () => throw new NotSupportedException()),
            IntArrayEq);
    }

    [Fact]
    public void Checkpoint_rejects_incompatible_variation_strategy_identity_before_resume()
    {
        var problem = DoubleProblem();
        var options = Options(5);
        EvolutionState<double>? captured = null;
        new EvolutionEngine<double>(checkpoint: s => { if (s.Generation == 2) captured = s; }).Run(problem, options);
        Assert.NotNull(captured);

        var dir = NewTempDir();
        try
        {
            var path = Path.Combine(dir, "checkpoint.json");
            CheckpointFile.Save(path, problem, options, captured!, variationStrategyId: "Strategy.A");

            // Identidade divergente é rejeitada antes de qualquer estado ser materializado.
            Assert.Throws<InvalidDataException>(() => CheckpointFile.Load(path, problem, options, variationStrategyId: "Strategy.B"));

            // Identidade igual, ou lado ausente (null), é compatível.
            var loaded = CheckpointFile.Load(path, problem, options, variationStrategyId: "Strategy.A");
            Assert.Equal(captured!.Generation, loaded.Generation);
            var loadedLegacy = CheckpointFile.Load(path, problem, options);
            Assert.Equal(captured.Generation, loadedLegacy.Generation);
        }
        finally
        {
            Directory.Delete(dir, recursive: true);
        }
    }

    [Fact]
    public void Checkpoint_population_round_trip_preserves_state()
    {
        var problem = DoubleProblem();
        var options = Options(5);
        EvolutionState<double>? captured = null;
        new EvolutionEngine<double>(checkpoint: s => { if (s.Generation == 3) captured = s; })
            .Run(problem, options);
        Assert.NotNull(captured);

        var dir = NewTempDir();
        try
        {
            var path = Path.Combine(dir, "checkpoint.json");
            CheckpointFile.Save(path, problem, options, captured!);
            var loaded = CheckpointFile.Load(path, problem, options);

            Assert.Equal(captured!.Generation, loaded.Generation);
            Assert.Equal(captured.LastImprovement, loaded.LastImprovement);
            Assert.Equal(captured.BestFitness, loaded.BestFitness);
            Assert.Equal(captured.Fitness, loaded.Fitness);
            Assert.Equal(captured.Population.Length, loaded.Population.Length);
            for (var i = 0; i < captured.Population.Length; i++)
            {
                AssertSameGenome(captured.Population[i], loaded.Population[i], problem, DoubleBits);
            }

            Assert.NotNull(loaded.BestGenome);
            AssertSameGenome(captured.BestGenome!, loaded.BestGenome!, problem, DoubleBits);
            Assert.Single(Directory.GetFiles(dir));
        }
        finally
        {
            Directory.Delete(dir, recursive: true);
        }
    }

    [Fact]
    public void Checkpoint_corrupted_or_incompatible_files_are_rejected_and_do_not_destroy_last_valid()
    {
        var problem = DoubleProblem();
        var options = Options(5);
        EvolutionState<double>? captured = null;
        new EvolutionEngine<double>(checkpoint: s => { if (s.Generation == 2) captured = s; })
            .Run(problem, options);

        var dir = NewTempDir();
        try
        {
            var good = Path.Combine(dir, "good.json");
            CheckpointFile.Save(good, problem, options, captured!);
            var text = File.ReadAllText(good);

            var corruptions = new Dictionary<string, string>
            {
                ["truncated"] = text[..(text.Length / 2)],
                ["problemId"] = text.Replace("evolution-test-problem", "other-problem"),
                ["functionSetId"] = text.Replace(StandardDoubleFunctionSet.Instance.Id, "other-functions"),
                ["rngVersion"] = text.Replace(RandomStream.Version, "rng/other/v0"),
                ["valueTypeId"] = text.Replace("\"double\"", "\"int32\""),
                ["fingerprint"] = text.Replace("\"phenotypeFingerprint\": \"", "\"phenotypeFingerprint\": \"00"),
            };

            foreach (var (name, content) in corruptions)
            {
                var bad = Path.Combine(dir, name + ".json");
                File.WriteAllText(bad, content);
                Assert.Throws<InvalidDataException>(() => CheckpointFile.Load(bad, problem, options));
            }

            var wrongSeed = Options(5) with { Seed = 8 };
            Assert.Throws<InvalidDataException>(() => CheckpointFile.Load(good, problem, wrongSeed));

            var loaded = CheckpointFile.Load(good, problem, options);
            Assert.Equal(captured!.Generation, loaded.Generation);
            Assert.Equal(text, File.ReadAllText(good));
        }
        finally
        {
            Directory.Delete(dir, recursive: true);
        }
    }

    [Fact]
    public void Checkpoint_resume_equals_continuous_run_for_every_worker_count()
    {
        const int generations = 20;
        const int checkpointAt = 7;
        var problem = DoubleProblem();

        foreach (var workers in new[] { 1, 4 })
        {
            var options = Options(generations) with { WorkerCount = workers };
            var fullProgress = new List<EvolutionProgress>();
            EvolutionState<double>? state = null;
            var full = new EvolutionEngine<double>(
                p => fullProgress.Add(p),
                s => { if (s.Generation == checkpointAt) state = s; })
                .Run(problem, options);
            Assert.NotNull(state);

            var dir = NewTempDir();
            try
            {
                var path = Path.Combine(dir, "checkpoint.json");
                CheckpointFile.Save(path, problem, options, state!);
                var loaded = CheckpointFile.Load(path, problem, options);

                var resumeProgress = new List<EvolutionProgress>();
                var resumed = new EvolutionEngine<double>(p => resumeProgress.Add(p))
                    .Resume(problem, options, loaded);

                Assert.Equal(full.BestFitness, resumed.BestFitness);
                Assert.Equal(full.GenerationsRun, resumed.GenerationsRun);
                Assert.Equal(full.Termination, resumed.Termination);
                AssertSameGenome(full.BestGenome, resumed.BestGenome, problem, DoubleBits);
                Assert.Equal(fullProgress.Skip(checkpointAt), resumeProgress);
            }
            finally
            {
                Directory.Delete(dir, recursive: true);
            }
        }
    }
}
