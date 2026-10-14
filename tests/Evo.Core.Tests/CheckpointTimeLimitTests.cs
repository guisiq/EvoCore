using System.IO;
using Evo.Core.Functions;
using Evo.Core.Persistence;
using Evo.Core.Values;
using Xunit;

namespace Evo.Core.Tests;

public sealed class CheckpointTimeLimitTests
{
    private static readonly GenomeShape Shape = new(1, 8, 1, 2, 3);

    private sealed class ManualClock
    {
        public TimeSpan Now { get; set; }
    }

    private static EvolutionTestProblem<double> DoubleProblem() => new(
        Shape,
        StandardDoubleFunctionSet.Instance,
        new DoubleValueSemantics(),
        new DoubleParameterStrategy(),
        () => new QuadraticFitness());

    private static EvolutionOptions Options(int maxGenerations, TimeSpan? timeLimit = null) => new()
    {
        PopulationSize = 24,
        MaxGenerations = maxGenerations,
        TournamentSize = 3,
        EliteCount = 2,
        Seed = 7,
        WorkerCount = 1,
        TimeLimit = timeLimit,
    };

    private static string NewTempDir()
    {
        var dir = Path.Combine(Path.GetTempPath(), "evo-tl-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(dir);
        return dir;
    }

    [Fact]
    public void Legacy_v1_checkpoint_loads_with_unknown_elapsed_diagnostic_and_zero_carried_time()
    {
        var problem = DoubleProblem();
        var options = Options(5);
        EvolutionState<double>? captured = null;
        new EvolutionEngine<double>(checkpoint: s => { if (s.Generation == 2) captured = s; }).Run(problem, options);

        var dir = NewTempDir();
        try
        {
            var path = Path.Combine(dir, "checkpoint.json");
            CheckpointFile.Save(path, problem, options, captured!);
            var legacy = File.ReadAllText(path).Replace(CheckpointFile.SchemaVersion, CheckpointFile.LegacySchemaVersion);
            File.WriteAllText(path, legacy);

            var loaded = CheckpointFile.Load(path, problem, options);
            Assert.False(loaded.ElapsedKnown);
            Assert.Equal(TimeSpan.Zero, loaded.ActiveElapsed);
            Assert.Equal(captured!.Generation, loaded.Generation);
        }
        finally
        {
            Directory.Delete(dir, recursive: true);
        }
    }

    [Fact]
    public void Checkpoint_with_different_TimeLimit_is_rejected_before_state_is_built()
    {
        var problem = DoubleProblem();
        var options = Options(5, TimeSpan.FromSeconds(5));
        EvolutionState<double>? captured = null;
        new EvolutionEngine<double>(checkpoint: s => { if (s.Generation == 1) captured = s; }).Run(problem, options);

        var dir = NewTempDir();
        try
        {
            var path = Path.Combine(dir, "checkpoint.json");
            CheckpointFile.Save(path, problem, options, captured!);

            Assert.Throws<InvalidDataException>(() =>
                CheckpointFile.Load(path, problem, Options(5, TimeSpan.FromSeconds(6))));
            var loaded = CheckpointFile.Load(path, problem, options);
            Assert.True(loaded.ElapsedKnown);
            Assert.Equal(captured!.ActiveElapsed, loaded.ActiveElapsed);
        }
        finally
        {
            Directory.Delete(dir, recursive: true);
        }
    }

    [Fact]
    public void TimeLimit_is_cumulative_across_resumed_segments_and_excludes_paused_time()
    {
        var problem = DoubleProblem();
        var limit = TimeSpan.FromSeconds(5);
        var options = Options(100, limit);

        // Cada progresso avança o relógio fake em 1 s de tempo ativo.
        var clock1 = new ManualClock();
        EvolutionState<double>? captured = null;
        var first = new EvolutionEngine<double>(
            progress: _ => clock1.Now += TimeSpan.FromSeconds(1),
            checkpoint: s => { if (s.Generation == 1) captured = s; },
            clock: () => clock1.Now)
            .Run(problem, options);

        Assert.Equal(TerminationReason.TimeLimit, first.Termination);
        Assert.Equal(4, first.GenerationsRun);
        Assert.NotNull(captured);
        Assert.Equal(TimeSpan.FromSeconds(2), captured!.ActiveElapsed);

        var dir = NewTempDir();
        try
        {
            var path = Path.Combine(dir, "checkpoint.json");
            CheckpointFile.Save(path, problem, options, captured);
            var loaded = CheckpointFile.Load(path, problem, options);
            Assert.True(loaded.ElapsedKnown);
            Assert.Equal(TimeSpan.FromSeconds(2), loaded.ActiveElapsed);

            // Pausa entre processos: o novo segmento começa com relógio zerado e não conta o intervalo anterior.
            var clock2 = new ManualClock();
            var resumed = new EvolutionEngine<double>(
                progress: _ => clock2.Now += TimeSpan.FromSeconds(1),
                checkpoint: null,
                clock: () => clock2.Now)
                .Resume(problem, options, loaded);

            Assert.Equal(TerminationReason.TimeLimit, resumed.Termination);
            Assert.Equal(3, resumed.GenerationsRun);
        }
        finally
        {
            Directory.Delete(dir, recursive: true);
        }
    }

    [Fact]
    public void Transient_replace_failure_is_retried_and_succeeds()
    {
        var problem = DoubleProblem();
        var options = Options(5);
        EvolutionState<double>? captured = null;
        new EvolutionEngine<double>(checkpoint: s => { if (s.Generation == 2) captured = s; }).Run(problem, options);

        var dir = NewTempDir();
        try
        {
            var path = Path.Combine(dir, "checkpoint.json");
            var key = Path.GetFullPath(path);
            var attempts = 0;
            CheckpointFile.ReplaceFaults[key] = attempt =>
            {
                Interlocked.Increment(ref attempts);
                return attempt <= 2 ? new IOException("falha transitória simulada") : null;
            };
            try
            {
                CheckpointFile.Save(path, problem, options, captured!);
            }
            finally
            {
                CheckpointFile.ReplaceFaults.TryRemove(key, out _);
            }

            Assert.Equal(3, attempts);
            Assert.Equal(captured!.Generation, CheckpointFile.Load(path, problem, options).Generation);
        }
        finally
        {
            Directory.Delete(dir, recursive: true);
        }
    }

    [Fact]
    public void Permanent_replace_failure_is_bounded_reported_and_preserves_last_valid_checkpoint()
    {
        var problem = DoubleProblem();
        var options = Options(5);
        EvolutionState<double>? early = null;
        EvolutionState<double>? late = null;
        new EvolutionEngine<double>(checkpoint: s =>
        {
            if (s.Generation == 1) early = s;
            if (s.Generation == 3) late = s;
        }).Run(problem, options);

        var dir = NewTempDir();
        try
        {
            var path = Path.Combine(dir, "checkpoint.json");
            CheckpointFile.Save(path, problem, options, early!);
            var before = File.ReadAllText(path);

            var key = Path.GetFullPath(path);
            var attempts = 0;
            CheckpointFile.ReplaceFaults[key] = _ =>
            {
                Interlocked.Increment(ref attempts);
                return new UnauthorizedAccessException("bloqueio permanente simulado");
            };
            try
            {
                var error = Assert.Throws<IOException>(() => CheckpointFile.Save(path, problem, options, late!));
                Assert.Contains("20 tentativas", error.Message);
                Assert.IsType<UnauthorizedAccessException>(error.InnerException);
            }
            finally
            {
                CheckpointFile.ReplaceFaults.TryRemove(key, out _);
            }

            Assert.Equal(CheckpointFile.MaxReplaceAttempts, attempts);
            Assert.Equal(before, File.ReadAllText(path));
            Assert.Equal(early!.Generation, CheckpointFile.Load(path, problem, options).Generation);
            Assert.Empty(Directory.GetFiles(dir, "*.tmp"));
        }
        finally
        {
            Directory.Delete(dir, recursive: true);
        }
    }
}
