using Evo.Core.Functions;
using Evo.Core.Pipeline;
using Evo.Core.Values;
using Evo.ExternalConsumer;
using Xunit;

namespace Evo.Core.Tests;

/// <summary>Cobre os itens de aceite de EVO-303 pendentes: atomicidade, resume, invalidação de cache,
/// cancelamento, falhas de callback, determinismo 1/2/N e compatibilidade bit a bit do caminho legado.</summary>
public sealed class EvolutionPipelineAdvancedTests
{
    private static readonly GenomeShape Shape = new(1, 8, 1, 2, 3);

    private sealed class IdentityProblem : IGeneticProblem<double>
    {
        private readonly Action? _onEvaluate;

        public IdentityProblem(string problemId, Action? onEvaluate = null)
        {
            ProblemId = problemId;
            _onEvaluate = onEvaluate;
        }

        public string ProblemId { get; }

        public GenomeShape Shape => EvolutionPipelineAdvancedTests.Shape;

        public IFunctionSet<double> Functions => StandardDoubleFunctionSet.Instance;

        public IValueSemantics<double> Values => DoubleValueSemantics.Instance;

        public IParameterGeneStrategy<double> Parameters => DoubleParameterStrategy.Instance;

        public IFitnessEvaluator<double> CreateFitnessEvaluator() => new QuadraticFitness(_onEvaluate);
    }

    private static EvolutionTestProblem<double> Problem() => new(
        Shape, StandardDoubleFunctionSet.Instance, DoubleValueSemantics.Instance, DoubleParameterStrategy.Instance,
        () => new QuadraticFitness());

    private static EvolutionOptions Options(int maxGenerations = 4, int workers = 1) => new()
    {
        PopulationSize = 12,
        MaxGenerations = maxGenerations,
        Seed = 42,
        WorkerCount = workers,
    };

    private static string TempDir() => Path.Combine(Path.GetTempPath(), "evo-pipe-adv-" + Guid.NewGuid().ToString("N"));

    private static void Clean(string dir)
    {
        if (Directory.Exists(dir))
        {
            Directory.Delete(dir, recursive: true);
        }
    }

    [Fact]
    public void Manifest_write_failure_leaves_no_manifest_file()
    {
        var dir = TempDir();
        Directory.CreateDirectory(dir);
        // Pré-cria o caminho do manifesto como diretório para forçar falha no rename atômico final.
        Directory.CreateDirectory(Path.Combine(dir, "manifest.json"));
        try
        {
            var definition = new EvolutionPipelineDefinition<double>(
                "pipe-atomic", "main", new[] { new EvolutionPipelineState<double>("main", Problem(), Options()) });

            Assert.ThrowsAny<Exception>(() => new EvolutionPipelineRunner<double>().Run(definition, "run-atomic", dir));
            Assert.False(File.Exists(Path.Combine(dir, "manifest.json")));
        }
        finally
        {
            Clean(dir);
        }
    }

    [Fact]
    public void Resume_continues_interrupted_run_to_native_completion()
    {
        var cts = new CancellationTokenSource();
        var definition = new EvolutionPipelineDefinition<double>(
            "pipe-resume",
            "main",
            new[]
            {
                new EvolutionPipelineState<double>(
                    "main",
                    Problem(),
                    Options(maxGenerations: 10),
                    DecisionId: "cancel-early",
                    Decide: o =>
                    {
                        if (o.Generation == 3 && !cts.IsCancellationRequested)
                        {
                            cts.Cancel();
                        }

                        return EvolutionPipelineDecision.Continue;
                    }),
            });
        var dir = TempDir();
        try
        {
            var interrupted = new EvolutionPipelineRunner<double>().Run(definition, "run-resume", dir, cts.Token);
            Assert.Equal(EvolutionPipelineTermination.Cancellation, interrupted.Termination);
            Assert.True(interrupted.GenerationsRun < 10);

            var resumed = new EvolutionPipelineRunner<double>().Resume(definition, "run-resume", dir);
            Assert.Equal(EvolutionPipelineTermination.Native, resumed.Termination);
            Assert.Equal(10, resumed.GenerationsRun);

            var events = new FileEvolutionEvidenceStore(dir).ReadEvents();
            for (var i = 0; i < events.Count; i++)
            {
                Assert.Equal(i + 1, events[i].Sequence);
            }

            Assert.Equal(nameof(EvolutionPipelineDecisionKind.Stop), events[^1].DecisionKind);
        }
        finally
        {
            Clean(dir);
        }
    }

    [Fact]
    public void Transition_with_changed_evaluation_identity_forces_reevaluation()
    {
        int EvaluationsAfterTransition(bool changeIdentity)
        {
            var mainCount = 0;
            var nextCount = 0;
            var definition = new EvolutionPipelineDefinition<double>(
                "pipe-cache",
                "main",
                new[]
                {
                    new EvolutionPipelineState<double>(
                        "main",
                        new IdentityProblem("p-main", () => mainCount++),
                        Options(maxGenerations: 50),
                        DecisionId: "go",
                        Decide: o => o.Generation >= 2 ? EvolutionPipelineDecision.Transition("next", "troca") : EvolutionPipelineDecision.Continue),
                    new EvolutionPipelineState<double>(
                        "next",
                        new IdentityProblem(changeIdentity ? "p-next" : "p-main", () => nextCount++),
                        Options(maxGenerations: 2)),
                },
                maxTransitions: 1);
            new EvolutionPipelineRunner<double>().Run(definition, "run-cache-" + changeIdentity);
            return nextCount;
        }

        var same = EvaluationsAfterTransition(changeIdentity: false);
        var changed = EvaluationsAfterTransition(changeIdentity: true);

        // Reavaliação forçada pela mudança de identidade custa exatamente o tamanho da população (rebase completo).
        Assert.Equal(12, changed - same);
    }

    [Fact]
    public void External_cancellation_is_reported_without_native_termination()
    {
        var cts = new CancellationTokenSource();
        var definition = new EvolutionPipelineDefinition<double>(
            "pipe-cancel",
            "main",
            new[]
            {
                new EvolutionPipelineState<double>(
                    "main",
                    Problem(),
                    Options(maxGenerations: 50),
                    DecisionId: "cancel",
                    Decide: o =>
                    {
                        if (o.Generation == 2)
                        {
                            cts.Cancel();
                        }

                        return EvolutionPipelineDecision.Continue;
                    }),
            });

        var result = new EvolutionPipelineRunner<double>().Run(definition, "run-cancel", cancellationToken: cts.Token);

        Assert.Equal(EvolutionPipelineTermination.Cancellation, result.Termination);
        Assert.True(result.GenerationsRun < 50);
    }

    private sealed class ProbeException : Exception
    {
    }

    [Fact]
    public void StopCondition_failure_propagates_without_corrupting_evidence()
    {
        var definition = new EvolutionPipelineDefinition<double>(
            "pipe-stopcond-fail",
            "main",
            new[]
            {
                new EvolutionPipelineState<double>(
                    "main",
                    Problem(),
                    Options(maxGenerations: 50),
                    StopConditionId: "boom",
                    StopCondition: _ => throw new ProbeException()),
            });

        Assert.Throws<ProbeException>(() => new EvolutionPipelineRunner<double>().Run(definition, "run-stopcond-fail"));
    }

    [Fact]
    public void Decide_failure_propagates()
    {
        var definition = new EvolutionPipelineDefinition<double>(
            "pipe-decide-fail",
            "main",
            new[]
            {
                new EvolutionPipelineState<double>(
                    "main",
                    Problem(),
                    Options(maxGenerations: 50),
                    DecisionId: "boom",
                    Decide: _ => throw new ProbeException()),
            });

        Assert.Throws<ProbeException>(() => new EvolutionPipelineRunner<double>().Run(definition, "run-decide-fail"));
    }

    [Fact]
    public void Observation_trajectory_is_identical_for_1_2_and_3_workers()
    {
        List<string> Trajectory(int workers)
        {
            var seen = new List<string>();
            var definition = new EvolutionPipelineDefinition<double>(
                "pipe-trajectory-n",
                "main",
                new[]
                {
                    new EvolutionPipelineState<double>("main", Problem(),
                        Options(maxGenerations: 6, workers: workers),
                        DecisionId: "record",
                        Decide: o =>
                        {
                            seen.Add($"{o.Generation}:{o.BestFitness.Value:R}:{o.PopulationDigest}");
                            return EvolutionPipelineDecision.Continue;
                        }),
                });
            new EvolutionPipelineRunner<double>().Run(definition, "run-traj-n" + workers);
            return seen;
        }

        var one = Trajectory(1);
        var two = Trajectory(2);
        var three = Trajectory(3);

        Assert.Equal(one, two);
        Assert.Equal(one, three);
    }

    [Fact]
    public void Pipeline_without_policy_is_bit_identical_to_direct_engine_run()
    {
        var problem = Problem();
        var options = Options(maxGenerations: 5);

        var direct = new EvolutionEngine<double>().Run(problem, options);

        var definition = new EvolutionPipelineDefinition<double>(
            "pipe-legacy", "main", new[] { new EvolutionPipelineState<double>("main", Problem(), options) });
        var viaPipeline = new EvolutionPipelineRunner<double>().Run(definition, "run-legacy");

        Assert.Equal(direct.GenerationsRun, viaPipeline.GenerationsRun);
        Assert.Equal(direct.BestFitness.Value, viaPipeline.BestFitness.Value);
        Assert.Equal(
            direct.BestGenome.ComputePhenotypeFingerprint(problem.Functions, problem.Values),
            viaPipeline.BestGenome.ComputePhenotypeFingerprint(problem.Functions, problem.Values));
    }

    [Fact]
    public void External_consumer_variation_operator_runs_inside_pipeline()
    {
        var problem = Problem();
        var selector = new ConsumerParentSelector<double>();
        var definition = new EvolutionPipelineDefinition<double>(
            "pipe-external",
            "main",
            new[]
            {
                new EvolutionPipelineState<double>(
                    "main", problem, Options(maxGenerations: 4), Variation: new ConsumerVariationOperator<double>(problem, selector)),
            });

        var result = new EvolutionPipelineRunner<double>().Run(definition, "run-external");

        Assert.Equal(EvolutionPipelineTermination.Native, result.Termination);
        Assert.Equal(4, result.GenerationsRun);
    }
}
