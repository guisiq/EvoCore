using System.Text.Json;
using Evo.Core.Functions;
using Evo.Core.Pipeline;
using Evo.Core.Values;
using Xunit;

namespace Evo.Core.Tests;

public sealed class EvolutionPipelineDocumentTests
{
    private static readonly GenomeShape Shape = new(1, 8, 1, 2, 3);

    private static EvolutionTestProblem<double> Problem() => new(
        Shape,
        new StandardDoubleFunctionSet(),
        new DoubleValueSemantics(),
        new DoubleParameterStrategy(),
        () => new QuadraticFitness());

    private static EvolutionPipelineRegistry<double> Registry()
    {
        var registry = new EvolutionPipelineRegistry<double>();
        registry.RegisterProblem("quad", Problem);
        registry.RegisterStopCondition("stop-at-two", o => o.Generation >= 2);
        registry.RegisterDecision("keep", _ => EvolutionPipelineDecision.Continue);
        return registry;
    }

    private const string ValidJson = """
        {
          "schema": "evo-pipeline/v1",
          "pipelineId": "pipe-json",
          "initialState": "main",
          "maxTransitions": 0,
          "states": [
            {
              "stateId": "main",
              "problem": "quad",
              "options": { "populationSize": 12, "maxGenerations": 50, "seed": 42, "workerCount": 1 },
              "stopCondition": "stop-at-two",
              "decision": "keep"
            }
          ]
        }
        """;

    [Fact]
    public void Valid_document_loads_and_matches_code_definition()
    {
        var loaded = EvolutionPipelineDocument.Load(ValidJson, Registry());
        var fromCode = new EvolutionPipelineDefinition<double>(
            "pipe-json",
            "main",
            new[]
            {
                new EvolutionPipelineState<double>(
                    "main",
                    Problem(),
                    new EvolutionOptions { PopulationSize = 12, MaxGenerations = 50, Seed = 42, WorkerCount = 1 },
                    StopConditionId: "stop-at-two",
                    StopCondition: o => o.Generation >= 2,
                    DecisionId: "keep",
                    Decide: _ => EvolutionPipelineDecision.Continue),
            });

        var a = new EvolutionPipelineRunner<double>().Run(loaded, "run-json");
        var b = new EvolutionPipelineRunner<double>().Run(fromCode, "run-code");

        Assert.Equal(b.BestFitness.Value, a.BestFitness.Value);
        Assert.Equal(b.GenerationsRun, a.GenerationsRun);
        Assert.Equal(EvolutionPipelineTermination.PolicyStop, a.Termination);
    }

    [Fact]
    public void Unknown_field_is_rejected()
    {
        var json = ValidJson.Replace("\"maxTransitions\": 0,", "\"maxTransitions\": 0, \"extra\": 1,");
        Assert.Throws<InvalidDataException>(() => EvolutionPipelineDocument.Load(json, Registry()));
    }

    [Fact]
    public void Unknown_option_field_is_rejected()
    {
        var json = ValidJson.Replace("\"workerCount\": 1", "\"workerCount\": 1, \"bogus\": 2");
        Assert.Throws<InvalidDataException>(() => EvolutionPipelineDocument.Load(json, Registry()));
    }

    [Fact]
    public void Unregistered_id_is_rejected()
    {
        var json = ValidJson.Replace("\"decision\": \"keep\"", "\"decision\": \"missing\"");
        Assert.Throws<InvalidDataException>(() => EvolutionPipelineDocument.Load(json, Registry()));
    }

    [Fact]
    public void Rejected_document_creates_no_evidence_files()
    {
        var dir = Path.Combine(Path.GetTempPath(), "evo-pipe-" + Guid.NewGuid().ToString("N"));
        var json = ValidJson.Replace("\"quad\"", "\"nao-registrado\"");
        try
        {
            Assert.Throws<InvalidDataException>(() => EvolutionPipelineDocument.Load(json, Registry()));
            Assert.False(Directory.Exists(dir));
        }
        finally
        {
            if (Directory.Exists(dir))
            {
                Directory.Delete(dir, recursive: true);
            }
        }
    }

    [Fact]
    public void Events_have_monotonic_sequence_and_resume_of_completed_run_is_rejected()
    {
        var definition = EvolutionPipelineDocument.Load(ValidJson, Registry());
        var dir = Path.Combine(Path.GetTempPath(), "evo-pipe-run-" + Guid.NewGuid().ToString("N"));
        try
        {
            new EvolutionPipelineRunner<double>().Run(definition, "run-events", dir);
            var events = new FileEvolutionEvidenceStore(dir).ReadEvents();
            Assert.NotEmpty(events);
            for (var i = 0; i < events.Count; i++)
            {
                Assert.Equal(i + 1, events[i].Sequence);
            }

            Assert.Equal(nameof(EvolutionPipelineDecisionKind.Stop), events[^1].DecisionKind);
            Assert.ThrowsAny<Exception>(() =>
                new EvolutionPipelineRunner<double>().Resume(definition, "run-events", dir));
        }
        finally
        {
            if (Directory.Exists(dir))
            {
                Directory.Delete(dir, recursive: true);
            }
        }
    }

    [Fact]
    public void Max_transitions_stops_a_cycle()
    {
        var registry = Registry();
        registry.RegisterDecision("ping", _ => EvolutionPipelineDecision.Transition("pong", "ciclo"));
        registry.RegisterDecision("pong-back", _ => EvolutionPipelineDecision.Transition("ping-state", "ciclo"));
        var definition = new EvolutionPipelineDefinition<double>(
            "pipe-cycle",
            "ping-state",
            new[]
            {
                new EvolutionPipelineState<double>("ping-state", Problem(), new EvolutionOptions { PopulationSize = 12, MaxGenerations = 500, Seed = 1 },
                    DecisionId: "ping", Decide: _ => EvolutionPipelineDecision.Transition("pong", "ciclo")),
                new EvolutionPipelineState<double>("pong", Problem(), new EvolutionOptions { PopulationSize = 12, MaxGenerations = 500, Seed = 1 },
                    DecisionId: "pong-back", Decide: _ => EvolutionPipelineDecision.Transition("ping-state", "ciclo")),
            },
            maxTransitions: 3);

        Assert.Throws<InvalidOperationException>(() => new EvolutionPipelineRunner<double>().Run(definition, "run-cycle"));
    }

    [Fact]
    public void Observation_trajectory_is_identical_for_1_and_2_workers()
    {
        List<string> Trajectory(int workers)
        {
            var seen = new List<string>();
            var definition = new EvolutionPipelineDefinition<double>(
                "pipe-trajectory",
                "main",
                new[]
                {
                    new EvolutionPipelineState<double>("main", Problem(),
                        new EvolutionOptions { PopulationSize = 12, MaxGenerations = 6, Seed = 7, WorkerCount = workers },
                        DecisionId: "record",
                        Decide: o =>
                        {
                            seen.Add($"{o.Generation}:{o.BestFitness.Value:R}:{o.PopulationDigest}");
                            return EvolutionPipelineDecision.Continue;
                        }),
                });
            new EvolutionPipelineRunner<double>().Run(definition, "run-traj" + workers);
            return seen;
        }

        Assert.Equal(Trajectory(1), Trajectory(2));
    }
}
