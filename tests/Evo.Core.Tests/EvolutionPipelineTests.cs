using Evo.Core.Functions;
using Evo.Core.Pipeline;
using Evo.Core.Values;
using Xunit;

namespace Evo.Core.Tests;

public sealed class EvolutionPipelineTests
{
    private static readonly GenomeShape Shape = new(1, 8, 1, 2, 3);

    private static EvolutionTestProblem<double> Problem() => new(
        Shape,
        new StandardDoubleFunctionSet(),
        new DoubleValueSemantics(),
        new DoubleParameterStrategy(),
        () => new QuadraticFitness());

    private static EvolutionOptions Options(int maxGenerations = 3, int workers = 1) => new()
    {
        PopulationSize = 12,
        MaxGenerations = maxGenerations,
        Seed = 42,
        WorkerCount = workers,
    };

    [Fact]
    public void Pipeline_without_policy_terminates_natively()
    {
        var definition = new EvolutionPipelineDefinition<double>(
            "pipe-native",
            "main",
            new[] { new EvolutionPipelineState<double>("main", Problem(), Options(maxGenerations: 3)) });

        var result = new EvolutionPipelineRunner<double>().Run(definition, "run-1");

        Assert.Equal(EvolutionPipelineTermination.Native, result.Termination);
        Assert.Equal("main", result.FinalStateId);
        Assert.Equal(0, result.Transitions);
    }

    [Fact]
    public void Stop_decision_yields_policy_stop()
    {
        var definition = new EvolutionPipelineDefinition<double>(
            "pipe-stop",
            "main",
            new[]
            {
                new EvolutionPipelineState<double>(
                    "main",
                    Problem(),
                    Options(maxGenerations: 50),
                    DecisionId: "stop-at-two",
                    Decide: o => o.Generation >= 2 ? EvolutionPipelineDecision.Stop("limite") : EvolutionPipelineDecision.Continue),
            });

        var result = new EvolutionPipelineRunner<double>().Run(definition, "run-stop");

        Assert.Equal(EvolutionPipelineTermination.PolicyStop, result.Termination);
        Assert.Equal("limite", result.Reason);
    }

    [Fact]
    public void Transition_to_unknown_state_is_rejected()
    {
        var definition = new EvolutionPipelineDefinition<double>(
            "pipe-bad",
            "main",
            new[]
            {
                new EvolutionPipelineState<double>(
                    "main",
                    Problem(),
                    Options(maxGenerations: 50),
                    DecisionId: "to-missing",
                    Decide: _ => EvolutionPipelineDecision.Transition("inexistente", "teste")),
            });

        Assert.ThrowsAny<Exception>(() => new EvolutionPipelineRunner<double>().Run(definition, "run-bad"));
    }

    [Fact]
    public void Worker_count_does_not_change_pipeline_result()
    {
        EvolutionPipelineResult<double> Run(int workers)
        {
            var definition = new EvolutionPipelineDefinition<double>(
                "pipe-workers",
                "main",
                new[] { new EvolutionPipelineState<double>("main", Problem(), Options(maxGenerations: 4, workers: workers)) });
            return new EvolutionPipelineRunner<double>().Run(definition, "run-w" + workers);
        }

        var one = Run(1);
        var two = Run(2);

        Assert.Equal(one.BestFitness.Value, two.BestFitness.Value);
        Assert.Equal(one.GenerationsRun, two.GenerationsRun);
    }
}
