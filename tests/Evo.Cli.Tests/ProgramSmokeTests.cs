using System.Linq;
using Evo.Core;
using Evo.Core.Functions;
using Xunit;

namespace Evo.Cli.Tests;

/// <summary>Smoke tests of the CLI host scaffolded by EVO-001.</summary>
public sealed class ProgramSmokeTests
{
    [Fact]
    public void CliAssemblyReferencesCore()
    {
        var referenced = typeof(Evo.Cli.Program).Assembly
            .GetReferencedAssemblies()
            .Select(a => a.Name)
            .ToArray();

        Assert.Contains("Evo.Core", referenced);
    }

    [Fact]
    public void CliEntryPointRunsWithoutArguments()
    {
        Assert.Equal(0, Evo.Cli.Program.Main([]));
    }

    [Fact]
    public void UnknownCommandIsRejected()
    {
        Assert.Equal(2, Evo.Cli.Program.Main(["inspect"]));
    }

    [Fact]
    public void RegressionSuiteWritesGenerationAndReuseReports()
    {
        var dir = Path.Combine(Path.GetTempPath(), "evo-suite-test-" + Guid.NewGuid().ToString("N"));
        try
        {
            Assert.Equal(0, Evo.Cli.Program.Main([
                "benchmark", "symbolic-regression-suite",
                "--generations", "1",
                "--population", "8",
                "--workers", "1",
                "--out", dir]));

            var summary = File.ReadAllLines(Path.Combine(dir, "suite-results.csv"));
            var history = File.ReadAllLines(Path.Combine(dir, "evolution-history.csv"));
            var generationSummary = File.ReadAllLines(Path.Combine(dir, "generation-summary.csv"));
            var problemSummary = File.ReadAllLines(Path.Combine(dir, "problem-summary.csv"));
            Assert.Equal(181, summary.Length);
            Assert.Equal(361, history.Length);
            Assert.Equal(13, generationSummary.Length);
            Assert.Equal(7, problemSummary.Length);
            Assert.Contains("bestReusedActiveNodes", summary[0]);
            Assert.Contains("populationReuseEdgeRatioDelta", summary[0]);
            Assert.Contains("bestValidationMse", history[0]);
            Assert.Contains("populationMeanReuseEdgeRatio", history[0]);
            Assert.Contains("meanPopulationReuseEdgeRatio", generationSummary[0]);
            Assert.Contains("meanPopulationReuseEdgeRatioDelta", problemSummary[0]);
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
    public void RegressionSuiteUsesPublishedNguyenProblemsAndDomains()
    {
        var problems = RegressionSuiteCatalog.All;

        Assert.Equal(
            ["nguyen-1", "nguyen-2", "nguyen-3", "nguyen-4", "nguyen-7", "nguyen-8"],
            problems.Select(problem => problem.Id));
        Assert.All(problems.Take(4), problem =>
        {
            Assert.Equal(-1.0, problem.InputMinimum);
            Assert.Equal(1.0, problem.InputMaximum);
        });
        Assert.Equal((0.0, 2.0), (problems[4].InputMinimum, problems[4].InputMaximum));
        Assert.Equal((0.0, 4.0), (problems[5].InputMinimum, problems[5].InputMaximum));
        Assert.All(problems, problem =>
        {
            Assert.Equal(101, problem.Dataset.TrainingInputs.Length);
            Assert.Equal(100, problem.Dataset.ValidationInputs.Length);
        });
    }

    [Fact]
    public void DagReuseMetricsCountOnlyActiveSemanticReferences()
    {
        var genome = new Genome<double>(
            new GenomeShape(InputCount: 1, NodeCount: 3, OutputCount: 1, MaxArity: 2, LevelsBack: 3));
        for (var node = 0; node < 3; node++)
        {
            genome.SetFunctionGene(node, 0);
        }

        genome.SetConnectionGene(0, 0, 0);
        genome.SetConnectionGene(0, 1, 0);
        genome.SetConnectionGene(1, 0, 1);
        genome.SetConnectionGene(1, 1, 1);
        genome.SetConnectionGene(2, 0, 1);
        genome.SetConnectionGene(2, 1, 2);
        genome.SetOutputGene(0, 3);

        var metrics = DagReuseMetrics.Measure(genome, StandardDoubleFunctionSet.Instance);

        Assert.Equal(3, metrics.ActiveNodes);
        Assert.Equal(1, metrics.ReusedActiveNodes);
        Assert.Equal(2, metrics.ReuseEdges);
        Assert.Equal(5, metrics.SemanticReferences);
        Assert.Equal(1.0 / 3.0, metrics.ReusedActiveNodeRatio);
        Assert.Equal(2.0 / 5.0, metrics.ReuseEdgeRatio);
    }
}
