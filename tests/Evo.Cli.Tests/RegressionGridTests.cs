using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Xunit;

namespace Evo.Cli.Tests;

/// <summary>Testes da grade exploratória de regressão (comando <c>benchmark regression-grid</c>).</summary>
public sealed class RegressionGridTests
{
    [Fact]
    public void CatalogExposesThreeIncreasingLevelsPerFamily()
    {
        var problems = RegressionGridCatalog.All;

        Assert.Equal(problems.Select(p => p.Id).Distinct(StringComparer.Ordinal).Count(), problems.Count);
        foreach (var family in problems.GroupBy(p => p.Family, StringComparer.Ordinal))
        {
            Assert.Equal([1, 2, 3], family.Select(p => p.Level).OrderBy(level => level));
            Assert.Single(family.Select(p => (p.Definition.InputMinimum, p.Definition.InputMaximum)).Distinct());
        }

        Assert.Equal(5, problems.Select(p => p.Family).Distinct(StringComparer.Ordinal).Count());
    }

    [Fact]
    public void EveryTargetIsFiniteOverItsDomain()
    {
        foreach (var problem in RegressionGridCatalog.All)
        {
            var dataset = problem.Definition.Dataset;
            foreach (var input in dataset.TrainingInputs.Concat(dataset.ValidationInputs))
            {
                Assert.True(
                    double.IsFinite(problem.Definition.Target(input)),
                    $"{problem.Id} produziu valor não finito em x={input}");
            }
        }
    }

    [Fact]
    public void BaselinesShareTheEvaluationBudgetOfTheDefaultVariation()
    {
        var configurations = RegressionGridConfigurations.All;
        var baselines = configurations.Where(c => c.Kind == RegressionGridKind.Baseline).ToArray();
        var variations = configurations.Where(c => c.Kind == RegressionGridKind.Variation).ToArray();
        var @default = configurations.Single(c => c.Id == "evo-default");

        Assert.True(baselines.Length > 1, "é preciso mais de uma função de baseline.");
        Assert.Equal(3, variations.Length);
        Assert.Contains(baselines, c => c.Strategy == RegressionGridStrategy.RandomSearch);
        foreach (var baseline in baselines)
        {
            Assert.Equal(@default.Options.PopulationSize, baseline.Options.PopulationSize);
            Assert.Equal(@default.Options.MaxGenerations, baseline.Options.MaxGenerations);
        }
    }

    [Fact]
    public void AccuracyDigitsIsContinuousAndMatchesTheLegacyThreshold()
    {
        Assert.Equal(3.0, RegressionGridReport.AccuracyDigits(1e-3), 10);
        Assert.Equal(0.0, RegressionGridReport.AccuracyDigits(1.0), 10);
        Assert.True(RegressionGridReport.AccuracyDigits(5e-4) > RegressionGridReport.AccuracyDigits(1e-3));
        Assert.Equal(16.0, RegressionGridReport.AccuracyDigits(0.0));
    }

    [Fact]
    public void GridCommandWritesTidyReports()
    {
        var dir = Path.Combine(Path.GetTempPath(), "evo-grid-test-" + Guid.NewGuid().ToString("N"));
        try
        {
            Assert.Equal(0, Evo.Cli.Program.Main([
                "benchmark", "regression-grid",
                "--seeds", "1",
                "--stride", "1",
                "--scale", "100",
                "--out", dir]));

            var generations = File.ReadAllLines(Path.Combine(dir, "grid-generations.csv"));
            var runs = File.ReadAllLines(Path.Combine(dir, "grid-runs.csv"));
            var configurations = File.ReadAllLines(Path.Combine(dir, "grid-configurations.csv"));
            var problems = File.ReadAllLines(Path.Combine(dir, "grid-problems.csv"));

            var expectedRuns = RegressionGridConfigurations.All.Count * RegressionGridCatalog.All.Count;
            Assert.Equal(expectedRuns + 1, runs.Length);
            Assert.Equal(RegressionGridConfigurations.All.Count + 1, configurations.Length);
            Assert.Equal(RegressionGridCatalog.All.Count + 1, problems.Length);
            Assert.Contains("accuracyDigits", generations[0]);
            Assert.Contains("evaluations", generations[0]);
            Assert.Contains("populationMeanReuseEdgeRatio", generations[0]);
            Assert.True(File.Exists(Path.Combine(dir, "grid-summary.txt")));

            // Com --scale 100 o evo-gen400 ainda tem o dobro de gerações do evo-default.
            var rows = generations.Skip(1).Select(line => line.Split(',')).ToArray();
            int Generations(string experiment) => rows.Count(r => r[0] == experiment && r[4] == "poly-1");
            Assert.Equal(2 * (Generations("evo-default") - 1) + 1, Generations("evo-gen400"));
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
    public void GridCommandIsDeterministic()
    {
        var first = RunTinyGrid();
        var second = RunTinyGrid();
        Assert.Equal(first, second);
    }

    private static IReadOnlyList<string> RunTinyGrid()
    {
        var dir = Path.Combine(Path.GetTempPath(), "evo-grid-det-" + Guid.NewGuid().ToString("N"));
        try
        {
            Assert.Equal(0, Evo.Cli.Program.Main([
                "benchmark", "regression-grid",
                "--seeds", "1",
                "--stride", "1",
                "--scale", "200",
                "--out", dir]));

            // elapsedSeconds é a única coluna dependente de relógio e é removida.
            return File.ReadAllLines(Path.Combine(dir, "grid-runs.csv"))
                .Select(line => string.Join(',', line.Split(',').Where((_, index) => index != 12)))
                .ToArray();
        }
        finally
        {
            if (Directory.Exists(dir))
            {
                Directory.Delete(dir, recursive: true);
            }
        }
    }
}
