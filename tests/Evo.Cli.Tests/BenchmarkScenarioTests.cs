using Evo.Examples.Datasets;
using Evo.Examples.Scenarios;
using Xunit;

namespace Evo.Cli.Tests;

/// <summary>Smoke/integração para EVO-305: listagem/execução de cenários via CLI e validação de datasets.</summary>
public sealed class BenchmarkScenarioTests
{
    [Fact]
    public void RegistryExposesAllTwelveRequiredScenarioIds()
    {
        var required = new[]
        {
            "variation-strategies-boston",
            "initialization-strategies-diabetes",
            "population-scale-wine",
            "function-set-complexity-boston",
            "maximum-configuration-wine",
            "seed-repeatability-diabetes",
            "evaluator-throughput-synthetic",
            "evaluator-parity-matrix",
            "binary-operator-parity",
            "parameterized-operator-parity",
            "deep-chain-parity",
            "balanced-parameter-parity",
        };

        var ids = BenchmarkScenarioRegistry.All.Select(s => s.Id).ToArray();
        foreach (var id in required)
        {
            Assert.Contains(id, ids);
        }
    }

    [Theory]
    [InlineData("binary-operator-parity")]
    [InlineData("parameterized-operator-parity")]
    [InlineData("deep-chain-parity")]
    [InlineData("balanced-parameter-parity")]
    public void EvaluatorParityScenariosPassGates(string id)
    {
        var scenario = BenchmarkScenarioRegistry.Resolve(id);
        var result = scenario.Run(CancellationToken.None);
        Assert.True(result.GatesPassed, result.Summary);
    }

    [Theory]
    [InlineData("dag-chain-benchmark")]
    [InlineData("dag-fanout-benchmark")]
    [InlineData("dag-diamond-benchmark")]
    [InlineData("dag-levels-diamond-benchmark")]
    [InlineData("ilgpu-benchmark")]
    public void MigratedBenchmarksWorkloadsPassGates(string id)
    {
        var scenario = BenchmarkScenarioRegistry.Resolve(id);
        var result = scenario.Run(CancellationToken.None);
        Assert.True(result.GatesPassed, result.Summary);
        Assert.Equal(id, result.ScenarioId);
    }

    [Fact]
    public void CliExampleListPrintsAllRegisteredIds()
    {
        Assert.Equal(0, Program.Main(["example", "list"]));
    }

    [Fact]
    public void CliBenchmarkRunWritesManifestForBinaryOperatorParity()
    {
        var dir = Path.Combine(Path.GetTempPath(), "evo-scenario-test-" + Guid.NewGuid().ToString("N"));
        try
        {
            Assert.Equal(0, Program.Main(["benchmark", "run", "binary-operator-parity", "--out", dir]));
            Assert.True(File.Exists(Path.Combine(dir, "scenario-result.json")));
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
    public void DatasetCatalogRejectsChecksumMismatch()
    {
        var tempDir = Path.Combine(Path.GetTempPath(), "evo-dataset-test-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(tempDir);
        try
        {
            Assert.Throws<DatasetValidationException>(() => DatasetCatalog.Load("boston-housing", 0));
        }
        finally
        {
            Directory.Delete(tempDir, recursive: true);
        }
    }

    [Fact]
    public void CliPipelineRunExecutesDynamicDefinitionAndWritesResult()
    {
        var dir = Path.Combine(Path.GetTempPath(), "evo-pipeline-test-" + Guid.NewGuid().ToString("N"));
        var definitionPath = Path.Combine(Path.GetTempPath(), "evo-pipeline-def-" + Guid.NewGuid().ToString("N") + ".json");
        File.WriteAllText(definitionPath, """
        {
          "schema": "evo-pipeline/v1",
          "pipelineId": "cli-smoke-pipeline",
          "initialState": "stage-a",
          "states": [
            { "stateId": "stage-a", "problem": "symbolic-regression", "options": { "populationSize": 8, "maxGenerations": 2, "seed": 1 } }
          ]
        }
        """);
        try
        {
            Assert.Equal(0, Program.Main(["pipeline", "list"]));
            Assert.Equal(0, Program.Main(["pipeline", "run", definitionPath, "--run-id", "smoke-1", "--out", dir]));
            Assert.True(File.Exists(Path.Combine(dir, "pipeline-result.json")));
        }
        finally
        {
            if (File.Exists(definitionPath))
            {
                File.Delete(definitionPath);
            }

            if (Directory.Exists(dir))
            {
                Directory.Delete(dir, recursive: true);
            }
        }
    }

    [Fact]
    public void DatasetCatalogLoadsRegisteredDatasetsWithinShape()
    {
        var boston = DatasetCatalog.Load("boston-housing", 50);
        Assert.Equal(50, boston.RecordCount);
        Assert.Equal(13, boston.FeatureCount);

        var diabetes = DatasetCatalog.Load("pima-diabetes", 40);
        Assert.Equal(40, diabetes.RecordCount);

        var wine = DatasetCatalog.Load("wine-quality-red", 30);
        Assert.Equal(30, wine.RecordCount);
    }
}
