using System.IO;
using Xunit;

namespace Evo.Cli.Tests;

/// <summary>Testes rápidos dos comandos da demo (execução curta, sem benchmark de 30 seeds).</summary>
public sealed class DemoCommandTests
{
    [Fact]
    public void DemoWritesArtifactsAndReturnsZero()
    {
        var dir = Path.Combine(Path.GetTempPath(), "evo-cli-test-" + Guid.NewGuid().ToString("N"));
        try
        {
            var code = Evo.Cli.Program.Main(["demo", "symbolic-regression", "--seed", "42", "--generations", "3", "--population", "20", "--out", dir]);
            Assert.Equal(0, code);
            foreach (var name in new[] { "config.json", "history.csv", "best-genome.json", "best-program.txt", "checkpoint.json" })
            {
                Assert.True(File.Exists(Path.Combine(dir, name)), name);
            }

            Assert.Equal(0, Evo.Cli.Program.Main(["inspect", Path.Combine(dir, "best-genome.json")]));
            Assert.Equal(0, Evo.Cli.Program.Main(["resume", Path.Combine(dir, "checkpoint.json"), "--out", Path.Combine(dir, "resumed")]));
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
    public void ResumeProducesSameArtifactsAsContinuousRun()
    {
        var root = Path.Combine(Path.GetTempPath(), "evo-cli-resume-" + Guid.NewGuid().ToString("N"));
        try
        {
            var continuous = Path.Combine(root, "continuous");
            var partial = Path.Combine(root, "partial");
            var resumed = Path.Combine(root, "resumed");
            Assert.Equal(0, Evo.Cli.Program.Main(["demo", "symbolic-regression", "--seed", "42", "--generations", "3", "--population", "20", "--out", continuous]));
            Assert.Equal(0, Evo.Cli.Program.Main(["demo", "symbolic-regression", "--seed", "42", "--generations", "1", "--population", "20", "--out", partial]));
            Assert.Equal(0, Evo.Cli.Program.Main(["resume", Path.Combine(partial, "checkpoint.json"), "--generations", "3", "--out", resumed]));

            Assert.Equal(File.ReadAllBytes(Path.Combine(continuous, "best-genome.json")), File.ReadAllBytes(Path.Combine(resumed, "best-genome.json")));
            Assert.Equal(File.ReadAllBytes(Path.Combine(continuous, "history.csv")), File.ReadAllBytes(Path.Combine(resumed, "history.csv")));
        }
        finally
        {
            if (Directory.Exists(root))
            {
                Directory.Delete(root, recursive: true);
            }
        }
    }

    [Fact]
    public void InvalidNumericArgumentReturnsUsageCode()
    {
        Assert.Equal(2, Evo.Cli.Program.Main(["demo", "symbolic-regression", "--seed", "abc"]));
        Assert.Equal(2, Evo.Cli.Program.Main(["demo", "symbolic-regression", "--population", "99999999999999999999"]));
    }

    [Fact]
    public void DatasetIsFixedWith201PointsAndParitySplit()
    {
        Assert.Equal(201, Evo.Cli.SymbolicRegressionDataset.Inputs.Length);
        Assert.Equal(-1.0, Evo.Cli.SymbolicRegressionDataset.Inputs[0]);
        Assert.Equal(101, Evo.Cli.SymbolicRegressionDataset.TrainingIndices.Length);
        Assert.Equal(100, Evo.Cli.SymbolicRegressionDataset.ValidationIndices.Length);
    }
}
