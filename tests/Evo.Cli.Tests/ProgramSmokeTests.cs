using System.Linq;
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
}
