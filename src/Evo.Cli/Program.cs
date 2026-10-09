using Evo.Core;

namespace Evo.Cli;

/// <summary>
/// CLI entry point. Commands (<c>demo</c>, <c>inspect</c>, <c>resume</c>) are
/// implemented by EVO-009; EVO-001 only scaffolds the executable.
/// </summary>
public static class Program
{
    /// <summary>Prints the planned usage and exits.</summary>
    public static int Main(string[] args)
    {
        Console.WriteLine("Evo.Cli — commands are implemented in EVO-009.");
        Console.WriteLine("Planned usage:");
        Console.WriteLine("  demo symbolic-regression [--seed N] [--generations N] [--population N] [--workers N]");
        Console.WriteLine("  inspect <genome.json>");
        Console.WriteLine("  resume <checkpoint.json>");
        Console.WriteLine($"RNG contract: {RandomStream.Version}; default generations: {new EvolutionOptions().MaxGenerations}");
        return args.Length == 0 ? 0 : 2;
    }
}
