namespace Evo.Benchmarks;

/// <summary>
/// Benchmark host. The measured baseline (individuals/s, active nodes/s, time per
/// generation, allocations and effective backend) is implemented by EVO-010.
/// </summary>
public static class Program
{
    /// <summary>Reports that no benchmark is registered yet.</summary>
    public static int Main()
    {
        Console.WriteLine("Evo.Benchmarks — baseline benchmarks are implemented in EVO-010.");
        return 0;
    }
}
