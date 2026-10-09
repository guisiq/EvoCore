namespace Evo.Core.Compiled;

/// <summary>
/// Opt-in execution counters of the compiled and SIMD paths, indexed by node index (not by address).
/// Pass no counters in timing runs. Not thread-safe: use one instance per evaluating thread.
/// </summary>
public sealed class CompiledExecutionCounters
{
    /// <summary>Scalar executions per node: one per active node per case on the scalar path.</summary>
    public int[] ScalarExecutions { get; private set; } = [];

    /// <summary>SIMD kernel executions per node: one per active node per vector block.</summary>
    public int[] KernelExecutions { get; private set; } = [];

    /// <summary>Scalar executions per node for the tail cases (cases that do not fill a vector block).</summary>
    public int[] TailExecutions { get; private set; } = [];

    /// <summary>Allocates the counter arrays for a genome with <paramref name="nodeCount"/> nodes if needed.</summary>
    internal void Prepare(int nodeCount)
    {
        if (ScalarExecutions.Length == nodeCount)
        {
            return;
        }

        ScalarExecutions = new int[nodeCount];
        KernelExecutions = new int[nodeCount];
        TailExecutions = new int[nodeCount];
    }

    /// <summary>Sets every counter to zero.</summary>
    public void Reset()
    {
        Array.Clear(ScalarExecutions);
        Array.Clear(KernelExecutions);
        Array.Clear(TailExecutions);
    }
}
