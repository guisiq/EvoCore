namespace Evo.Core;

/// <summary>
/// Opt-in execution counter of the reference evaluator, indexed by node index.
/// Pass no counters in timing runs (the default is <c>null</c>). Not thread-safe: use one instance per evaluating thread.
/// </summary>
internal sealed class ReferenceExecutionCounters
{
    /// <summary>Executions per node: one per active node per call; inactive nodes stay at zero.</summary>
    public int[] NodeExecutions { get; private set; } = [];

    /// <summary>Allocates the counter array for a genome with <paramref name="nodeCount"/> nodes if needed.</summary>
    internal void Prepare(int nodeCount)
    {
        if (NodeExecutions.Length != nodeCount)
        {
            NodeExecutions = new int[nodeCount];
        }
    }

    /// <summary>Sets every counter to zero.</summary>
    public void Reset() => Array.Clear(NodeExecutions);
}
