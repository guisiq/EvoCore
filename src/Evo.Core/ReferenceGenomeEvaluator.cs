namespace Evo.Core;

/// <summary>
/// Reference (oracle) evaluator. Computes each active node exactly once per call,
/// in ascending node order (topological, since connections only point backwards),
/// and reuses its value for every consumer and output. Inactive nodes are never computed.
/// </summary>
/// <remarks>
/// Clarity over speed: this class allocates its scratch buffers per call. Its
/// output is the correctness oracle for later compiled and SIMD evaluators.
/// </remarks>
/// <typeparam name="T">Logical value produced by a terminal/node.</typeparam>
public sealed class ReferenceGenomeEvaluator<T> : IGenomeEvaluator<T>
{
    private readonly IFunctionSet<T> _functions;
    private readonly ReferenceExecutionCounters? _counters;

    /// <summary>Creates a reference evaluator bound to a function set, optionally with execution counters.</summary>
    public ReferenceGenomeEvaluator(IFunctionSet<T> functions, ReferenceExecutionCounters? counters = null)
    {
        ArgumentNullException.ThrowIfNull(functions);
        _functions = functions;
        _counters = counters;
    }

    /// <inheritdoc />
    public void Evaluate(Genome<T> genome, ReadOnlySpan<T> inputs, Span<T> outputs)
    {
        ArgumentNullException.ThrowIfNull(genome);

        var shape = genome.Shape;
        if (inputs.Length != shape.InputCount)
        {
            throw new ArgumentException(
                $"Expected {shape.InputCount} inputs, got {inputs.Length}.", nameof(inputs));
        }

        if (outputs.Length != shape.OutputCount)
        {
            throw new ArgumentException(
                $"Expected {shape.OutputCount} output slots, got {outputs.Length}.", nameof(outputs));
        }

        var active = genome.Analyze(_functions).ActiveNodes;

        // Addresses: inputs in [0, InputCount), node n at InputCount + n.
        var slots = new T[shape.InputCount + shape.NodeCount];
        inputs.CopyTo(slots);

        var arguments = new T[Math.Max(shape.MaxArity, 1)];
        var parameters = genome.ParameterGenes;
        var functionGenes = genome.FunctionGenes;

        // Single null check outside the per-node loop; disabled counters add no per-node work.
        var nodeExecutions = _counters is null ? null : PrepareCounters(_counters, shape.NodeCount);

        foreach (var node in active)
        {
            var function = _functions.Get(functionGenes[node]);
            var arity = function.Arity;
            for (var slot = 0; slot < arity; slot++)
            {
                arguments[slot] = slots[genome.GetConnection(node, slot)];
            }

            slots[shape.InputCount + node] = function.Invoke(arguments.AsSpan(0, arity), parameters[node]);
            if (nodeExecutions is not null)
            {
                nodeExecutions[node]++;
            }
        }

        var outputGenes = genome.OutputGenes;
        for (var output = 0; output < outputs.Length; output++)
        {
            outputs[output] = slots[outputGenes[output]];
        }
    }

    private static int[] PrepareCounters(ReferenceExecutionCounters counters, int nodeCount)
    {
        counters.Prepare(nodeCount);
        return counters.NodeExecutions;
    }
}
