using Evo.Core.Functions;

namespace Evo.Core.Compiled;

/// <summary>Kernel opcode of an active node on the SIMD path.</summary>
internal enum SimdOp : byte
{
    Unsupported = 0,
    Constant,
    Add,
    Subtract,
    Multiply,
    Negate,
    Abs,
    Square,
    Min,
    Max,
}

/// <summary>
/// Typed compiled plan of the active closure of one genome (02 section 3.1): one instruction per active
/// node, in ascending node order, executed once per call. Shared nodes are never expanded as a tree.
/// Immutable after construction and safe to share between calls.
/// </summary>
internal sealed class CompiledGenomePlan<T>
{
    private readonly IFunction<T>[] _functions;
    private readonly int[] _nodes;
    private readonly int[] _destinations;
    private readonly int[] _arities;
    private readonly int[] _arguments;
    private readonly T[] _parameters;
    private readonly SimdOp[] _simdOps;
    private readonly int[] _outputs;

    private CompiledGenomePlan(
        GenomeShape shape,
        IFunction<T>[] functions,
        int[] nodes,
        int[] destinations,
        int[] arities,
        int[] arguments,
        T[] parameters,
        SimdOp[] simdOps,
        int[] outputs,
        string? simdFallbackReason)
    {
        InputCount = shape.InputCount;
        NodeCount = shape.NodeCount;
        OutputCount = shape.OutputCount;
        MaxArity = shape.MaxArity;
        _functions = functions;
        _nodes = nodes;
        _destinations = destinations;
        _arities = arities;
        _arguments = arguments;
        _parameters = parameters;
        _simdOps = simdOps;
        _outputs = outputs;
        SimdFallbackReason = simdFallbackReason;
    }

    /// <summary>Number of inputs of the genome shape.</summary>
    public int InputCount { get; }

    /// <summary>Number of nodes of the genome shape.</summary>
    public int NodeCount { get; }

    /// <summary>Number of outputs of the genome shape.</summary>
    public int OutputCount { get; }

    /// <summary>MaxArity of the genome shape (stride of the argument table).</summary>
    public int MaxArity { get; }

    /// <summary>Number of value slots: inputs followed by nodes.</summary>
    public int SlotCount => InputCount + NodeCount;

    /// <summary>Number of active nodes (instructions).</summary>
    public int InstructionCount => _nodes.Length;

    /// <summary>Null when every active node has a SIMD kernel; otherwise the explicit reason for scalar fallback.</summary>
    public string? SimdFallbackReason { get; }

    /// <summary>True when the whole active closure can run on the SIMD path.</summary>
    public bool SimdEligible => SimdFallbackReason is null;

    internal int[] Nodes => _nodes;

    internal int[] Destinations => _destinations;

    internal int[] Arguments => _arguments;

    internal T[] Parameters => _parameters;

    internal SimdOp[] SimdOps => _simdOps;

    internal int[] OutputAddresses => _outputs;

    /// <summary>Builds the plan of the active closure of <paramref name="genome"/>.</summary>
    internal static CompiledGenomePlan<T> Compile(
        Genome<T> genome,
        IFunctionSet<T> functions,
        IValueSemantics<T> values)
    {
        var shape = genome.Shape;
        var active = genome.Analyze(functions).ActiveNodes;
        var count = active.Length;
        var arguments = new int[count * shape.MaxArity];
        var nodeFunctions = new IFunction<T>[count];
        var nodes = new int[count];
        var destinations = new int[count];
        var arities = new int[count];
        var parameters = new T[count];
        var simdOps = new SimdOp[count];

        var simdSet = SupportsSimdSet(functions.Id);
        string? reason = simdSet ? null : $"Função de valor '{values.ValueTypeId}' ou conjunto '{functions.Id}' sem kernel SIMD.";
        for (var k = 0; k < count; k++)
        {
            var node = active[k];
            var function = functions.Get(genome.FunctionGenes[node]);
            nodeFunctions[k] = function;
            nodes[k] = node;
            destinations[k] = shape.InputCount + node;
            arities[k] = function.Arity;
            for (var slot = 0; slot < function.Arity; slot++)
            {
                arguments[k * shape.MaxArity + slot] = genome.GetConnection(node, slot);
            }

            parameters[k] = values.Clone(genome.ParameterGenes[node]);
            simdOps[k] = simdSet ? MapSimdOp(function.Id) : SimdOp.Unsupported;
            if (reason is null && simdOps[k] == SimdOp.Unsupported)
            {
                reason = $"Função ativa '{function.Id}' do nó {node} não tem kernel SIMD; fallback explícito para compiled scalar.";
            }
        }

        return new CompiledGenomePlan<T>(
            shape,
            nodeFunctions,
            nodes,
            destinations,
            arities,
            arguments,
            parameters,
            simdOps,
            genome.OutputGenes.ToArray(),
            reason);
    }

    /// <summary>
    /// Executes the plan for one case on the scalar path. Each active node is computed once; the
    /// optional counter array receives one increment per executed node.
    /// </summary>
    internal void RunScalar(ReadOnlySpan<T> inputs, Span<T> outputs, int[]? executionCounts)
    {
        var slots = new T[SlotCount];
        inputs.CopyTo(slots);
        var args = new T[Math.Max(MaxArity, 1)];
        for (var k = 0; k < _functions.Length; k++)
        {
            var arity = _arities[k];
            var baseIndex = k * MaxArity;
            for (var a = 0; a < arity; a++)
            {
                args[a] = slots[_arguments[baseIndex + a]];
            }

            slots[_destinations[k]] = _functions[k].Invoke(args.AsSpan(0, arity), _parameters[k]);
            if (executionCounts is not null)
            {
                executionCounts[_nodes[k]]++;
            }
        }

        for (var o = 0; o < _outputs.Length; o++)
        {
            outputs[o] = slots[_outputs[o]];
        }
    }

    private static bool SupportsSimdSet(string functionSetId) =>
        (typeof(T) == typeof(double) && functionSetId == StandardDoubleFunctionSet.Instance.Id)
        || (typeof(T) == typeof(int) && functionSetId == StandardInt32FunctionSet.Instance.Id);

    private static SimdOp MapSimdOp(string id) => id switch
    {
        "Constant" => SimdOp.Constant,
        "Add" => SimdOp.Add,
        "Subtract" => SimdOp.Subtract,
        "Multiply" => SimdOp.Multiply,
        "Negate" => SimdOp.Negate,
        "Abs" => SimdOp.Abs,
        "Square" => SimdOp.Square,
        "Min" => SimdOp.Min,
        "Max" => SimdOp.Max,
        _ => SimdOp.Unsupported,
    };
}
