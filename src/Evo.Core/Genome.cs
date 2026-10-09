using System.Security.Cryptography;
using System.Text;

namespace Evo.Core;

/// <summary>
/// Raised when a genome gene, shape or imported gene vector violates the DAG or
/// layout rules of 02-dag-e-evolucao.md.
/// </summary>
public sealed class InvalidGenomeException : Exception
{
    /// <summary>Creates the exception with a precise description of the violated rule.</summary>
    public InvalidGenomeException(string message)
        : base(message)
    {
    }
}

/// <summary>
/// Active graph metadata of a genome for one function set (02-dag-e-evolucao.md section 3).
/// </summary>
/// <param name="ActiveNodeCount">Number of distinct active nodes (each address counted once).</param>
/// <param name="ActiveDepth">Longest path, in node levels, from an input to an output.</param>
/// <param name="UsedInputs">Ascending indices of the inputs reachable from the outputs.</param>
/// <param name="ActiveNodes">Ascending node indices of the active closure (topological order).</param>
public sealed record ActiveGraph(int ActiveNodeCount, int ActiveDepth, int[] UsedInputs, int[] ActiveNodes);

/// <summary>
/// CGP genome stored as four logical gene vectors:
/// <c>FunctionGenes[NodeCount]</c>, <c>ConnectionGenes[NodeCount * MaxArity]</c>,
/// <c>ParameterGenes[NodeCount]</c> and <c>OutputGenes[OutputCount]</c>.
/// Normative in 02-dag-e-evolucao.md sections 1–3.
/// </summary>
/// <remarks>
/// Node <c>n</c>, argument <c>a</c> is stored at <c>ConnectionGenes[n * MaxArity + a]</c>.
/// Addresses: inputs occupy <c>[0, InputCount)</c>; node <c>n</c> is <c>InputCount + n</c>.
/// Every connection slot (active or not) must reference an input or an earlier node within
/// <c>LevelsBack</c>; self, future and out-of-range addresses are rejected.
/// Instances are owned by a single worker at a time.
/// </remarks>
/// <typeparam name="T">Logical value produced by a terminal/node.</typeparam>
public sealed class Genome<T>
{
    /// <summary>Version tag of the canonical phenotype fingerprint encoding.</summary>
    public const string PhenotypeFingerprintVersion = "evo-phenotype-fingerprint/v1";

    private readonly int[] _functionGenes;
    private readonly int[] _connectionGenes;
    private readonly T[] _parameterGenes;
    private readonly int[] _outputGenes;

    /// <summary>Creates a genome of the given shape with all genes zeroed (parameters set to <c>default</c>).</summary>
    public Genome(GenomeShape shape)
    {
        ValidateShape(shape);
        Shape = shape;
        _functionGenes = new int[shape.NodeCount];
        _connectionGenes = new int[shape.NodeCount * shape.MaxArity];
        _parameterGenes = new T[shape.NodeCount];
        _outputGenes = new int[shape.OutputCount];
    }

    /// <summary>Fixed shape of this genome.</summary>
    public GenomeShape Shape { get; }

    /// <summary>Function gene ids, one per node.</summary>
    public ReadOnlySpan<int> FunctionGenes => _functionGenes;

    /// <summary>Connection genes, <c>NodeCount * MaxArity</c> entries in node-major order.</summary>
    public ReadOnlySpan<int> ConnectionGenes => _connectionGenes;

    /// <summary>Parameter genes, one per node.</summary>
    public ReadOnlySpan<T> ParameterGenes => _parameterGenes;

    /// <summary>Output gene addresses, one per output.</summary>
    public ReadOnlySpan<int> OutputGenes => _outputGenes;

    /// <summary>Returns the connection address of node <paramref name="node"/>, argument <paramref name="slot"/>.</summary>
    public int GetConnection(int node, int slot)
    {
        CheckNode(node);
        CheckSlot(slot);
        return _connectionGenes[ConnectionIndex(node, slot)];
    }

    /// <summary>Sets a function gene. The id is checked against the function set by <see cref="Validate"/>.</summary>
    public void SetFunctionGene(int node, int functionId)
    {
        CheckNode(node);
        if (functionId < 0)
        {
            throw new InvalidGenomeException($"Function gene of node {node} must be >= 0, got {functionId}.");
        }

        _functionGenes[node] = functionId;
    }

    /// <summary>Sets a connection gene after checking the DAG and LevelsBack rules.</summary>
    public void SetConnectionGene(int node, int slot, int address)
    {
        CheckNode(node);
        CheckSlot(slot);
        CheckConnectionAddress(node, address);
        _connectionGenes[ConnectionIndex(node, slot)] = address;
    }

    /// <summary>Sets a parameter gene. The value is stored as given; callers own deep-copy semantics.</summary>
    public void SetParameterGene(int node, T value)
    {
        CheckNode(node);
        _parameterGenes[node] = value;
    }

    /// <summary>Sets an output gene after checking that it addresses an input or a node.</summary>
    public void SetOutputGene(int output, int address)
    {
        if (output < 0 || output >= Shape.OutputCount)
        {
            throw new InvalidGenomeException($"Output index {output} is out of range [0, {Shape.OutputCount}).");
        }

        if (address < 0 || address >= Shape.InputCount + Shape.NodeCount)
        {
            throw new InvalidGenomeException(
                $"Output gene {output} address {address} is out of range [0, {Shape.InputCount + Shape.NodeCount}).");
        }

        _outputGenes[output] = address;
    }

    /// <summary>
    /// Builds a genome from imported gene vectors. Arrays are copied and parameter values are
    /// deep-cloned through <paramref name="values"/>; the result is fully validated.
    /// </summary>
    public static Genome<T> FromGenes(
        GenomeShape shape,
        ReadOnlySpan<int> functionGenes,
        ReadOnlySpan<int> connectionGenes,
        ReadOnlySpan<T> parameterGenes,
        ReadOnlySpan<int> outputGenes,
        IValueSemantics<T> values,
        IFunctionSet<T> functions)
    {
        ArgumentNullException.ThrowIfNull(values);
        ArgumentNullException.ThrowIfNull(functions);

        var genome = new Genome<T>(shape);
        if (functionGenes.Length != shape.NodeCount)
        {
            throw new InvalidGenomeException($"FunctionGenes length {functionGenes.Length} must be NodeCount {shape.NodeCount}.");
        }

        if (connectionGenes.Length != shape.NodeCount * shape.MaxArity)
        {
            throw new InvalidGenomeException(
                $"ConnectionGenes length {connectionGenes.Length} must be NodeCount*MaxArity {shape.NodeCount * shape.MaxArity}.");
        }

        if (parameterGenes.Length != shape.NodeCount)
        {
            throw new InvalidGenomeException($"ParameterGenes length {parameterGenes.Length} must be NodeCount {shape.NodeCount}.");
        }

        if (outputGenes.Length != shape.OutputCount)
        {
            throw new InvalidGenomeException($"OutputGenes length {outputGenes.Length} must be OutputCount {shape.OutputCount}.");
        }

        functionGenes.CopyTo(genome._functionGenes);
        connectionGenes.CopyTo(genome._connectionGenes);
        outputGenes.CopyTo(genome._outputGenes);
        for (var i = 0; i < parameterGenes.Length; i++)
        {
            genome._parameterGenes[i] = parameterGenes[i] is null ? default! : values.Clone(parameterGenes[i]);
        }

        genome.Validate(functions);
        return genome;
    }

    /// <summary>Returns a deep copy of this genome (parameters cloned through <paramref name="values"/>).</summary>
    public Genome<T> Clone(IValueSemantics<T> values)
    {
        ArgumentNullException.ThrowIfNull(values);
        var copy = new Genome<T>(Shape);
        _functionGenes.CopyTo(copy._functionGenes, 0);
        _connectionGenes.CopyTo(copy._connectionGenes, 0);
        _outputGenes.CopyTo(copy._outputGenes, 0);
        for (var i = 0; i < _parameterGenes.Length; i++)
        {
            copy._parameterGenes[i] = _parameterGenes[i] is null ? default! : values.Clone(_parameterGenes[i]);
        }

        return copy;
    }

    /// <summary>
    /// Validates every gene: function ids against <paramref name="functions"/>, every connection slot
    /// (active or inactive), output addresses and presence of parameter values.
    /// </summary>
    public void Validate(IFunctionSet<T> functions)
    {
        ArgumentNullException.ThrowIfNull(functions);

        for (var node = 0; node < Shape.NodeCount; node++)
        {
            var functionId = _functionGenes[node];
            if (functionId < 0 || functionId >= functions.Count)
            {
                throw new InvalidGenomeException(
                    $"Node {node} function id {functionId} is outside the function set [0, {functions.Count}).");
            }

            if (_parameterGenes[node] is null)
            {
                throw new InvalidGenomeException($"Node {node} has a null parameter gene.");
            }

            for (var slot = 0; slot < Shape.MaxArity; slot++)
            {
                CheckConnectionAddress(node, _connectionGenes[ConnectionIndex(node, slot)]);
            }
        }

        foreach (var address in _outputGenes)
        {
            if (address < 0 || address >= Shape.InputCount + Shape.NodeCount)
            {
                throw new InvalidGenomeException(
                    $"Output address {address} is out of range [0, {Shape.InputCount + Shape.NodeCount}).");
            }
        }
    }

    /// <summary>
    /// Computes the active closure of the outputs, the depth and the used inputs. Each active address
    /// is counted once; inactive genes are ignored. Only the first <c>Arity</c> connections of a node are active.
    /// </summary>
    public ActiveGraph Analyze(IFunctionSet<T> functions)
    {
        ArgumentNullException.ThrowIfNull(functions);

        var inputCount = Shape.InputCount;
        var nodeCount = Shape.NodeCount;
        var active = new bool[nodeCount];
        var arity = new int[nodeCount];
        for (var node = 0; node < nodeCount; node++)
        {
            arity[node] = ArityOf(node, functions);
        }

        foreach (var address in _outputGenes)
        {
            if (address >= inputCount)
            {
                active[address - inputCount] = true;
            }
        }

        // Connections only point backwards, so a descending sweep closes the active set in one pass.
        for (var node = nodeCount - 1; node >= 0; node--)
        {
            if (!active[node])
            {
                continue;
            }

            for (var slot = 0; slot < arity[node]; slot++)
            {
                var address = _connectionGenes[ConnectionIndex(node, slot)];
                if (address >= inputCount)
                {
                    active[address - inputCount] = true;
                }
            }
        }

        var depth = new int[nodeCount];
        var activeNodes = new List<int>();
        for (var node = 0; node < nodeCount; node++)
        {
            if (!active[node])
            {
                continue;
            }

            activeNodes.Add(node);
            var maxArg = 0;
            for (var slot = 0; slot < arity[node]; slot++)
            {
                maxArg = Math.Max(maxArg, DepthOf(_connectionGenes[ConnectionIndex(node, slot)], depth, inputCount));
            }

            depth[node] = maxArg + 1;
        }

        var activeDepth = 0;
        var usedInputs = new SortedSet<int>();
        foreach (var address in _outputGenes)
        {
            activeDepth = Math.Max(activeDepth, DepthOf(address, depth, inputCount));
            if (address < inputCount)
            {
                usedInputs.Add(address);
            }
        }

        foreach (var node in activeNodes)
        {
            for (var slot = 0; slot < arity[node]; slot++)
            {
                var address = _connectionGenes[ConnectionIndex(node, slot)];
                if (address < inputCount)
                {
                    usedInputs.Add(address);
                }
            }
        }

        return new ActiveGraph(activeNodes.Count, activeDepth, usedInputs.ToArray(), activeNodes.ToArray());
    }

    /// <summary>
    /// Computes the SHA-256 canonical fingerprint of the active phenotype: encoding version, shape,
    /// function set id, outputs, and for each active node in topological order its index, function id,
    /// effectively used connections and parameter (via <see cref="IValueSemantics{T}.AppendFingerprint"/>).
    /// Inactive genes do not contribute.
    /// </summary>
    public byte[] ComputePhenotypeFingerprint(IFunctionSet<T> functions, IValueSemantics<T> values)
    {
        ArgumentNullException.ThrowIfNull(functions);
        ArgumentNullException.ThrowIfNull(values);

        var graph = Analyze(functions);
        using var hash = IncrementalHash.CreateHash(HashAlgorithmName.SHA256);

        AppendString(hash, PhenotypeFingerprintVersion);
        AppendInt(hash, Shape.InputCount);
        AppendInt(hash, Shape.NodeCount);
        AppendInt(hash, Shape.OutputCount);
        AppendInt(hash, Shape.MaxArity);
        AppendInt(hash, Shape.LevelsBack);
        AppendString(hash, functions.Id);

        AppendInt(hash, _outputGenes.Length);
        foreach (var address in _outputGenes)
        {
            AppendInt(hash, address);
        }

        AppendInt(hash, graph.ActiveNodeCount);
        foreach (var node in graph.ActiveNodes)
        {
            var functionId = _functionGenes[node];
            var function = functions.Get(functionId);
            AppendInt(hash, node);
            AppendInt(hash, functionId);
            AppendInt(hash, function.Arity);
            for (var slot = 0; slot < function.Arity; slot++)
            {
                AppendInt(hash, _connectionGenes[ConnectionIndex(node, slot)]);
            }

            values.AppendFingerprint(_parameterGenes[node], hash);
        }

        return hash.GetHashAndReset();
    }

    private static void ValidateShape(GenomeShape shape)
    {
        if (shape.InputCount <= 0)
        {
            throw new InvalidGenomeException($"InputCount must be > 0, got {shape.InputCount}.");
        }

        if (shape.NodeCount <= 0)
        {
            throw new InvalidGenomeException($"NodeCount must be > 0, got {shape.NodeCount}.");
        }

        if (shape.OutputCount <= 0)
        {
            throw new InvalidGenomeException($"OutputCount must be > 0, got {shape.OutputCount}.");
        }

        if (shape.MaxArity <= 0)
        {
            throw new InvalidGenomeException($"MaxArity must be > 0, got {shape.MaxArity}.");
        }

        if (shape.LevelsBack <= 0)
        {
            throw new InvalidGenomeException($"LevelsBack must be > 0, got {shape.LevelsBack}.");
        }

        if ((long)shape.NodeCount * shape.MaxArity > int.MaxValue)
        {
            throw new InvalidGenomeException("NodeCount * MaxArity exceeds the supported gene vector size.");
        }
    }

    private void CheckConnectionAddress(int node, int address)
    {
        var inputCount = Shape.InputCount;
        if (address < 0)
        {
            throw new InvalidGenomeException($"Node {node} connection address {address} must be >= 0.");
        }

        if (address < inputCount)
        {
            return;
        }

        var target = address - inputCount;
        if (target >= node)
        {
            throw new InvalidGenomeException(
                $"Node {node} connection to node {target} is self or future; only earlier nodes are allowed.");
        }

        if (target < node - Shape.LevelsBack)
        {
            throw new InvalidGenomeException(
                $"Node {node} connection to node {target} violates LevelsBack={Shape.LevelsBack}.");
        }
    }

    private int ArityOf(int node, IFunctionSet<T> functions)
    {
        var functionId = _functionGenes[node];
        if (functionId < 0 || functionId >= functions.Count)
        {
            throw new InvalidGenomeException(
                $"Node {node} function id {functionId} is outside the function set [0, {functions.Count}).");
        }

        var arity = functions.Get(functionId).Arity;
        if (arity < 0 || arity > Shape.MaxArity)
        {
            throw new InvalidGenomeException(
                $"Function {functionId} arity {arity} is outside [0, MaxArity={Shape.MaxArity}].");
        }

        return arity;
    }

    private static int DepthOf(int address, int[] nodeDepth, int inputCount) =>
        address < inputCount ? 0 : nodeDepth[address - inputCount];

    private int ConnectionIndex(int node, int slot) => node * Shape.MaxArity + slot;

    private void CheckNode(int node)
    {
        if (node < 0 || node >= Shape.NodeCount)
        {
            throw new InvalidGenomeException($"Node index {node} is out of range [0, {Shape.NodeCount}).");
        }
    }

    private void CheckSlot(int slot)
    {
        if (slot < 0 || slot >= Shape.MaxArity)
        {
            throw new InvalidGenomeException($"Connection slot {slot} is out of range [0, {Shape.MaxArity}).");
        }
    }

    private static void AppendInt(IncrementalHash hash, int value)
    {
        Span<byte> buffer = stackalloc byte[sizeof(int)];
        System.Buffers.Binary.BinaryPrimitives.WriteInt32LittleEndian(buffer, value);
        hash.AppendData(buffer);
    }

    private static void AppendString(IncrementalHash hash, string value)
    {
        var bytes = Encoding.UTF8.GetBytes(value);
        AppendInt(hash, bytes.Length);
        hash.AppendData(bytes);
    }
}
