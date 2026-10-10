using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.OpenCL;
using Evo.Core;
using Evo.Core.Functions;

namespace Evo.Ilgpu;

/// <summary>
/// ILGPU evaluator of one genome for one case. Supports <c>double</c>, <c>int</c>, <c>double[]</c> and <c>int[]</c>
/// with the standard function sets. The active DAG is lowered to an opcode program executed by an ILGPU kernel;
/// array operands are element lanes (each lane runs the same program on one element). Anything the kernel does not
/// cover (custom function sets, mismatched array lengths, unsupported types) is evaluated by the fallback evaluator
/// and the reason is recorded in <see cref="LastFallbackReason"/>.
/// </summary>
/// <remarks>Each worker owns its instance; accelerators and buffers are not shared between threads.</remarks>
public sealed class IlgpuGenomeEvaluator<T> : IGenomeEvaluator<T>, IDisposable
{
    private const int OpAdd = 1;
    private const int OpSubtract = 2;
    private const int OpMultiply = 3;
    private const int OpProtectedDivide = 4;
    private const int OpNegate = 5;
    private const int OpAbs = 6;
    private const int OpSquare = 7;
    private const int OpProtectedSqrt = 8;
    private const int OpProtectedLog = 9;
    private const int OpMin = 10;
    private const int OpMax = 11;
    private const int OpGreaterThan = 12;
    private const int OpSelect = 13;
    private const int OpConstant = 14;
    private const int Stride = 5;

    private readonly IFunctionSet<T> _functions;
    private readonly IGenomeEvaluator<T> _fallback;
    private readonly IlgpuBackendSelection _selection;
    private readonly bool _kindSupported;
    private readonly bool _functionSetSupported;
    private Context? _context;
    private Accelerator? _accelerator;
    private Action<Index1D, ArrayView<int>, ArrayView<double>, ArrayView<double>, ArrayView<int>, ArrayView<double>, int, int>? _doubleKernel;
    private Action<Index1D, ArrayView<int>, ArrayView<int>, ArrayView<int>, ArrayView<int>, ArrayView<int>, int, int>? _intKernel;

    /// <summary>Creates an evaluator using the process-wide ILGPU selection.</summary>
    /// <param name="functions">Function set; only the standard sets for the supported types run on the device.</param>
    /// <param name="fallback">Evaluator used when the device cannot run the call; defaults to the reference evaluator.</param>
    public IlgpuGenomeEvaluator(IFunctionSet<T> functions, IGenomeEvaluator<T>? fallback = null)
    {
        ArgumentNullException.ThrowIfNull(functions);
        _functions = functions;
        _fallback = fallback ?? new ReferenceGenomeEvaluator<T>(functions);
        _selection = IlgpuBackendSelector.Selection;
        _kindSupported = typeof(T) == typeof(double) || typeof(T) == typeof(int)
            || typeof(T) == typeof(double[]) || typeof(T) == typeof(int[]);
        _functionSetSupported = IsStandardSetFor(functions);
        LastFallbackReason = _selection.FallbackReason;
        Backend = _selection.BackendName;
        DeviceName = _selection.DeviceName;

        if (_kindSupported && _functionSetSupported && _selection.Kind != IlgpuBackendKind.CpuFallback)
        {
            _context = Context.Create(builder => builder.Default().OpenCL());
            _accelerator = _context.Devices[_selection.DeviceIndex].CreateAccelerator(_context);
            if (typeof(T) == typeof(double) || typeof(T) == typeof(double[]))
            {
                _doubleKernel = _accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<int>, ArrayView<double>, ArrayView<double>, ArrayView<int>, ArrayView<double>, int, int>(DoubleKernel);
            }
            else
            {
                _intKernel = _accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<int>, ArrayView<int>, ArrayView<int>, ArrayView<int>, ArrayView<int>, int, int>(IntKernel);
            }
        }
        else
        {
            Backend = "CPU fallback";
            DeviceName = "nenhum";
            if (!_kindSupported)
            {
                LastFallbackReason = $"tipo {typeof(T).Name} não suportado pelo backend ILGPU";
            }
            else if (!_functionSetSupported)
            {
                LastFallbackReason = "conjunto de funções não padrão";
            }
        }
    }

    /// <summary>Effective backend of the last call (ILGPU/OpenCL, ILGPU/CPU or CPU fallback).</summary>
    public string Backend { get; private set; }

    /// <summary>Device name used by the effective backend.</summary>
    public string DeviceName { get; private set; }

    /// <summary>Cause of the last fallback to CPU, or the selection cause when no OpenCL device was used.</summary>
    public string? LastFallbackReason { get; private set; }

    /// <inheritdoc />
    public void Evaluate(Genome<T> genome, ReadOnlySpan<T> inputs, Span<T> outputs)
    {
        ArgumentNullException.ThrowIfNull(genome);
        var inputArray = inputs.ToArray();
        var outputArray = new T[outputs.Length];

        string? reason = null;
        if (_accelerator is null || !TryEvaluateOnDevice(genome, inputArray, outputArray, out reason))
        {
            if (_accelerator is not null)
            {
                LastFallbackReason = reason;
            }

            Backend = "CPU fallback";
            _fallback.Evaluate(genome, inputArray, outputArray);
        }
        else
        {
            Backend = _selection.BackendName;
            LastFallbackReason = _selection.FallbackReason;
        }

        outputArray.CopyTo(outputs);
    }

    /// <inheritdoc />
    public void Dispose()
    {
        _context?.Dispose();
        _accelerator = null;
        _context = null;
    }

    private static bool IsStandardSetFor(IFunctionSet<T> functions)
    {
        if (typeof(T) == typeof(double)) return functions is StandardDoubleFunctionSet;
        if (typeof(T) == typeof(int)) return functions is StandardInt32FunctionSet;
        if (typeof(T) == typeof(double[])) return functions is StandardDoubleArrayFunctionSet;
        if (typeof(T) == typeof(int[])) return functions is StandardInt32ArrayFunctionSet;
        return false;
    }

    private bool TryEvaluateOnDevice(Genome<T> genome, T[] inputs, T[] outputs, out string? reason)
    {
        reason = null;
        var shape = genome.Shape;
        if (inputs.Length != shape.InputCount || outputs.Length != shape.OutputCount)
        {
            reason = "forma de entrada/saída inválida; fallback preserva o erro de validação";
            return false;
        }

        var isArray = typeof(T) == typeof(double[]) || typeof(T) == typeof(int[]);
        var isDouble = typeof(T) == typeof(double) || typeof(T) == typeof(double[]);
        var lanes = 1;
        if (isArray)
        {
            lanes = -1;
            foreach (var input in inputs)
            {
                var length = ((Array)(object)input!).Length;
                if (lanes < 0)
                {
                    lanes = length;
                }
                else if (lanes != length)
                {
                    reason = "operandos de array com comprimentos diferentes; fallback preserva o erro controlado";
                    return false;
                }
            }

            if (lanes < 0)
            {
                reason = "genoma de array sem entradas de comprimento definido";
                return false;
            }
        }

        var active = genome.Analyze(_functions).ActiveNodes;
        var program = new int[active.Length * Stride];
        var parameters = genome.ParameterGenes;
        var constants = new double[0];
        var intConstants = new int[0];
        if (isDouble)
        {
            constants = new double[active.Length * lanes];
        }
        else
        {
            intConstants = new int[active.Length * lanes];
        }

        for (var i = 0; i < active.Length; i++)
        {
            var node = active[i];
            var function = genome.FunctionGenes[node];
            var op = OpCodeOf(function, isDouble);
            if (op == 0)
            {
                reason = $"função {function} não suportada pelo kernel";
                return false;
            }

            var arity = ArityOf(op);
            var b = i * Stride;
            program[b] = op;
            program[b + 1] = shape.InputCount + node;
            for (var slot = 0; slot < arity; slot++)
            {
                program[b + 2 + slot] = genome.GetConnection(node, slot);
            }

            if (op == OpConstant)
            {
                for (var lane = 0; lane < lanes; lane++)
                {
                    var parameter = parameters[node];
                    if (isArray)
                    {
                        var length = ((Array)(object)parameter!).Length;
                        if (length != lanes)
                        {
                            reason = "parâmetro de array com comprimento diferente; fallback preserva a semântica";
                            return false;
                        }

                        var element = isDouble
                            ? (object)((double[])(object)parameter!)[lane]
                            : ((int[])(object)parameter!)[lane];
                        if (isDouble) constants[i * lanes + lane] = (double)element;
                        else intConstants[i * lanes + lane] = (int)element;
                    }
                    else if (isDouble)
                    {
                        constants[i * lanes + lane] = (double)(object)parameter!;
                    }
                    else
                    {
                        intConstants[i * lanes + lane] = (int)(object)parameter!;
                    }
                }
            }
        }

        var outputSlots = new int[shape.OutputCount];
        for (var o = 0; o < outputSlots.Length; o++)
        {
            outputSlots[o] = genome.OutputGenes[o];
        }

        var slotCount = shape.InputCount + shape.NodeCount;
        var result = new T[shape.OutputCount];
        if (isDouble)
        {
            var slots = new double[slotCount * lanes];
            for (var s = 0; s < shape.InputCount; s++)
            {
                for (var lane = 0; lane < lanes; lane++)
                {
                    slots[s * lanes + lane] = isArray
                        ? ((double[])(object)inputs[s]!)[lane]
                        : (double)(object)inputs[s]!;
                }
            }

            var values = RunDouble(program, constants, slots, outputSlots, lanes, active.Length);
            for (var o = 0; o < shape.OutputCount; o++)
            {
                result[o] = isArray
                    ? (T)(object)CopyLanes(values, o * lanes, lanes)
                    : (T)(object)values[o * lanes];
            }
        }
        else
        {
            var slots = new int[slotCount * lanes];
            for (var s = 0; s < shape.InputCount; s++)
            {
                for (var lane = 0; lane < lanes; lane++)
                {
                    slots[s * lanes + lane] = isArray
                        ? ((int[])(object)inputs[s]!)[lane]
                        : (int)(object)inputs[s]!;
                }
            }

            var values = RunInt(program, intConstants, slots, outputSlots, lanes, active.Length);
            for (var o = 0; o < shape.OutputCount; o++)
            {
                result[o] = isArray
                    ? (T)(object)CopyLanes(values, o * lanes, lanes)
                    : (T)(object)values[o * lanes];
            }
        }

        result.CopyTo(outputs, 0);
        return true;
    }

    private double[] RunDouble(int[] program, double[] constants, double[] slots, int[] outputSlots, int lanes, int instructions)
    {
        var accelerator = _accelerator!;
        using var programBuffer = accelerator.Allocate1D(Pad(program));
        using var constantBuffer = accelerator.Allocate1D(Pad(constants));
        using var slotBuffer = accelerator.Allocate1D(Pad(slots));
        using var outSlotBuffer = accelerator.Allocate1D(outputSlots.Length == 0 ? new int[1] : outputSlots);
        using var outputBuffer = accelerator.Allocate1D<double>(Math.Max(1, outputSlots.Length * lanes));
        _doubleKernel!(new Index1D(lanes), programBuffer.View, constantBuffer.View,
            slotBuffer.View, outSlotBuffer.View, outputBuffer.View, instructions, lanes);
        accelerator.Synchronize();
        var host = new double[Math.Max(1, outputSlots.Length * lanes)];
        outputBuffer.CopyToCPU(host);
        return host;
    }

    private int[] RunInt(int[] program, int[] constants, int[] slots, int[] outputSlots, int lanes, int instructions)
    {
        var accelerator = _accelerator!;
        using var programBuffer = accelerator.Allocate1D(Pad(program));
        using var constantBuffer = accelerator.Allocate1D(Pad(constants));
        using var slotBuffer = accelerator.Allocate1D(Pad(slots));
        using var outSlotBuffer = accelerator.Allocate1D(outputSlots.Length == 0 ? new int[1] : outputSlots);
        using var outputBuffer = accelerator.Allocate1D<int>(Math.Max(1, outputSlots.Length * lanes));
        _intKernel!(new Index1D(lanes), programBuffer.View, constantBuffer.View,
            slotBuffer.View, outSlotBuffer.View, outputBuffer.View, instructions, lanes);
        accelerator.Synchronize();
        var host = new int[Math.Max(1, outputSlots.Length * lanes)];
        outputBuffer.CopyToCPU(host);
        return host;
    }

    private static TValue[] Pad<TValue>(TValue[] values) where TValue : unmanaged
    {
        return values.Length == 0 ? new TValue[1] : values;
    }

    private static Array CopyLanes(Array values, int offset, int lanes)
    {
        var copy = Array.CreateInstance(values.GetType().GetElementType()!, lanes);
        Array.Copy(values, offset, copy, 0, lanes);
        return copy;
    }

    private static int OpCodeOf(int functionId, bool isDouble)
    {
        var ids = new[] { OpAdd, OpSubtract, OpMultiply, OpProtectedDivide, OpNegate, OpAbs, OpSquare };
        if (functionId >= 0 && functionId < ids.Length)
        {
            return ids[functionId];
        }

        if (isDouble)
        {
            return functionId switch
            {
                7 => OpProtectedSqrt,
                8 => OpProtectedLog,
                9 => OpMin,
                10 => OpMax,
                11 => OpGreaterThan,
                12 => OpSelect,
                13 => OpConstant,
                _ => 0,
            };
        }

        return functionId switch
        {
            7 => OpMin,
            8 => OpMax,
            9 => OpGreaterThan,
            10 => OpSelect,
            11 => OpConstant,
            _ => 0,
        };
    }

    private static int ArityOf(int op) => op switch
    {
        OpConstant => 0,
        OpNegate or OpAbs or OpSquare or OpProtectedSqrt or OpProtectedLog => 1,
        OpSelect => 3,
        _ => 2,
    };

    private static void DoubleKernel(
        Index1D lane,
        ArrayView<int> program,
        ArrayView<double> constants,
        ArrayView<double> slots,
        ArrayView<int> outputSlots,
        ArrayView<double> outputs,
        int instructions,
        int lanes)
    {
        var l = lane.X;
        for (var i = 0; i < instructions; i++)
        {
            var b = i * Stride;
            var op = program[b];
            double r;
            if (op == OpConstant)
            {
                r = constants[i * lanes + l];
            }
            else
            {
                r = DoubleOp(op,
                    slots[program[b + 2] * lanes + l],
                    slots[program[b + 3] * lanes + l],
                    slots[program[b + 4] * lanes + l]);
            }

            slots[program[b + 1] * lanes + l] = r;
        }

        for (var o = 0; o < outputSlots.Length; o++)
        {
            outputs[o * lanes + l] = slots[outputSlots[o] * lanes + l];
        }
    }

    private static double DoubleOp(int op, double x0, double x1, double x2)
    {
        var nan = x0 != x0 || x1 != x1;
        switch (op)
        {
            case OpAdd: return x0 + x1;
            case OpSubtract: return x0 - x1;
            case OpMultiply: return x0 * x1;
            case OpProtectedDivide:
                if (nan) return x0 + x1;
                return Math.Abs(x1) <= 1e-12 ? x0 : x0 / x1;
            case OpNegate: return -x0;
            case OpAbs: return Math.Abs(x0);
            case OpSquare: return x0 * x0;
            case OpProtectedSqrt: return x0 != x0 ? x0 : Math.Sqrt(Math.Abs(x0));
            case OpProtectedLog: return x0 != x0 ? x0 : Math.Log(Math.Abs(x0) + 1e-12);
            case OpMin:
                if (nan) return x0 + x1;
                return x0 < x1 ? x0 : x1;
            case OpMax:
                if (nan) return x0 + x1;
                return x0 > x1 ? x0 : x1;
            case OpGreaterThan:
                if (nan) return x0 + x1;
                return x0 > x1 ? 1.0 : 0.0;
            case OpSelect: return x0 > 0 ? x1 : x2;
            default: return 0.0;
        }
    }

    private static void IntKernel(
        Index1D lane,
        ArrayView<int> program,
        ArrayView<int> constants,
        ArrayView<int> slots,
        ArrayView<int> outputSlots,
        ArrayView<int> outputs,
        int instructions,
        int lanes)
    {
        var l = lane.X;
        for (var i = 0; i < instructions; i++)
        {
            var b = i * Stride;
            var op = program[b];
            int r;
            if (op == OpConstant)
            {
                r = constants[i * lanes + l];
            }
            else
            {
                r = IntOp(op,
                    slots[program[b + 2] * lanes + l],
                    slots[program[b + 3] * lanes + l],
                    slots[program[b + 4] * lanes + l]);
            }

            slots[program[b + 1] * lanes + l] = r;
        }

        for (var o = 0; o < outputSlots.Length; o++)
        {
            outputs[o * lanes + l] = slots[outputSlots[o] * lanes + l];
        }
    }

    private static int IntOp(int op, int x0, int x1, int x2)
    {
        const int intMin = -2147483647 - 1;
        const int intMax = 2147483647;
        switch (op)
        {
            case OpAdd: return x0 + x1;
            case OpSubtract: return x0 - x1;
            case OpMultiply: return x0 * x1;
            case OpProtectedDivide:
                if (x1 == 0) return x0;
                if (x0 == intMin && x1 == -1) return intMax;
                return x0 / x1;
            case OpNegate: return 0 - x0;
            case OpAbs:
                if (x0 == intMin) return intMax;
                return x0 < 0 ? 0 - x0 : x0;
            case OpSquare: return x0 * x0;
            case OpMin: return x0 < x1 ? x0 : x1;
            case OpMax: return x0 > x1 ? x0 : x1;
            case OpGreaterThan: return x0 > x1 ? 1 : 0;
            case OpSelect: return x0 > 0 ? x1 : x2;
            default: return 0;
        }
    }
}


