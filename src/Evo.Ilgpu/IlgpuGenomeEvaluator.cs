using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.CPU;
using ILGPU.Runtime.OpenCL;
using Evo.Core;
using Evo.Core.Compiled;
using Evo.Core.Functions;
using Evo.Core.Values;

[assembly: System.Runtime.CompilerServices.InternalsVisibleTo("Evo.Ilgpu.Tests")]

namespace Evo.Ilgpu;

/// <summary>
/// Raised when a worker fails to initialize the backend/device that the process-wide selection already resolved
/// for the current run, or when a worker is created after another worker reported that same failure. Per
/// <c>06-v2-ilgpu-backend.md</c> decision 2, a failure after the initial selection never falls back silently: the
/// evaluation fails explicitly identifying the worker, the attempted backend and the cause, so the run never
/// continues with different effective backends across workers.
/// </summary>
public sealed class IlgpuBackendInitializationException : Exception
{
    /// <summary>Identifier of the worker (evaluator instance) that observed the failure.</summary>
    public long WorkerId { get; }

    /// <summary>Backend name that was attempted (for example, <c>ILGPU/OpenCL</c> or <c>ILGPU/CPU</c>).</summary>
    public string Backend { get; }

    /// <summary>Underlying cause reported by the failing worker or by the worker that first observed it.</summary>
    public string Cause { get; }

    internal IlgpuBackendInitializationException(long workerId, string backend, string cause, Exception? innerException = null)
        : base($"worker={workerId}: falha ao inicializar {backend} depois da seleção compartilhada: {cause}", innerException)
    {
        WorkerId = workerId;
        Backend = backend;
        Cause = cause;
    }
}

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
    private readonly bool _eligibleForDevice;
    private readonly Action? _beforeKernelExecution;
    private readonly Action<Action>? _launchInvoker;
    private readonly Action<Accelerator>? _synchronizeInvoker;
    private readonly Action<MemoryBuffer[]>? _bufferObserver;
    private Context? _context;
    private Accelerator? _accelerator;
    private string? _unavailableReason;
    private bool _disposed;
    private Action<Index1D, ArrayView<int>, ArrayView<double>, ArrayView<double>, ArrayView<int>, ArrayView<double>, ArrayView<int>, int, int>? _doubleKernel;
    private Action<Index1D, ArrayView<int>, ArrayView<int>, ArrayView<int>, ArrayView<int>, ArrayView<int>, ArrayView<int>, int, int>? _intKernel;
    private static long _workerCounter;

    /// <summary>Creates an evaluator using the process-wide ILGPU selection.</summary>
    /// <param name="functions">Function set; only the standard sets for the supported types run on the device.</param>
    /// <param name="fallback">Evaluator used when the device cannot run the call; defaults to the compiled CPU evaluator.</param>
    public IlgpuGenomeEvaluator(IFunctionSet<T> functions, IGenomeEvaluator<T>? fallback = null)
        : this(functions, fallback, null, null, null, null)
    {
    }

    internal IlgpuGenomeEvaluator(
        IFunctionSet<T> functions,
        IGenomeEvaluator<T>? fallback,
        IlgpuBackendSelection? selectionOverride,
        Func<IlgpuBackendSelection, Context>? contextFactory,
        Func<Context, IlgpuBackendSelection, Accelerator>? acceleratorFactory,
        Action? beforeKernelExecution,
        Action<Action>? launchInvoker = null,
        Action<Accelerator>? synchronizeInvoker = null,
        Action<MemoryBuffer[]>? bufferObserver = null)
    {
        ArgumentNullException.ThrowIfNull(functions);
        WorkerId = System.Threading.Interlocked.Increment(ref _workerCounter) - 1;
        _functions = functions;
        _fallback = fallback ?? CreateDefaultFallback(functions);
        _selection = selectionOverride ?? IlgpuBackendSelector.Selection;
        _kindSupported = typeof(T) == typeof(double) || typeof(T) == typeof(int)
            || typeof(T) == typeof(double[]) || typeof(T) == typeof(int[]);
        _functionSetSupported = IsStandardSetFor(functions);
        // Workers whose type/function set are not supported, or whose selection already resolved to the
        // CPU-fallback kind, never attempt the device backend; a shared initialization failure recorded by
        // another worker does not apply to them. Only workers that would otherwise run on the device must keep
        // failing explicitly (06-v2-ilgpu-backend.md decision 2), independently of whether their own
        // accelerator handle was already released by ReleaseDevice() after observing the shared failure.
        _eligibleForDevice = _kindSupported && _functionSetSupported && _selection.Kind != IlgpuBackendKind.CpuFallback;
        _beforeKernelExecution = beforeKernelExecution;
        _launchInvoker = launchInvoker;
        _synchronizeInvoker = synchronizeInvoker;
        _bufferObserver = bufferObserver;
        LastFallbackReason = _selection.FallbackReason;
        Backend = _selection.BackendName;
        DeviceName = _selection.DiagnosticDeviceName;
        FallbackEvaluatorKind = fallback is null ? "compiled-default" : "custom";

        if (!_kindSupported)
        {
            _unavailableReason = $"tipo {typeof(T).Name} não suportado pelo backend ILGPU";
        }
        else if (!_functionSetSupported)
        {
            _unavailableReason = "conjunto de funções não padrão";
        }
        else if (_selection.Kind == IlgpuBackendKind.CpuFallback)
        {
            _unavailableReason = _selection.FallbackReason;
        }
        else if (IlgpuBackendSelector.TryGetForcedFallbackReason(_selection, out var priorFailureReason))
        {
            // Outro worker com a mesma seleção já falhou ao inicializar o backend/dispositivo depois da
            // escolha compartilhada. Por 06-v2-ilgpu-backend.md decisão 2, a execução nunca continua com
            // backends efetivos diferentes entre workers: este worker também falha de forma explícita,
            // em vez de convergir silenciosamente para o fallback de CPU.
            throw new IlgpuBackendInitializationException(WorkerId, _selection.BackendName, priorFailureReason);
        }
        else
        {
            try
            {
                InitializeDevice(contextFactory, acceleratorFactory);
            }
            catch (Exception ex) when (ex is not OutOfMemoryException)
            {
                ReleaseDevice();
                var cause = $"falha ao inicializar {_selection.BackendName} em '{_selection.DeviceName}': {ex.GetType().Name}: {ex.Message}";
                IlgpuBackendSelector.ReportInitializationFailure(_selection, cause);
                throw new IlgpuBackendInitializationException(WorkerId, _selection.BackendName, cause);
            }
            catch (Exception ex)
            {
                // Qualquer acelerador/contexto parcialmente construído (incluindo OutOfMemoryException,
                // que não cai no fallback acima de propósito) precisa ser liberado antes de propagar a
                // exceção original sem mascará-la como fallback bem-sucedido. Mesmo quando a exceção
                // original é preservada sem ser substituída (por exemplo OutOfMemoryException), a falha
                // compartilhada precisa ser registrada: caso contrário outro worker com a mesma seleção
                // poderia construir com sucesso depois desta falha, divergindo do worker=06 decisão 2.
                ReleaseDevice();
                var cause = $"falha ao inicializar {_selection.BackendName} em '{_selection.DeviceName}': {ex.GetType().Name}: {ex.Message}";
                IlgpuBackendSelector.ReportInitializationFailure(_selection, cause);
                // A causa original é preservada em InnerException; o worker e o backend tentado são expostos
                // no erro entregue ao chamador, sem sucesso por fallback (06-v2-ilgpu-backend.md decisão 2).
                throw new IlgpuBackendInitializationException(WorkerId, _selection.BackendName, cause, ex);
            }
        }

        if (_accelerator is null)
        {
            Backend = "CPU fallback";
            DeviceName = "nenhum";
            LastFallbackReason = _unavailableReason;
        }
    }

    /// <summary>Identifier of this worker/evaluator instance, used in explicit initialization-failure messages.</summary>
    public long WorkerId { get; }

    /// <summary>Internal handles used by tests to prove resource release.</summary>
    internal Accelerator? InternalAccelerator => _accelerator;

    /// <summary>Internal handles used by tests to prove resource release.</summary>
    internal Context? InternalContext => _context;

    /// <summary>True when the last evaluation executed an ILGPU kernel instead of the CPU fallback.</summary>
    internal bool LastEvaluationUsedDevice { get; private set; }

    /// <summary>Per-slot execution counts produced by the device path of the last call (inputs stay at 0; active nodes at 1).</summary>
    internal int[]? LastDeviceExecutionCounts { get; private set; }

    /// <summary>Effective backend of the last call (ILGPU/OpenCL, ILGPU/CPU or CPU fallback).</summary>
    public string Backend { get; private set; }

    /// <summary>Device name used by the effective backend.</summary>
    public string DeviceName { get; private set; }

    /// <summary>Cause of the last fallback to CPU, or the selection cause when no OpenCL device was used.</summary>
    public string? LastFallbackReason { get; private set; }

    /// <summary>Distinguishes the built-in compiled CPU fallback from a custom fallback evaluator.</summary>
    public string FallbackEvaluatorKind { get; }

    /// <inheritdoc />
    public void Evaluate(Genome<T> genome, ReadOnlySpan<T> inputs, Span<T> outputs)
    {
        ArgumentNullException.ThrowIfNull(genome);
        var inputArray = inputs.ToArray();
        var outputArray = new T[outputs.Length];
        LastEvaluationUsedDevice = false;
        LastDeviceExecutionCounts = null;

        if (_eligibleForDevice && IlgpuBackendSelector.TryGetForcedFallbackReason(_selection, out var forcedReason))
        {
            // Outro worker com a mesma seleção falhou ao inicializar o dispositivo depois da escolha comum.
            // Por 06-v2-ilgpu-backend.md decisão 2, a avaliação falha de forma explícita em vez de convergir
            // silenciosamente para o fallback de CPU: a execução nunca continua com backends efetivos
            // diferentes entre workers. A checagem independe de _accelerator ainda estar vivo: depois que
            // ReleaseDevice() zera o handle na primeira tentativa posterior à falha compartilhada, as
            // tentativas seguintes deste worker elegível para o dispositivo continuam falhando de forma
            // explícita, nunca retornando sucesso por fallback de CPU.
            ReleaseDevice();
            throw new IlgpuBackendInitializationException(WorkerId, _selection.BackendName, forcedReason);
        }

        string? reason = _disposed ? "avaliador descartado" : _unavailableReason;
        if (_accelerator is not null)
        {
            Backend = _selection.BackendName;
            DeviceName = _selection.DiagnosticDeviceName;
            LastFallbackReason = _selection.FallbackReason;
            if (TryEvaluateOnDevice(genome, inputArray, outputArray, out reason))
            {
                LastEvaluationUsedDevice = true;
            }
            else
            {
                Backend = "CPU fallback";
                DeviceName = "nenhum";
                LastFallbackReason = DescribeFallback(reason);
                _fallback.Evaluate(genome, inputArray, outputArray);
            }
        }
        else
        {
            Backend = "CPU fallback";
            DeviceName = "nenhum";
            LastFallbackReason = DescribeFallback(reason);
            _fallback.Evaluate(genome, inputArray, outputArray);
        }

        outputArray.CopyTo(outputs);
    }

    /// <inheritdoc />
    /// <remarks>Releases the accelerator before its context; repeated calls are no-ops. Exceptions are not masked.</remarks>
    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }

        _disposed = true;
        ReleaseDevice();
    }

    private void InitializeDevice(
        Func<IlgpuBackendSelection, Context>? contextFactory,
        Func<Context, IlgpuBackendSelection, Accelerator>? acceleratorFactory)
    {
        var context = (contextFactory ?? CreateContext)(_selection);
        _context = context;
        _accelerator = (acceleratorFactory ?? CreateAccelerator)(context, _selection);
        if (typeof(T) == typeof(double) || typeof(T) == typeof(double[]))
        {
            _doubleKernel = _accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<int>, ArrayView<double>, ArrayView<double>, ArrayView<int>, ArrayView<double>, ArrayView<int>, int, int>(DoubleKernel);
        }
        else
        {
            _intKernel = _accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<int>, ArrayView<int>, ArrayView<int>, ArrayView<int>, ArrayView<int>, ArrayView<int>, int, int>(IntKernel);
        }
    }

    private static Context CreateContext(IlgpuBackendSelection selection) =>
        Context.Create(builder => builder.OpenCL().CPU());

    private static Accelerator CreateAccelerator(Context context, IlgpuBackendSelection selection)
    {
        var devices = context.Devices.ToArray();
        var index = selection.DeviceIndex;
        if (index < 0 || index >= devices.Length)
        {
            throw new InvalidOperationException($"dispositivo '{selection.DeviceName}' não reconhecido no novo contexto");
        }

        var device = devices[index];
        if (!MatchesSelection(device, selection))
        {
            throw new InvalidOperationException($"dispositivo '{selection.DeviceName}' não reconhecido no novo contexto");
        }

        var accelerator = device.CreateAccelerator(context);
        if (!MatchesSelection(accelerator, selection))
        {
            accelerator.Dispose();
            throw new InvalidOperationException(
                $"dispositivo '{selection.DeviceName}' não reconhecido no novo contexto: capacidades divergentes");
        }

        return accelerator;
    }

    private static AcceleratorType ExpectedAcceleratorType(IlgpuBackendKind kind) =>
        kind == IlgpuBackendKind.OpenClDevice ? AcceleratorType.OpenCL : AcceleratorType.CPU;

    private static bool MatchesSelection(Device device, IlgpuBackendSelection selection)
    {
        if (device.Name != selection.DeviceName
            || device.AcceleratorType != ExpectedAcceleratorType(selection.Kind)
            || (selection.NumMultiprocessors != 0 && device.NumMultiprocessors != selection.NumMultiprocessors)
            || (selection.MaxNumThreadsPerGroup != 0 && device.MaxNumThreadsPerGroup != selection.MaxNumThreadsPerGroup))
        {
            return false;
        }

        if (selection.Kind == IlgpuBackendKind.OpenClDevice)
        {
            if (device is not CLDevice openCl)
            {
                return false;
            }

            return openCl.PlatformName == selection.PlatformName
                && openCl.PlatformId.ToString() == selection.PlatformId
                && openCl.DeviceId.ToString() == selection.OpenClDeviceId;
        }

        if (selection.Kind == IlgpuBackendKind.IlgpuCpu)
        {
            return device is CPUDevice cpu
                && (selection.CpuNumThreads == 0 || cpu.NumThreads == selection.CpuNumThreads);
        }

        return false;
    }

    private static bool MatchesSelection(Accelerator accelerator, IlgpuBackendSelection selection)
    {
        return (selection.WarpSize == 0 || accelerator.WarpSize == selection.WarpSize)
            && (selection.MaxNumThreads == 0 || accelerator.MaxNumThreads == selection.MaxNumThreads);
    }

    private void ReleaseDevice()
    {
        var accelerator = _accelerator;
        var context = _context;
        _accelerator = null;
        _context = null;
        _doubleKernel = null;
        _intKernel = null;
        try
        {
            accelerator?.Dispose();
        }
        finally
        {
            context?.Dispose();
        }
    }

    private static IGenomeEvaluator<T> CreateDefaultFallback(IFunctionSet<T> functions)
    {
        object? values = typeof(T) == typeof(double) ? DoubleValueSemantics.Instance
            : typeof(T) == typeof(int) ? Int32ValueSemantics.Instance
            : typeof(T) == typeof(double[]) ? DoubleArrayValueSemantics.Instance
            : typeof(T) == typeof(int[]) ? Int32ArrayValueSemantics.Instance
            : null;
        return values is null
            ? new ReferenceGenomeEvaluator<T>(functions)
            : new CompiledGenomeEvaluator<T>(functions, (IValueSemantics<T>)values);
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

            if (lanes == 0)
            {
                reason = "arrays vazios não são executados no kernel; fallback CPU preserva a semântica";
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

            var values = RunDouble(program, constants, slots, outputSlots, lanes, active.Length, slotCount);
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

            var values = RunInt(program, intConstants, slots, outputSlots, lanes, active.Length, slotCount);
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

    private double[] RunDouble(int[] program, double[] constants, double[] slots, int[] outputSlots, int lanes, int instructions, int slotCount)
    {
        var accelerator = _accelerator!;
        using var programBuffer = accelerator.Allocate1D(Pad(program));
        using var constantBuffer = accelerator.Allocate1D(Pad(constants));
        using var slotBuffer = accelerator.Allocate1D(Pad(slots));
        using var outSlotBuffer = accelerator.Allocate1D(outputSlots.Length == 0 ? new int[1] : outputSlots);
        using var executionBuffer = accelerator.Allocate1D(new int[Math.Max(1, slotCount)]);
        using var outputBuffer = accelerator.Allocate1D<double>(Math.Max(1, outputSlots.Length * lanes));
        var buffers = new MemoryBuffer[] { programBuffer, constantBuffer, slotBuffer, outSlotBuffer, executionBuffer, outputBuffer };
        _bufferObserver?.Invoke(buffers);
        _beforeKernelExecution?.Invoke();
        (_launchInvoker ?? InvokeLaunch)(() => _doubleKernel!(new Index1D(lanes), programBuffer.View, constantBuffer.View,
            slotBuffer.View, outSlotBuffer.View, outputBuffer.View, executionBuffer.View, instructions, lanes));
        (_synchronizeInvoker ?? SynchronizeAccelerator)(accelerator);
        var host = new double[Math.Max(1, outputSlots.Length * lanes)];
        var counts = new int[Math.Max(1, slotCount)];
        outputBuffer.CopyToCPU(host);
        executionBuffer.CopyToCPU(counts);
        LastDeviceExecutionCounts = counts;
        return host;
    }

    private int[] RunInt(int[] program, int[] constants, int[] slots, int[] outputSlots, int lanes, int instructions, int slotCount)
    {
        var accelerator = _accelerator!;
        using var programBuffer = accelerator.Allocate1D(Pad(program));
        using var constantBuffer = accelerator.Allocate1D(Pad(constants));
        using var slotBuffer = accelerator.Allocate1D(Pad(slots));
        using var outSlotBuffer = accelerator.Allocate1D(outputSlots.Length == 0 ? new int[1] : outputSlots);
        using var executionBuffer = accelerator.Allocate1D(new int[Math.Max(1, slotCount)]);
        using var outputBuffer = accelerator.Allocate1D<int>(Math.Max(1, outputSlots.Length * lanes));
        var buffers = new MemoryBuffer[] { programBuffer, constantBuffer, slotBuffer, outSlotBuffer, executionBuffer, outputBuffer };
        _bufferObserver?.Invoke(buffers);
        _beforeKernelExecution?.Invoke();
        (_launchInvoker ?? InvokeLaunch)(() => _intKernel!(new Index1D(lanes), programBuffer.View, constantBuffer.View,
            slotBuffer.View, outSlotBuffer.View, outputBuffer.View, executionBuffer.View, instructions, lanes));
        (_synchronizeInvoker ?? SynchronizeAccelerator)(accelerator);
        var host = new int[Math.Max(1, outputSlots.Length * lanes)];
        var counts = new int[Math.Max(1, slotCount)];
        outputBuffer.CopyToCPU(host);
        executionBuffer.CopyToCPU(counts);
        LastDeviceExecutionCounts = counts;
        return host;
    }

    private static void InvokeLaunch(Action launch)
    {
        launch();
    }

    private static void SynchronizeAccelerator(Accelerator accelerator)
    {
        accelerator.Synchronize();
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

    internal static void DoubleKernel(
        Index1D lane,
        ArrayView<int> program,
        ArrayView<double> constants,
        ArrayView<double> slots,
        ArrayView<int> outputSlots,
        ArrayView<double> outputs,
        ArrayView<int> executionCounts,
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
            if (l == 0)
            {
                executionCounts[program[b + 1]]++;
            }
        }

        for (var o = 0; o < outputSlots.Length; o++)
        {
            outputs[o * lanes + l] = slots[outputSlots[o] * lanes + l];
        }
    }

    private static double DoubleOp(int op, double x0, double x1, double x2)
    {
        var nan = double.IsNaN(x0) || double.IsNaN(x1);
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
            case OpProtectedSqrt: return double.IsNaN(x0) ? x0 : Math.Sqrt(Math.Abs(x0));
            case OpProtectedLog: return double.IsNaN(x0) ? x0 : Math.Log(Math.Abs(x0) + 1e-12);
            case OpMin:
                if (nan) return x0 + x1;
                if (x0 < x1) return x0;
                if (x1 < x0) return x1;
                return x0 == 0.0 && x1 == 0.0
                    ? (IsNegativeZeroBit(x0) ? x0 : x1)
                    : x1;
            case OpMax:
                if (nan) return x0 + x1;
                if (x0 > x1) return x0;
                if (x1 > x0) return x1;
                return x0 == 0.0 && x1 == 0.0
                    ? (IsNegativeZeroBit(x0) ? x1 : x0)
                    : x1;
            case OpGreaterThan:
                if (nan) return x0 + x1;
                return x0 > x1 ? 1.0 : 0.0;
            case OpSelect: return x0 > 0 ? x1 : x2;
            default: return 0.0;
        }
    }

    // Detecta o sinal de um zero via reinterpretação de bits (sem divisão), evitando o
    // intrínseco RcpF (1.0/x0) que o backend ILGPU/OpenCL 1.2.0 não implementa.
    private static bool IsNegativeZeroBit(double x) => BitConverter.DoubleToInt64Bits(x) < 0L;

    private string DescribeFallback(string? reason)
    {
        var fallback = FallbackEvaluatorKind == "compiled-default" ? "compiled CPU default" : "custom fallback evaluator";
        return reason is null ? fallback : $"{reason}; fallback={fallback}";
    }

    internal static void IntKernel(
        Index1D lane,
        ArrayView<int> program,
        ArrayView<int> constants,
        ArrayView<int> slots,
        ArrayView<int> outputSlots,
        ArrayView<int> outputs,
        ArrayView<int> executionCounts,
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
            if (l == 0)
            {
                executionCounts[program[b + 1]]++;
            }
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
