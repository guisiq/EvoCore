using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.CPU;
using ILGPU.Runtime.Cuda;
using ILGPU.Runtime.OpenCL;

namespace Evo.Ilgpu;

/// <summary>Explicit per-run backend policy. <c>LegacyOpenCl</c> keeps the v2 OpenCL/CPU selection.</summary>
public enum IlgpuBackendPolicy
{
    /// <summary>v2 behavior: OpenCL device, then ILGPU CPU, then CPU fallback.</summary>
    LegacyOpenCl,

    /// <summary>CUDA first; on failure falls back to the legacy selection with the cause recorded.</summary>
    CudaPreferred,

    /// <summary>Only a validated CUDA device; fails explicitly otherwise.</summary>
    CudaRequired,
}

/// <summary>Kind of accelerator effectively selected for an ILGPU evaluation.</summary>
public enum IlgpuBackendKind
{
    /// <summary>ILGPU kernels executed on an OpenCL device.</summary>
    OpenClDevice,

    /// <summary>ILGPU kernels executed on a NVIDIA device through the ILGPU CUDA backend.</summary>
    CudaDevice,

    /// <summary>ILGPU kernels executed on the ILGPU CPU accelerator (no OpenCL device usable).</summary>
    IlgpuCpu,

    /// <summary>No ILGPU accelerator usable; evaluation falls back to CPU evaluators.</summary>
    CpuFallback,
}

/// <summary>Stable backend/device selection resolved once per process.</summary>
/// <param name="Kind">Effective backend kind.</param>
/// <param name="DeviceIndex">Index in <c>Context.Devices</c> of the selected device, or -1 for CPU fallback.</param>
/// <param name="DeviceName">Name of the selected device, or a fallback description.</param>
/// <param name="FallbackReason">Why OpenCL was not used; null when an OpenCL device was selected.</param>
/// <param name="WarpSize">Capability captured when the device was probed, used to detect identity drift across contexts.</param>
/// <param name="MaxNumThreads">Capability captured when the device was probed, used to detect identity drift across contexts.</param>
/// <param name="NumMultiprocessors">Multiprocessor count captured when the device was probed.</param>
/// <param name="MaxNumThreadsPerGroup">Per-group thread count captured when the device was probed.</param>
/// <param name="CpuNumThreads">CPU thread configuration captured for ILGPU CPU devices.</param>
/// <param name="PlatformName">OpenCL platform name, when applicable.</param>
/// <param name="PlatformId">OpenCL platform identity token, when applicable.</param>
/// <param name="OpenClDeviceId">OpenCL device identity token, when applicable.</param>
/// <param name="CudaPciId">PCI identity (domain:bus:device) of the CUDA device, when applicable.</param>
/// <param name="CudaCatalog">Observable CUDA catalog entries (descoberto/validado/rejeitado) recorded during the probe.</param>
public sealed record IlgpuBackendSelection(
    IlgpuBackendKind Kind,
    int DeviceIndex,
    string DeviceName,
    string? FallbackReason,
    int WarpSize = 0,
    int MaxNumThreads = 0,
    int NumMultiprocessors = 0,
    int MaxNumThreadsPerGroup = 0,
    int CpuNumThreads = 0,
    string? PlatformName = null,
    string? PlatformId = null,
    string? OpenClDeviceId = null,
    string? CudaPciId = null,
    IReadOnlyList<string>? CudaCatalog = null)
{
    /// <summary>Human-readable backend name for diagnostics.</summary>
    public string BackendName => Kind switch
    {
        IlgpuBackendKind.OpenClDevice => "ILGPU/OpenCL",
        IlgpuBackendKind.CudaDevice => "ILGPU/CUDA",
        IlgpuBackendKind.IlgpuCpu => "ILGPU/CPU",
        _ => "CPU fallback",
    };

    /// <summary>Human-readable platform/device diagnostic text for the selected accelerator.</summary>
    public string DiagnosticDeviceName => Kind == IlgpuBackendKind.OpenClDevice && !string.IsNullOrWhiteSpace(PlatformName)
        ? $"{PlatformName} :: {DeviceName}"
        : DeviceName;
}

/// <summary>
/// Detects ILGPU accelerators at runtime. OpenCL devices are preferred; the ILGPU CPU accelerator is
/// used when OpenCL is unavailable; otherwise the selection reports an explicit CPU fallback with its cause.
/// </summary>
public static class IlgpuBackendSelector
{
    private static readonly Lazy<IlgpuBackendSelection> Cached = new(Probe, isThreadSafe: true);

    /// <summary>
    /// Tracks, per selection instance, whether a worker observed a device initialization failure after the
    /// shared selection pointed to a real accelerator. Once set, every evaluator sharing that selection
    /// (already running or newly created) converges to CPU fallback, so a run never mixes effective backends
    /// across workers.
    /// </summary>
    private static readonly System.Collections.Concurrent.ConcurrentDictionary<IlgpuBackendSelection, string> ForcedFallbackReasons = new();

    /// <summary>Selection resolved once per process, so every worker of a run uses the same device.</summary>
    public static IlgpuBackendSelection Selection => Cached.Value;

    /// <summary>
    /// Records that a worker failed to initialize the device for <paramref name="selection"/> after the shared
    /// selection pointed to a real accelerator. Idempotent: only the first reported reason is kept.
    /// </summary>
    internal static void ReportInitializationFailure(IlgpuBackendSelection selection, string reason)
    {
        if (selection.Kind != IlgpuBackendKind.CpuFallback)
        {
            ForcedFallbackReasons.TryAdd(selection, reason);
        }
    }

    /// <summary>True when some worker sharing <paramref name="selection"/> already reported an initialization failure.</summary>
    internal static bool TryGetForcedFallbackReason(IlgpuBackendSelection selection, out string reason)
    {
        if (ForcedFallbackReasons.TryGetValue(selection, out var found))
        {
            reason = found;
            return true;
        }

        reason = string.Empty;
        return false;
    }

    /// <summary>Test-only reset of the shared forced-fallback state so test selections do not leak across cases.</summary>
    internal static void ResetForcedFallbackForTests(IlgpuBackendSelection selection) =>
        ForcedFallbackReasons.TryRemove(selection, out _);

    internal static IlgpuBackendSelection ProbeForTests(
        Func<Context> contextFactory,
        Func<Context, int, Accelerator> acceleratorFactory,
        Func<Accelerator, Exception?>? kernelProbeOverride = null) =>
        Probe(contextFactory, acceleratorFactory, kernelProbeOverride);

    internal static IlgpuBackendSelection ProbeCandidatesForTests(
        IEnumerable<IlgpuBackendSelection> openClCandidates,
        IEnumerable<IlgpuBackendSelection> cpuCandidates,
        Func<IlgpuBackendSelection, Exception?> validator) =>
        SelectCandidate(openClCandidates.ToArray(), cpuCandidates.ToArray(), validator);

    /// <summary>
    /// Resolves the selection for an explicit policy. CUDA candidates are validated by loading the required
    /// kernels before being chosen; <c>CudaRequired</c> throws when no CUDA device validates.
    /// </summary>
    public static IlgpuBackendSelection SelectFor(IlgpuBackendPolicy policy)
    {
        if (policy == IlgpuBackendPolicy.LegacyOpenCl)
        {
            return Selection;
        }

        var cuda = ProbeCuda();
        if (cuda.Kind == IlgpuBackendKind.CudaDevice)
        {
            return cuda;
        }

        if (policy == IlgpuBackendPolicy.CudaRequired)
        {
            throw new InvalidOperationException($"CUDA obrigatório indisponível: {cuda.FallbackReason}");
        }

        return Selection with { FallbackReason = $"CUDA indisponível ({cuda.FallbackReason}); usando seleção legada" };
    }

    private static IlgpuBackendSelection ProbeCuda()
    {
        try
        {
            using var context = Context.Create(builder => builder.Cuda());
            var devices = context.Devices.ToArray();
            var candidates = devices
                .Select((device, index) => (device, index))
                .Where(item => item.device.AcceleratorType == AcceleratorType.Cuda)
                .Select(item => CreateCudaCandidate(item.device, item.index))
                .OrderBy(candidate => candidate.CudaPciId, StringComparer.Ordinal)
                .ToArray();
            if (candidates.Length == 0)
            {
                return NoCuda("nenhum dispositivo CUDA encontrado pelo runtime ILGPU");
            }

            // Catálogo observável: cada candidato CUDA descoberto recebe estado (descoberto -> validado) e motivo de rejeição.
            var catalog = new List<string>();
            foreach (var candidate in candidates)
            {
                catalog.Add($"descoberto {candidate.DeviceName} pci={candidate.CudaPciId}");
            }

            string? lastError = null;
            foreach (var candidate in candidates)
            {
                try
                {
                    using var accelerator = context.Devices[candidate.DeviceIndex].CreateAccelerator(context);
                    var error = ProbeRequiredKernelSupport(accelerator);
                    if (error is null)
                    {
                        catalog.Add($"validado {candidate.DeviceName} pci={candidate.CudaPciId}");
                        return candidate with { CudaCatalog = catalog.ToArray() };
                    }

                    lastError = $"dispositivo CUDA '{candidate.DeviceName}' não utilizável: {error.GetType().Name}";
                    catalog.Add($"rejeitado {candidate.DeviceName}: {error.GetType().Name}");
                }
                catch (Exception ex) when (ex is not OutOfMemoryException)
                {
                    lastError = $"dispositivo CUDA '{candidate.DeviceName}' não utilizável: {ex.GetType().Name}";
                    catalog.Add($"rejeitado {candidate.DeviceName}: {ex.GetType().Name}");
                }
            }

            return NoCuda(lastError ?? "nenhum dispositivo CUDA utilizável") with { CudaCatalog = catalog.ToArray() };
        }
        catch (Exception ex)
        {
            return NoCuda($"falha de inicialização do runtime CUDA: {ex.GetType().Name}");
        }
    }

    private static IlgpuBackendSelection NoCuda(string reason) =>
        new(IlgpuBackendKind.CpuFallback, -1, "nenhum", reason);

    private static IlgpuBackendSelection CreateCudaCandidate(Device device, int index)
    {
        var cuda = (CudaDevice)device;
        return new IlgpuBackendSelection(
            IlgpuBackendKind.CudaDevice,
            index,
            device.Name,
            null,
            device.WarpSize,
            device.MaxNumThreads,
            device.NumMultiprocessors,
            device.MaxNumThreadsPerGroup,
            CpuNumThreads: 0,
            CudaPciId: $"{cuda.PCIDomainId:X4}:{cuda.PCIBusId:X2}:{cuda.PCIDeviceId:X2}");
    }

    private static IlgpuBackendSelection Probe() =>
        Probe(
            () => Context.Create(builder => builder.OpenCL().CPU()),
            static (context, index) => context.Devices[index].CreateAccelerator(context),
            kernelProbeOverride: null);

    private static IlgpuBackendSelection Probe(
        Func<Context> contextFactory,
        Func<Context, int, Accelerator> acceleratorFactory,
        Func<Accelerator, Exception?>? kernelProbeOverride)
    {
        try
        {
            using var context = contextFactory();
            var devices = context.Devices.ToArray();
            var openClCandidates = devices
                .Select((device, index) => CreateCandidate(device, index, openClFallbackReason: null))
                .Where(candidate => candidate.Kind == IlgpuBackendKind.OpenClDevice)
                .ToArray();
            var cpuCandidates = devices
                .Select((device, index) => CreateCandidate(device, index, openClFallbackReason: "nenhum dispositivo OpenCL encontrado pelo runtime ILGPU"))
                .Where(candidate => candidate.Kind == IlgpuBackendKind.IlgpuCpu)
                .ToArray();
            return SelectCandidate(openClCandidates, cpuCandidates, candidate =>
            {
                try
                {
                    using var accelerator = acceleratorFactory(context, candidate.DeviceIndex);
                    return (kernelProbeOverride ?? ProbeRequiredKernelSupport)(accelerator);
                }
                catch (Exception ex) when (ex is not OutOfMemoryException)
                {
                    return ex;
                }
            });
        }
        catch (Exception ex)
        {
            return new IlgpuBackendSelection(
                IlgpuBackendKind.CpuFallback,
                -1,
                "nenhum",
                $"falha de inicialização do runtime ILGPU/OpenCL: {ex.GetType().Name}");
        }
    }

    private static IlgpuBackendSelection SelectCandidate(
        IReadOnlyList<IlgpuBackendSelection> openClCandidates,
        IReadOnlyList<IlgpuBackendSelection> cpuCandidates,
        Func<IlgpuBackendSelection, Exception?> validator)
    {
        string? openClFailure = null;
        foreach (var candidate in openClCandidates)
        {
            var error = validator(candidate);
            if (error is null)
            {
                return candidate;
            }

            openClFailure = $"dispositivo OpenCL '{candidate.DiagnosticDeviceName}' não utilizável: {error.GetType().Name}";
        }

        openClFailure ??= "nenhum dispositivo OpenCL encontrado pelo runtime ILGPU";
        foreach (var candidate in cpuCandidates)
        {
            var cpuCandidate = candidate with { FallbackReason = openClFailure };
            var error = validator(cpuCandidate);
            if (error is null)
            {
                return cpuCandidate;
            }

            return new IlgpuBackendSelection(
                IlgpuBackendKind.CpuFallback,
                -1,
                "nenhum",
                $"{openClFailure}; acelerador CPU do ILGPU não utilizável: {error.GetType().Name}");
        }

        return new IlgpuBackendSelection(
            IlgpuBackendKind.CpuFallback,
            -1,
            "nenhum",
            openClFailure + "; acelerador CPU do ILGPU indisponível");
    }

    private static IlgpuBackendSelection CreateCandidate(Device device, int index, string? openClFallbackReason)
    {
        if (device.AcceleratorType == AcceleratorType.OpenCL)
        {
            var openCl = (CLDevice)device;
            return new IlgpuBackendSelection(
                IlgpuBackendKind.OpenClDevice,
                index,
                device.Name,
                null,
                device.WarpSize,
                device.MaxNumThreads,
                device.NumMultiprocessors,
                device.MaxNumThreadsPerGroup,
                CpuNumThreads: 0,
                PlatformName: openCl.PlatformName,
                PlatformId: openCl.PlatformId.ToString(),
                OpenClDeviceId: openCl.DeviceId.ToString());
        }

        if (device.AcceleratorType == AcceleratorType.CPU)
        {
            var cpu = (CPUDevice)device;
            return new IlgpuBackendSelection(
                IlgpuBackendKind.IlgpuCpu,
                index,
                device.Name,
                openClFallbackReason,
                device.WarpSize,
                device.MaxNumThreads,
                device.NumMultiprocessors,
                device.MaxNumThreadsPerGroup,
                cpu.NumThreads);
        }

        return new IlgpuBackendSelection(
            IlgpuBackendKind.CpuFallback,
            -1,
            "nenhum",
            openClFallbackReason ?? "dispositivo não suportado");
    }

    private static Exception? ProbeRequiredKernelSupport(Accelerator accelerator)
    {
        try
        {
            accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<int>, ArrayView<double>, ArrayView<double>, ArrayView<int>, ArrayView<double>, ArrayView<int>, int, int>(
                IlgpuGenomeEvaluator<double>.DoubleKernel);
            accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<int>, ArrayView<int>, ArrayView<int>, ArrayView<int>, ArrayView<int>, ArrayView<int>, int, int>(
                IlgpuGenomeEvaluator<int>.IntKernel);
            return null;
        }
        catch (Exception ex) when (ex is not OutOfMemoryException)
        {
            return ex;
        }
    }
}
