using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.CPU;
using ILGPU.Runtime.OpenCL;

namespace Evo.Ilgpu;

/// <summary>Kind of accelerator effectively selected for an ILGPU evaluation.</summary>
public enum IlgpuBackendKind
{
    /// <summary>ILGPU kernels executed on an OpenCL device.</summary>
    OpenClDevice,

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
    string? OpenClDeviceId = null)
{
    /// <summary>Human-readable backend name for diagnostics.</summary>
    public string BackendName => Kind switch
    {
        IlgpuBackendKind.OpenClDevice => "ILGPU/OpenCL",
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

    /// <summary>Selection resolved once per process, so every worker of a run uses the same device.</summary>
    public static IlgpuBackendSelection Selection => Cached.Value;

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
