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
public sealed record IlgpuBackendSelection(
    IlgpuBackendKind Kind,
    int DeviceIndex,
    string DeviceName,
    string? FallbackReason)
{
    /// <summary>Human-readable backend name for diagnostics.</summary>
    public string BackendName => Kind switch
    {
        IlgpuBackendKind.OpenClDevice => "ILGPU/OpenCL",
        IlgpuBackendKind.IlgpuCpu => "ILGPU/CPU",
        _ => "CPU fallback",
    };
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

    private static IlgpuBackendSelection Probe()
    {
        string? openClFailure = null;
        try
        {
            using var context = Context.Create(builder => builder.OpenCL().CPU());
            var devices = context.Devices.ToArray();
            string? unusable = null;
            for (var i = 0; i < devices.Length; i++)
            {
                if (devices[i].AcceleratorType != AcceleratorType.OpenCL)
                {
                    continue;
                }

                try
                {
                    using var probe = devices[i].CreateAccelerator(context);
                    return new IlgpuBackendSelection(IlgpuBackendKind.OpenClDevice, i, devices[i].Name, null);
                }
                catch (Exception ex) when (ex is not OutOfMemoryException)
                {
                    unusable = $"dispositivo OpenCL '{devices[i].Name}' não inicializável: {ex.GetType().Name}";
                }
            }

            openClFailure = unusable ?? "nenhum dispositivo OpenCL encontrado pelo runtime ILGPU";
            for (var i = 0; i < devices.Length; i++)
            {
                if (devices[i].AcceleratorType.ToString() == "CPU")
                {
                    return new IlgpuBackendSelection(IlgpuBackendKind.IlgpuCpu, i, devices[i].Name, openClFailure);
                }
            }

            return new IlgpuBackendSelection(
                IlgpuBackendKind.CpuFallback, -1, "nenhum", openClFailure + "; acelerador CPU do ILGPU indisponível");
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
}
