using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.CPU;
using ILGPU.Runtime.OpenCL;
using Evo.Ilgpu;

namespace Evo.Ilgpu.Tests;

/// <summary>
/// MemberData compartilhado (EVO-102 / 010a): lista os dispositivos ILGPU visíveis neste PC. Cada linha é
/// <c>(rótulo, seleção, causa)</c>. Dispositivos utilizáveis têm <c>causa == null</c>; dispositivos indisponíveis
/// (ex.: OpenCL não encontrado ou não inicializável) aparecem com a causa e não podem ser aprovados.
/// </summary>
public static class IlgpuDeviceCatalog
{
    public static IEnumerable<object?[]> Entries()
    {
        foreach (var entry in Probe())
        {
            yield return new object?[] { entry.Label, entry.Selection, entry.UnavailableCause };
        }
    }

    public static IReadOnlyList<CatalogEntry> Probe()
    {
        var entries = new List<CatalogEntry>();
        var cpu = TryCpuSelection(out var cpuError);
        entries.Add(cpu is not null
            ? new CatalogEntry("ILGPU/CPU", cpu, null)
            : new CatalogEntry("ILGPU/CPU", null, "acelerador CPU do ILGPU indisponível: " + cpuError));

        var processSelection = IlgpuBackendSelector.Selection;
        if (processSelection.Kind == IlgpuBackendKind.OpenClDevice)
        {
            entries.Add(new CatalogEntry("ILGPU/OpenCL:" + processSelection.DiagnosticDeviceName, processSelection, null));
        }
        else
        {
            entries.Add(new CatalogEntry(
                "ILGPU/OpenCL:indisponível",
                null,
                "nenhum dispositivo OpenCL utilizável: " + processSelection.FallbackReason));
        }

        return entries;
    }

    private static IlgpuBackendSelection? TryCpuSelection(out string? error)
    {
        error = null;
        try
        {
            using var context = Context.Create(builder => builder.OpenCL().CPU());
            var devices = context.Devices.ToArray();
            for (var i = 0; i < devices.Length; i++)
            {
                var device = devices[i];
                if (device.AcceleratorType != AcceleratorType.CPU)
                {
                    continue;
                }

                using var accelerator = device.CreateAccelerator(context);
                return new IlgpuBackendSelection(
                    IlgpuBackendKind.IlgpuCpu,
                    i,
                    device.Name,
                    null,
                    accelerator.WarpSize,
                    accelerator.MaxNumThreads,
                    device.NumMultiprocessors,
                    device.MaxNumThreadsPerGroup,
                    ((CPUDevice)device).NumThreads);
            }

            error = "nenhum dispositivo CPU exposto pelo runtime ILGPU";
            return null;
        }
        catch (Exception ex)
        {
            error = ex.GetType().Name;
            return null;
        }
    }

    public sealed record CatalogEntry(string Label, IlgpuBackendSelection? Selection, string? UnavailableCause);
}
