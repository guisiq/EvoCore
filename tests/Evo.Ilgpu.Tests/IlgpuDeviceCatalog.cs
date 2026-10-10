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

    /// <summary>Entrada aprovável: seleção presente e sem causa de indisponibilidade.</summary>
    public static bool IsApproved(CatalogEntry entry) => entry.Selection is not null && entry.UnavailableCause is null;

    public static IReadOnlyList<CatalogEntry> Probe()
    {
        var entries = new List<CatalogEntry>();
        var cpu = TryCpuSelection(out var cpuError);
        entries.Add(cpu is not null
            ? new CatalogEntry("ILGPU/CPU", cpu, null)
            : new CatalogEntry("ILGPU/CPU", null, "acelerador CPU do ILGPU indisponível: " + cpuError));

        try
        {
            using var context = Context.Create(builder => builder.OpenCL().CPU());
            var devices = context.Devices.ToArray();
            var openClCandidates = new List<IlgpuBackendSelection>();
            for (var i = 0; i < devices.Length; i++)
            {
                if (devices[i] is CLDevice openCl)
                {
                    openClCandidates.Add(CreateOpenClSelection(openCl, i));
                }
            }

            entries.AddRange(BuildOpenClEntries(openClCandidates, candidate =>
            {
                try
                {
                    using var accelerator = devices[candidate.DeviceIndex].CreateAccelerator(context);
                    return null;
                }
                catch (Exception ex) when (ex is not OutOfMemoryException)
                {
                    return ex;
                }
            }));
        }
        catch (Exception ex)
        {
            entries.Add(new CatalogEntry(
                "ILGPU/OpenCL:indisponível",
                null,
                "falha de inicialização do runtime ILGPU/OpenCL: " + ex.GetType().Name));
        }

        return entries;
    }

    /// <summary>
    /// Converte cada dispositivo OpenCL enumerado em uma linha do catálogo, com identidade completa e causa
    /// individual quando o candidato não é utilizável. Não depende da seleção global do processo: um candidato
    /// inutilizável não oculta os demais. Sem candidatos, registra uma única linha indisponível com a causa.
    /// </summary>
    internal static IReadOnlyList<CatalogEntry> BuildOpenClEntries(
        IEnumerable<IlgpuBackendSelection> openClCandidates,
        Func<IlgpuBackendSelection, Exception?> validator)
    {
        var entries = new List<CatalogEntry>();
        foreach (var candidate in openClCandidates)
        {
            var label = "ILGPU/OpenCL:" + DescribeOpenClIdentity(candidate);
            var error = validator(candidate);
            entries.Add(error is null
                ? new CatalogEntry(label, candidate, null)
                : new CatalogEntry(
                    label,
                    candidate,
                    $"dispositivo OpenCL '{DescribeOpenClIdentity(candidate)}' não utilizável: {error.GetType().Name}"));
        }

        if (entries.Count == 0)
        {
            entries.Add(new CatalogEntry(
                "ILGPU/OpenCL:indisponível",
                null,
                "nenhum dispositivo OpenCL encontrado pelo runtime ILGPU"));
        }

        return entries;
    }

    /// <summary>Identidade textual completa: plataforma, nome e identificadores OpenCL (distingue homônimos).</summary>
    internal static string DescribeOpenClIdentity(IlgpuBackendSelection selection) =>
        $"{selection.PlatformName} :: {selection.DeviceName} [platformId={selection.PlatformId}, deviceId={selection.OpenClDeviceId}]";

    private static IlgpuBackendSelection CreateOpenClSelection(CLDevice device, int index) =>
        new(
            IlgpuBackendKind.OpenClDevice,
            index,
            device.Name,
            null,
            device.WarpSize,
            device.MaxNumThreads,
            device.NumMultiprocessors,
            device.MaxNumThreadsPerGroup,
            CpuNumThreads: 0,
            PlatformName: device.PlatformName,
            PlatformId: device.PlatformId.ToString(),
            OpenClDeviceId: device.DeviceId.ToString());

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
