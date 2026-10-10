using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.CPU;
using ILGPU.Runtime.OpenCL;
using Evo.Ilgpu;
using Xunit;
using Xunit.Abstractions;

namespace Evo.Ilgpu.Tests;

/// <summary>
/// Reparo R1-1: catálogo completo dos dispositivos visíveis ao runtime ILGPU neste PC (todas as plataformas
/// OpenCL e o acelerador CPU do próprio ILGPU), com identidade completa (plataforma, nome, tipo, identificadores,
/// e demais propriedades expostas pelo runtime, incluindo fabricante/versão quando o driver as relata) e prova de
/// que a identidade é estável entre contextos independentes e distingue dispositivos homônimos.
/// </summary>
public sealed class OpenClDeviceCatalogTests
{
    private readonly ITestOutputHelper _output;

    public OpenClDeviceCatalogTests(ITestOutputHelper output) => _output = output;

    private static string DescribeFullIdentity(Device device)
    {
        var parts = new List<string> { $"tipo={device.AcceleratorType}", $"nome='{device.Name}'" };
        if (device is CLDevice cl)
        {
            parts.Add($"plataforma='{cl.PlatformName}'");
            parts.Add($"platformId={cl.PlatformId}");
            parts.Add($"deviceId={cl.DeviceId}");
        }

        if (device is CPUDevice cpu)
        {
            parts.Add($"numThreads={cpu.NumThreads}");
        }

        parts.Add($"warpSize={device.WarpSize}");
        parts.Add($"maxNumThreads={device.MaxNumThreads}");
        parts.Add($"numMultiprocessors={device.NumMultiprocessors}");
        parts.Add($"maxNumThreadsPerGroup={device.MaxNumThreadsPerGroup}");

        // Além dos campos conhecidos acima, reporta por reflexão qualquer outra propriedade pública exposta
        // pelo runtime (ex.: fabricante/versão de driver quando disponíveis), sem depender de uma API ILGPU
        // específica que pode variar entre versões do pacote.
        foreach (var property in device.GetType().GetProperties().OrderBy(p => p.Name))
        {
            if (property.GetIndexParameters().Length != 0 || property.Name is nameof(Device.Name) or nameof(Device.AcceleratorType)
                or nameof(Device.WarpSize) or nameof(Device.MaxNumThreads) or nameof(Device.NumMultiprocessors) or nameof(Device.MaxNumThreadsPerGroup))
            {
                continue;
            }

            object? value;
            try
            {
                value = property.GetValue(device);
            }
            catch (Exception ex)
            {
                value = $"<erro ao ler: {ex.GetType().Name}>";
            }

            parts.Add($"{property.Name}={value}");
        }

        return string.Join(" | ", parts);
    }

    [Fact]
    public void Enumerates_every_runtime_device_visible_to_ilgpu_and_reports_identity()
    {
        using var context = Context.Create(builder => builder.OpenCL().CPU());
        var lines = new List<string>();
        var index = 0;
        foreach (var device in context.Devices)
        {
            var line = $"[{index}] {DescribeFullIdentity(device)}";
            lines.Add(line);
            _output.WriteLine(line);
            index++;
        }

        Assert.NotEmpty(lines);
        Assert.Contains(context.Devices, d => d.AcceleratorType == AcceleratorType.CPU);

        // A configuração CPU do ILGPU precisa estar explicitamente identificada (NumThreads do dispositivo
        // CPU real desta máquina), não apenas rotulada "CPU" genericamente.
        var cpuDevice = context.Devices.First(d => d.AcceleratorType == AcceleratorType.CPU);
        var cpu = Assert.IsType<CPUDevice>(cpuDevice);
        Assert.True(cpu.NumThreads > 0, "configuração ILGPU CPU deve relatar NumThreads > 0");
    }

    [Theory]
    [MemberData(nameof(IlgpuDeviceMatrixTests.Devices), MemberType = typeof(IlgpuDeviceMatrixTests))]
    public void Catalog_entry_asserts_device_identity(string label, IlgpuBackendSelection selection)
    {
        Assert.False(string.IsNullOrWhiteSpace(label));
        Assert.False(string.IsNullOrWhiteSpace(selection.DeviceName));
        Assert.True(selection.MaxNumThreads > 0, $"{label} deve relatar MaxNumThreads > 0");
        if (selection.Kind == IlgpuBackendKind.OpenClDevice)
        {
            // OpenCL não expõe WarpSize pelo runtime ILGPU (relata 0); a identidade é plataforma + ids.
            Assert.False(string.IsNullOrWhiteSpace(selection.PlatformName), $"{label} deve identificar a plataforma");
            Assert.False(string.IsNullOrWhiteSpace(selection.PlatformId), $"{label} deve identificar PlatformId");
            Assert.False(string.IsNullOrWhiteSpace(selection.OpenClDeviceId), $"{label} deve identificar DeviceId");
        }
        else
        {
            Assert.True(selection.WarpSize > 0, $"{label} deve relatar WarpSize > 0");
            Assert.True(selection.CpuNumThreads > 0, $"{label} deve relatar CpuNumThreads > 0");
        }
    }

    [Fact]
    public void Device_identity_is_stable_across_independent_contexts()
    {
        using var contextA = Context.Create(builder => builder.OpenCL().CPU());
        using var contextB = Context.Create(builder => builder.OpenCL().CPU());
        var devicesA = contextA.Devices.ToArray();
        var devicesB = contextB.Devices.ToArray();

        Assert.Equal(devicesA.Length, devicesB.Length);
        for (var i = 0; i < devicesA.Length; i++)
        {
            Assert.Equal(devicesA[i].AcceleratorType, devicesB[i].AcceleratorType);
            Assert.Equal(devicesA[i].Name, devicesB[i].Name);
            if (devicesA[i] is CLDevice clA)
            {
                var clB = Assert.IsType<CLDevice>(devicesB[i]);
                Assert.Equal(clA.PlatformId, clB.PlatformId);
                Assert.Equal(clA.DeviceId, clB.DeviceId);
                Assert.Equal(clA.PlatformName, clB.PlatformName);
            }
        }
    }

    [Fact]
    public void Divergent_identity_or_configuration_is_rejected_for_homonymous_devices()
    {
        // Dois dispositivos homônimos (mesmo nome de dispositivo e de plataforma) mas com identificadores
        // OpenCL distintos (ex.: dois GPUs idênticos do mesmo fabricante) não podem ser tratados como o mesmo
        // dispositivo: a seleção estável (IlgpuBackendSelection) carrega PlatformId/OpenClDeviceId exatamente
        // para distingui-los, e a igualdade de registro deve rejeitar a identidade divergente.
        var baseline = new IlgpuBackendSelection(
            IlgpuBackendKind.OpenClDevice,
            0,
            "Dispositivo Homônimo",
            null,
            WarpSize: 32,
            MaxNumThreads: 1024,
            NumMultiprocessors: 8,
            MaxNumThreadsPerGroup: 256,
            CpuNumThreads: 0,
            PlatformName: "Plataforma X",
            PlatformId: "platform-0",
            OpenClDeviceId: "device-0");

        var divergentDeviceId = baseline with { OpenClDeviceId = "device-1" };
        var divergentPlatformId = baseline with { PlatformId = "platform-1" };
        var divergentConfig = baseline with { WarpSize = 64 };

        Assert.NotEqual(baseline, divergentDeviceId);
        Assert.NotEqual(baseline, divergentPlatformId);
        Assert.NotEqual(baseline, divergentConfig);
        Assert.Equal(baseline, baseline with { });
    }
}
