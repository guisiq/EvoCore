using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.CPU;
using ILGPU.Runtime.OpenCL;
using Xunit;
using Xunit.Abstractions;

namespace Evo.Ilgpu.Tests;

public sealed class OpenClDeviceCatalogTests
{
    private readonly ITestOutputHelper _output;

    public OpenClDeviceCatalogTests(ITestOutputHelper output) => _output = output;

    [Fact]
    public void Enumerates_every_runtime_device_visible_to_ilgpu_and_reports_identity()
    {
        using var context = Context.Create(builder => builder.OpenCL().CPU());
        var lines = new List<string>();
        var index = 0;
        foreach (var device in context.Devices)
        {
            var line = device is CLDevice cl
                ? $"[{index}] OpenCL | plataforma='{cl.PlatformName}' | dispositivo='{cl.Name}' | platformId={cl.PlatformId} | deviceId={cl.DeviceId}"
                : $"[{index}] {device.AcceleratorType} | dispositivo='{device.Name}'";
            lines.Add(line);
            _output.WriteLine(line);
            index++;
        }

        Assert.NotEmpty(lines);
        Assert.Contains(context.Devices, d => d.AcceleratorType == AcceleratorType.CPU);
    }
}
