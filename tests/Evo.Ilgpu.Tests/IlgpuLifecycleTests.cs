using ILGPU.Runtime;
using Evo.Core;
using Evo.Core.Functions;
using Evo.Core.Values;
using Evo.Ilgpu;
using Xunit;

namespace Evo.Ilgpu.Tests;

public sealed class IlgpuLifecycleTests
{
    private static int IdOf(IFunctionSet<double[]> functions, string part)
    {
        for (var i = 0; i < functions.Count; i++)
        {
            if (functions.Get(i).Id.Contains(part, StringComparison.OrdinalIgnoreCase))
            {
                return i;
            }
        }

        throw new InvalidOperationException($"função '{part}' não encontrada");
    }

    [Fact]
    public void Constant_array_with_empty_input_returns_clone_of_parameter()
    {
        var functions = StandardDoubleArrayFunctionSet.Instance;
        var shape = new GenomeShape(InputCount: 1, NodeCount: 1, OutputCount: 1, MaxArity: 2, LevelsBack: 1);
        var genome = Genome<double[]>.FromGenes(shape, new[] { IdOf(functions, "const") }, new[] { 0, 0 },
            new[] { new[] { 4.0, 5.0 } }, new[] { 1 }, DoubleArrayValueSemantics.Instance, functions);
        var inputs = new[] { Array.Empty<double>() };
        var expected = new double[1][];
        var actual = new double[1][];

        new ReferenceGenomeEvaluator<double[]>(functions).Evaluate(genome, inputs, expected);
        using var ilgpu = new IlgpuGenomeEvaluator<double[]>(functions);
        ilgpu.Evaluate(genome, inputs, actual);

        Assert.Equal(expected[0], actual[0]);
        Assert.Equal(new[] { 4.0, 5.0 }, actual[0]);
    }

    [Fact]
    public void Add_between_constant_and_empty_input_throws_same_error_as_reference()
    {
        var functions = StandardDoubleArrayFunctionSet.Instance;
        var shape = new GenomeShape(InputCount: 1, NodeCount: 2, OutputCount: 1, MaxArity: 2, LevelsBack: 2);
        var genome = Genome<double[]>.FromGenes(shape,
            new[] { IdOf(functions, "const"), IdOf(functions, "add") },
            new[] { 0, 0, 1, 0 },
            new[] { new[] { 4.0, 5.0 }, Array.Empty<double>() },
            new[] { 2 }, DoubleArrayValueSemantics.Instance, functions);
        var inputs = new[] { Array.Empty<double>() };

        var expected = Assert.ThrowsAny<Exception>(() =>
            new ReferenceGenomeEvaluator<double[]>(functions).Evaluate(genome, inputs, new double[1][]));
        using var ilgpu = new IlgpuGenomeEvaluator<double[]>(functions);
        var actual = Assert.ThrowsAny<Exception>(() => ilgpu.Evaluate(genome, inputs, new double[1][]));

        Assert.Equal(expected.GetType(), actual.GetType());
    }

    [Fact]
    public void Add_between_valid_empty_arrays_returns_empty_output()
    {
        var functions = StandardDoubleArrayFunctionSet.Instance;
        var shape = new GenomeShape(InputCount: 1, NodeCount: 1, OutputCount: 1, MaxArity: 2, LevelsBack: 1);
        var genome = Genome<double[]>.FromGenes(shape, new[] { IdOf(functions, "add") }, new[] { 0, 0 },
            new[] { Array.Empty<double>() }, new[] { 1 }, DoubleArrayValueSemantics.Instance, functions);
        var inputs = new[] { Array.Empty<double>() };
        var actual = new double[1][];

        using var ilgpu = new IlgpuGenomeEvaluator<double[]>(functions);
        ilgpu.Evaluate(genome, inputs, actual);

        Assert.Empty(actual[0]);
        Assert.Equal("CPU fallback", ilgpu.Backend);
    }

    [Fact]
    public void Mismatched_array_lengths_fall_back_with_reason_and_device_returns_after_valid_call()
    {
        var functions = StandardDoubleArrayFunctionSet.Instance;
        var shape = new GenomeShape(InputCount: 2, NodeCount: 1, OutputCount: 1, MaxArity: 2, LevelsBack: 1);
        var genome = Genome<double[]>.FromGenes(shape, new[] { IdOf(functions, "add") }, new[] { 0, 1 },
            new[] { Array.Empty<double>() }, new[] { 2 }, DoubleArrayValueSemantics.Instance, functions);
        var mismatched = new[] { new[] { 1.0, 2.0 }, new[] { 3.0 } };
        var valid = new[] { new[] { 1.0, 2.0 }, new[] { 3.0, 4.0 } };

        using var ilgpu = new IlgpuGenomeEvaluator<double[]>(functions);
        Assert.ThrowsAny<Exception>(() => ilgpu.Evaluate(genome, mismatched, new double[1][]));
        Assert.Equal("CPU fallback", ilgpu.Backend);
        Assert.Equal("nenhum", ilgpu.DeviceName);
        Assert.False(string.IsNullOrEmpty(ilgpu.LastFallbackReason));

        var outputs = new double[1][];
        ilgpu.Evaluate(genome, valid, outputs);
        Assert.Equal(new[] { 4.0, 6.0 }, outputs[0]);
        Assert.StartsWith("ILGPU", ilgpu.Backend);
    }

    [Fact]
    public void Dispose_releases_accelerator_then_context_and_is_idempotent()
    {
        var functions = StandardDoubleArrayFunctionSet.Instance;
        var ilgpu = new IlgpuGenomeEvaluator<double[]>(functions);
        var accelerator = ilgpu.InternalAccelerator;
        var context = ilgpu.InternalContext;
        Assert.NotNull(accelerator);
        Assert.NotNull(context);

        ilgpu.Dispose();
        ilgpu.Dispose();

        Assert.True(accelerator!.IsDisposed);
        Assert.True(context!.IsDisposed);
    }

    [Fact]
    public void Dispose_after_failed_call_still_releases_both()
    {
        var functions = StandardDoubleArrayFunctionSet.Instance;
        var shape = new GenomeShape(InputCount: 2, NodeCount: 1, OutputCount: 1, MaxArity: 2, LevelsBack: 1);
        var genome = Genome<double[]>.FromGenes(shape, new[] { IdOf(functions, "add") }, new[] { 0, 1 },
            new[] { Array.Empty<double>() }, new[] { 2 }, DoubleArrayValueSemantics.Instance, functions);
        var ilgpu = new IlgpuGenomeEvaluator<double[]>(functions);
        var accelerator = ilgpu.InternalAccelerator;
        var context = ilgpu.InternalContext;

        Assert.ThrowsAny<Exception>(() => ilgpu.Evaluate(genome, new[] { new[] { 1.0 }, new[] { 1.0, 2.0 } }, new double[1][]));
        ilgpu.Dispose();

        Assert.True(accelerator!.IsDisposed);
        Assert.True(context!.IsDisposed);
    }

    [Fact]
    public void Context_exposes_no_cuda_devices()
    {
        var functions = StandardDoubleArrayFunctionSet.Instance;
        using var ilgpu = new IlgpuGenomeEvaluator<double[]>(functions);
        var context = ilgpu.InternalContext;
        Assert.NotNull(context);

        Assert.DoesNotContain(context!.Devices.ToArray(), d => d.AcceleratorType == AcceleratorType.Cuda);
    }

    [Fact]
    public void Int_array_constant_with_empty_input_returns_clone_and_mismatched_add_throws_like_reference()
    {
        var functions = StandardInt32ArrayFunctionSet.Instance;
        var constId = -1;
        var addId = -1;
        for (var i = 0; i < functions.Count; i++)
        {
            var id = functions.Get(i).Id;
            if (constId < 0 && id.Contains("const", StringComparison.OrdinalIgnoreCase)) constId = i;
            if (addId < 0 && id.Contains("add", StringComparison.OrdinalIgnoreCase)) addId = i;
        }

        var constantShape = new GenomeShape(InputCount: 1, NodeCount: 1, OutputCount: 1, MaxArity: 2, LevelsBack: 1);
        var constant = Genome<int[]>.FromGenes(constantShape, new[] { constId }, new[] { 0, 0 },
            new[] { new[] { 4, 5 } }, new[] { 1 }, Int32ArrayValueSemantics.Instance, functions);
        var emptyInputs = new[] { Array.Empty<int>() };
        var expected = new int[1][];
        var actual = new int[1][];
        new ReferenceGenomeEvaluator<int[]>(functions).Evaluate(constant, emptyInputs, expected);
        using (var ilgpu = new IlgpuGenomeEvaluator<int[]>(functions))
        {
            ilgpu.Evaluate(constant, emptyInputs, actual);
        }

        Assert.Equal(new[] { 4, 5 }, actual[0]);
        Assert.Equal(expected[0], actual[0]);

        var addShape = new GenomeShape(InputCount: 1, NodeCount: 2, OutputCount: 1, MaxArity: 2, LevelsBack: 2);
        var add = Genome<int[]>.FromGenes(addShape, new[] { constId, addId }, new[] { 0, 0, 1, 0 },
            new[] { new[] { 4, 5 }, Array.Empty<int>() }, new[] { 2 }, Int32ArrayValueSemantics.Instance, functions);
        var referenceError = Assert.ThrowsAny<Exception>(() =>
            new ReferenceGenomeEvaluator<int[]>(functions).Evaluate(add, emptyInputs, new int[1][]));
        using var ilgpuAdd = new IlgpuGenomeEvaluator<int[]>(functions);
        var ilgpuError = Assert.ThrowsAny<Exception>(() => ilgpuAdd.Evaluate(add, emptyInputs, new int[1][]));
        Assert.Equal(referenceError.GetType(), ilgpuError.GetType());
    }
}
