using System.Diagnostics;
using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.CPU;
using ILGPU.Runtime.OpenCL;
using Evo.Core;
using Evo.Core.Functions;
using Evo.Core.Values;
using Evo.Ilgpu;
using Xunit;

namespace Evo.Ilgpu.Tests;

public sealed class IlgpuLifecycleTests
{
    private static IlgpuBackendSelection CpuSelection()
    {
        using var context = Context.Create(builder => builder.OpenCL().CPU());
        var devices = context.Devices.ToArray();
        for (var i = 0; i < devices.Length; i++)
        {
            if (devices[i].AcceleratorType == AcceleratorType.CPU)
            {
                using var accelerator = devices[i].CreateAccelerator(context);
                return new IlgpuBackendSelection(
                    IlgpuBackendKind.IlgpuCpu, i, devices[i].Name, "teste", accelerator.WarpSize, accelerator.MaxNumThreads);
            }
        }

        throw new InvalidOperationException("dispositivo CPU do ILGPU não encontrado");
    }

    [Fact]
    public void Constant_array_with_empty_input_returns_clone_of_parameter()
    {
        var functions = StandardDoubleArrayFunctionSet.Instance;
        var shape = new GenomeShape(InputCount: 1, NodeCount: 1, OutputCount: 1, MaxArity: 2, LevelsBack: 1);
        var genome = Genome<double[]>.FromGenes(shape, [IlgpuTestHelpers.FindFunctionId(functions, "const")], [0, 0],
            [new[] { 4.0, 5.0 }], [1], DoubleArrayValueSemantics.Instance, functions);
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
            [IlgpuTestHelpers.FindFunctionId(functions, "const"), IlgpuTestHelpers.FindFunctionId(functions, "add")],
            [0, 0, 1, 0],
            [new[] { 4.0, 5.0 }, Array.Empty<double>()],
            [2],
            DoubleArrayValueSemantics.Instance,
            functions);
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
        var genome = Genome<double[]>.FromGenes(shape, [IlgpuTestHelpers.FindFunctionId(functions, "add")], [0, 0],
            [Array.Empty<double>()], [1], DoubleArrayValueSemantics.Instance, functions);
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
        var genome = Genome<double[]>.FromGenes(shape, [IlgpuTestHelpers.FindFunctionId(functions, "add")], [0, 1],
            [Array.Empty<double>()], [2], DoubleArrayValueSemantics.Instance, functions);
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
    public void Direct_int_array_outputs_with_different_lengths_are_valid_via_cpu_fallback()
    {
        var functions = StandardInt32ArrayFunctionSet.Instance;
        var shape = new GenomeShape(InputCount: 2, NodeCount: 1, OutputCount: 2, MaxArity: 1, LevelsBack: 1);
        var genome = Genome<int[]>.FromGenes(shape, [IlgpuTestHelpers.FindFunctionId(functions, "const")], [0],
            [new[] { 9 }], [0, 1], Int32ArrayValueSemantics.Instance, functions);
        var inputs = new[] { new[] { 1, 2 }, new[] { 3, 4, 5 } };
        var outputs = new int[2][];

        using var ilgpu = new IlgpuGenomeEvaluator<int[]>(functions);
        ilgpu.Evaluate(genome, inputs, outputs);

        Assert.Equal(inputs[0], outputs[0]);
        Assert.Equal(inputs[1], outputs[1]);
        Assert.Equal("CPU fallback", ilgpu.Backend);
        Assert.Equal("nenhum", ilgpu.DeviceName);
        Assert.False(string.IsNullOrEmpty(ilgpu.LastFallbackReason));
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
        var genome = Genome<double[]>.FromGenes(shape, [IlgpuTestHelpers.FindFunctionId(functions, "add")], [0, 1],
            [Array.Empty<double>()], [2], DoubleArrayValueSemantics.Instance, functions);
        var ilgpu = new IlgpuGenomeEvaluator<double[]>(functions);
        var accelerator = ilgpu.InternalAccelerator;
        var context = ilgpu.InternalContext;

        Assert.ThrowsAny<Exception>(() => ilgpu.Evaluate(genome, new[] { new[] { 1.0 }, new[] { 1.0, 2.0 } }, new double[1][]));
        ilgpu.Dispose();

        Assert.True(accelerator!.IsDisposed);
        Assert.True(context!.IsDisposed);
    }

    [Fact]
    public void Construction_failure_releases_partial_context_and_falls_back_with_cause()
    {
        var functions = StandardDoubleFunctionSet.Instance;
        var selection = CpuSelection();
        Context? createdContext = null;
        using var evaluator = new IlgpuGenomeEvaluator<double>(
            functions,
            fallback: null,
            selectionOverride: selection,
            contextFactory: _ =>
            {
                createdContext = Context.Create(builder => builder.CPU());
                return createdContext;
            },
            acceleratorFactory: static (_, _) => throw new InvalidOperationException("falha-forcada"),
            beforeKernelExecution: null);

        Assert.NotNull(createdContext);
        Assert.True(createdContext!.IsDisposed);
        Assert.Equal("CPU fallback", evaluator.Backend);
        Assert.Equal("nenhum", evaluator.DeviceName);
        Assert.Contains("InvalidOperationException", evaluator.LastFallbackReason);
    }

    [Fact]
    public void Construction_failure_with_out_of_memory_still_releases_context_and_rethrows_original()
    {
        var functions = StandardDoubleFunctionSet.Instance;
        var selection = CpuSelection();
        Context? createdContext = null;

        var thrown = Assert.Throws<OutOfMemoryException>(() =>
            new IlgpuGenomeEvaluator<double>(
                functions,
                fallback: null,
                selectionOverride: selection,
                contextFactory: _ =>
                {
                    createdContext = Context.Create(builder => builder.CPU());
                    return createdContext;
                },
                acceleratorFactory: static (_, _) => throw new OutOfMemoryException("sem-memoria"),
                beforeKernelExecution: null));

        Assert.Equal("sem-memoria", thrown.Message);
        Assert.NotNull(createdContext);
        Assert.True(createdContext!.IsDisposed);
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
    public void Selector_reports_cpu_fallback_when_runtime_initialization_fails()
    {
        var selection = IlgpuBackendSelector.ProbeForTests(
            static () => throw new DllNotFoundException("runtime-ausente"),
            static (_, _) => throw new UnreachableException());

        Assert.Equal(IlgpuBackendKind.CpuFallback, selection.Kind);
        Assert.Equal("nenhum", selection.DeviceName);
        Assert.Contains("DllNotFoundException", selection.FallbackReason);
    }

    [Fact]
    public void Evaluator_rechecks_device_identity_in_new_context_and_falls_back_when_it_changes()
    {
        var functions = StandardDoubleFunctionSet.Instance;
        var selection = CpuSelection() with { DeviceName = "dispositivo-inexistente" };
        using var evaluator = new IlgpuGenomeEvaluator<double>(
            functions,
            fallback: null,
            selectionOverride: selection,
            contextFactory: _ => Context.Create(builder => builder.CPU()),
            acceleratorFactory: null,
            beforeKernelExecution: null);

        Assert.Equal("CPU fallback", evaluator.Backend);
        Assert.Equal("nenhum", evaluator.DeviceName);
        Assert.Contains("InvalidOperationException", evaluator.LastFallbackReason);
    }

    [Fact]
    public void Evaluator_falls_back_when_device_capabilities_diverge_from_selection()
    {
        var functions = StandardDoubleFunctionSet.Instance;
        var real = CpuSelection();
        // Capacidades divergentes da sondagem original: a reprodução reportada seleciona
        // NumThreads=16/WarpSize=4 e reinicializa com NumThreads=1024/WarpSize=32.
        var mismatched = real with { WarpSize = real.WarpSize + 1, MaxNumThreads = real.MaxNumThreads + 1 };
        using var evaluator = new IlgpuGenomeEvaluator<double>(
            functions,
            fallback: null,
            selectionOverride: mismatched,
            contextFactory: _ => Context.Create(builder => builder.CPU()),
            acceleratorFactory: null,
            beforeKernelExecution: null);

        Assert.Equal("CPU fallback", evaluator.Backend);
        Assert.Equal("nenhum", evaluator.DeviceName);
        Assert.Contains("InvalidOperationException", evaluator.LastFallbackReason);
    }

    [Fact]
    public void Kernel_execution_failures_remain_explicit_and_are_not_masked_by_fallback()
    {
        var functions = StandardDoubleFunctionSet.Instance;
        var selection = CpuSelection();
        var shape = new GenomeShape(InputCount: 2, NodeCount: 1, OutputCount: 1, MaxArity: 2, LevelsBack: 1);
        var genome = Genome<double>.FromGenes(shape, [IlgpuTestHelpers.FindFunctionId(functions, "add")], [0, 1],
            [0.0], [2], DoubleValueSemantics.Instance, functions);
        using var evaluator = new IlgpuGenomeEvaluator<double>(
            functions,
            fallback: null,
            selectionOverride: selection,
            contextFactory: null,
            acceleratorFactory: null,
            beforeKernelExecution: static () => throw new InvalidOperationException("falha-de-kernel"));

        var error = Assert.Throws<InvalidOperationException>(() => evaluator.Evaluate(genome, [1.0, 2.0], new double[1]));
        Assert.Equal("falha-de-kernel", error.Message);
        Assert.NotEqual("CPU fallback", evaluator.Backend);
    }

    [Fact]
    public void Kernel_execution_failures_after_buffer_acquisition_remain_explicit_for_double()
    {
        var functions = StandardDoubleFunctionSet.Instance;
        var selection = CpuSelection();
        var shape = new GenomeShape(InputCount: 2, NodeCount: 1, OutputCount: 1, MaxArity: 2, LevelsBack: 1);
        var genome = Genome<double>.FromGenes(shape, [IlgpuTestHelpers.FindFunctionId(functions, "add")], [0, 1],
            [0.0], [2], DoubleValueSemantics.Instance, functions);
        using var evaluator = new IlgpuGenomeEvaluator<double>(
            functions,
            fallback: null,
            selectionOverride: selection,
            contextFactory: null,
            acceleratorFactory: null,
            beforeKernelExecution: static () => throw new InvalidOperationException("falha-apos-buffers-double"));

        var error = Assert.Throws<InvalidOperationException>(() => evaluator.Evaluate(genome, [1.0, 2.0], new double[1]));
        Assert.Equal("falha-apos-buffers-double", error.Message);
        Assert.NotEqual("CPU fallback", evaluator.Backend);

        var context = evaluator.InternalContext;
        Assert.NotNull(context);
        Assert.False(context!.IsDisposed);
    }

    [Fact]
    public void Kernel_execution_failures_after_buffer_acquisition_remain_explicit_for_int()
    {
        var functions = StandardInt32FunctionSet.Instance;
        var selection = CpuSelection();
        var shape = new GenomeShape(InputCount: 2, NodeCount: 1, OutputCount: 1, MaxArity: 2, LevelsBack: 1);
        var genome = Genome<int>.FromGenes(shape, [IlgpuTestHelpers.FindFunctionId(functions, "add")], [0, 1],
            [0], [2], Int32ValueSemantics.Instance, functions);
        using var evaluator = new IlgpuGenomeEvaluator<int>(
            functions,
            fallback: null,
            selectionOverride: selection,
            contextFactory: null,
            acceleratorFactory: null,
            beforeKernelExecution: static () => throw new InvalidOperationException("falha-apos-buffers-int"));

        var error = Assert.Throws<InvalidOperationException>(() => evaluator.Evaluate(genome, [1, 2], new int[1]));
        Assert.Equal("falha-apos-buffers-int", error.Message);
        Assert.NotEqual("CPU fallback", evaluator.Backend);

        var context = evaluator.InternalContext;
        Assert.NotNull(context);
        Assert.False(context!.IsDisposed);
    }

    [Fact]
    public void Int_array_constant_with_empty_input_returns_clone_and_mismatched_add_throws_like_reference()
    {
        var functions = StandardInt32ArrayFunctionSet.Instance;
        var constId = IlgpuTestHelpers.FindFunctionId(functions, "const");
        var addId = IlgpuTestHelpers.FindFunctionId(functions, "add");

        var constantShape = new GenomeShape(InputCount: 1, NodeCount: 1, OutputCount: 1, MaxArity: 2, LevelsBack: 1);
        var constant = Genome<int[]>.FromGenes(constantShape, [constId], [0, 0],
            [new[] { 4, 5 }], [1], Int32ArrayValueSemantics.Instance, functions);
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
        var add = Genome<int[]>.FromGenes(addShape, [constId, addId], [0, 0, 1, 0],
            [new[] { 4, 5 }, Array.Empty<int>()], [2], Int32ArrayValueSemantics.Instance, functions);
        var referenceError = Assert.ThrowsAny<Exception>(() =>
            new ReferenceGenomeEvaluator<int[]>(functions).Evaluate(add, emptyInputs, new int[1][]));
        using var ilgpuAdd = new IlgpuGenomeEvaluator<int[]>(functions);
        var ilgpuError = Assert.ThrowsAny<Exception>(() => ilgpuAdd.Evaluate(add, emptyInputs, new int[1][]));
        Assert.Equal(referenceError.GetType(), ilgpuError.GetType());
    }
}
