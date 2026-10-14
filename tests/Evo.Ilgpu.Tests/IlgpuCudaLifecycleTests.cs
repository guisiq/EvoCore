using ILGPU.Runtime;
using Evo.Core;
using Evo.Core.Functions;
using Evo.Core.Values;
using Evo.Ilgpu;
using Xunit;

namespace Evo.Ilgpu.Tests;

/// <summary>
/// R1-4 (015-v3-cuda-backend): ciclo de vida real na GTX 1650 — descarte, falhas injetadas pós-seleção
/// (launch/sync), ausência de mistura entre instâncias ("workers") e determinismo com 1/2/N instâncias.
/// Sem CUDA válido nesta máquina, cada teste registra a limitação e retorna sem aprovar CUDA.
/// </summary>
public sealed class IlgpuCudaLifecycleTests
{
    private static GenomeShape Shape => new(InputCount: 2, NodeCount: 1, OutputCount: 1, MaxArity: 2, LevelsBack: 1);

    private static bool TryCudaSelection(out IlgpuBackendSelection selection)
    {
        try
        {
            selection = IlgpuBackendSelector.SelectFor(IlgpuBackendPolicy.CudaRequired);
            return true;
        }
        catch (InvalidOperationException)
        {
            selection = default!;
            return false;
        }
    }

    [Fact]
    public void Dispose_releases_cuda_accelerator_and_context_and_is_idempotent()
    {
        if (!TryCudaSelection(out var selection))
        {
            return;
        }

        var functions = StandardDoubleFunctionSet.Instance;
        var genome = Genome<double>.FromGenes(Shape, [IlgpuTestHelpers.FindFunctionId(functions, "add")], [0, 1],
            [0.0], [2], DoubleValueSemantics.Instance, functions);

        var evaluator = new IlgpuGenomeEvaluator<double>(
            functions, fallback: null, selectionOverride: selection, contextFactory: null, acceleratorFactory: null,
            beforeKernelExecution: null);
        evaluator.Evaluate(genome, [1.0, 2.0], new double[1]);
        var accelerator = evaluator.InternalAccelerator;
        var context = evaluator.InternalContext;
        Assert.NotNull(accelerator);
        Assert.NotNull(context);

        evaluator.Dispose();
        Assert.True(accelerator!.IsDisposed);
        Assert.True(context!.IsDisposed);

        // Descarte repetido não deve lançar.
        evaluator.Dispose();
        Assert.True(accelerator.IsDisposed);
        Assert.True(context.IsDisposed);

        // Uso após descarte deve ser explícito, nunca silencioso.
        Assert.ThrowsAny<Exception>(() => evaluator.Evaluate(genome, [1.0, 2.0], new double[1]));
    }

    [Fact]
    public void Launch_failure_after_cuda_selection_is_explicit_and_disposes_buffers_without_fallback()
    {
        if (!TryCudaSelection(out var selection))
        {
            return;
        }

        var functions = StandardDoubleFunctionSet.Instance;
        var genome = Genome<double>.FromGenes(Shape, [IlgpuTestHelpers.FindFunctionId(functions, "add")], [0, 1],
            [0.0], [2], DoubleValueSemantics.Instance, functions);
        MemoryBuffer[]? observedBuffers = null;
        var forcedFailure = new InvalidOperationException("falha-lancamento-cuda-double");
        using var evaluator = new IlgpuGenomeEvaluator<double>(
            functions,
            fallback: null,
            selectionOverride: selection,
            contextFactory: null,
            acceleratorFactory: null,
            beforeKernelExecution: null,
            launchInvoker: _ => throw forcedFailure,
            synchronizeInvoker: null,
            bufferObserver: buffers => observedBuffers = buffers.ToArray());

        var error = Assert.Throws<InvalidOperationException>(() => evaluator.Evaluate(genome, [1.0, 2.0], new double[1]));
        Assert.Same(forcedFailure, error);
        Assert.NotNull(observedBuffers);
        Assert.All(observedBuffers!, buffer => Assert.True(buffer.IsDisposed));
        Assert.Equal("ILGPU/CUDA", evaluator.Backend);
        Assert.NotEqual("CPU fallback", evaluator.Backend);
    }

    [Fact]
    public void Sync_failure_after_cuda_launch_is_explicit_and_disposes_buffers_without_fallback()
    {
        if (!TryCudaSelection(out var selection))
        {
            return;
        }

        var functions = StandardInt32FunctionSet.Instance;
        var genome = Genome<int>.FromGenes(Shape, [IlgpuTestHelpers.FindFunctionId(functions, "add")], [0, 1],
            [0], [2], Int32ValueSemantics.Instance, functions);
        MemoryBuffer[]? observedBuffers = null;
        var forcedFailure = new InvalidOperationException("falha-sync-cuda-int");
        using var evaluator = new IlgpuGenomeEvaluator<int>(
            functions,
            fallback: null,
            selectionOverride: selection,
            contextFactory: null,
            acceleratorFactory: null,
            beforeKernelExecution: null,
            launchInvoker: null,
            synchronizeInvoker: _ => throw forcedFailure,
            bufferObserver: buffers => observedBuffers = buffers.ToArray());

        var error = Assert.Throws<InvalidOperationException>(() => evaluator.Evaluate(genome, [1, 2], new int[1]));
        Assert.Same(forcedFailure, error);
        Assert.NotNull(observedBuffers);
        Assert.All(observedBuffers!, buffer => Assert.True(buffer.IsDisposed));
        Assert.Equal("ILGPU/CUDA", evaluator.Backend);
    }

    /// <summary>
    /// Uma falha pós-seleção em uma instância ("worker") não deve contaminar outra instância independente:
    /// cada <see cref="IlgpuGenomeEvaluator{T}"/> mantém seu próprio contexto/acelerador CUDA.
    /// </summary>
    [Fact]
    public void Failure_in_one_worker_instance_does_not_contaminate_another_independent_cuda_instance()
    {
        if (!TryCudaSelection(out var selectionA) || !TryCudaSelection(out var selectionB))
        {
            return;
        }

        var functions = StandardDoubleFunctionSet.Instance;
        var genome = Genome<double>.FromGenes(Shape, [IlgpuTestHelpers.FindFunctionId(functions, "add")], [0, 1],
            [0.0], [2], DoubleValueSemantics.Instance, functions);

        using var failingWorker = new IlgpuGenomeEvaluator<double>(
            functions,
            fallback: null,
            selectionOverride: selectionA,
            contextFactory: null,
            acceleratorFactory: null,
            beforeKernelExecution: null,
            launchInvoker: _ => throw new InvalidOperationException("falha-worker-a"));
        Assert.Throws<InvalidOperationException>(() => failingWorker.Evaluate(genome, [1.0, 2.0], new double[1]));

        using var healthyWorker = new IlgpuGenomeEvaluator<double>(functions, IlgpuBackendPolicy.CudaRequired);
        var output = new double[1];
        healthyWorker.Evaluate(genome, [1.0, 2.0], output);
        Assert.True(healthyWorker.LastEvaluationUsedDevice);
        Assert.Equal(3.0, output[0]);
    }

    /// <summary>Determinismo: 1, 2 e N instâncias CUDA independentes produzem a mesma saída para o mesmo genoma.</summary>
    [Fact]
    public void Deterministic_output_across_one_two_and_n_independent_cuda_worker_instances()
    {
        if (!TryCudaSelection(out _))
        {
            return;
        }

        var functions = StandardDoubleFunctionSet.Instance;
        var genome = Genome<double>.FromGenes(Shape, [IlgpuTestHelpers.FindFunctionId(functions, "mul")], [0, 1],
            [0.0], [2], DoubleValueSemantics.Instance, functions);
        var inputs = new[] { 1.75, -3.5 };

        foreach (var workerCount in new[] { 1, 2, 4 })
        {
            var results = new double[workerCount];
            for (var w = 0; w < workerCount; w++)
            {
                using var evaluator = new IlgpuGenomeEvaluator<double>(functions, IlgpuBackendPolicy.CudaRequired);
                var output = new double[1];
                evaluator.Evaluate(genome, inputs, output);
                Assert.True(evaluator.LastEvaluationUsedDevice);
                results[w] = output[0];
            }

            for (var w = 1; w < workerCount; w++)
            {
                Assert.Equal(
                    BitConverter.DoubleToInt64Bits(results[0]),
                    BitConverter.DoubleToInt64Bits(results[w]));
            }
        }
    }
}
