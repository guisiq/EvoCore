using System.Threading.Tasks;
using ILGPU.Runtime;
using Evo.Core;
using Evo.Core.Functions;
using Evo.Core.Values;
using Evo.Ilgpu;
using Xunit;

namespace Evo.Ilgpu.Tests;

/// <summary>
/// R1-4 (015-v3-cuda-backend): ciclo de vida real em CUDA — descarte, falhas injetadas pós-seleção
/// (launch/sync), ausência de mistura entre instâncias concorrentes ("workers") e determinismo bit a bit
/// com 1/2/N instâncias simultâneas. Sem CUDA válido, cada teste é reportado como ignorado (skip dinâmico
/// visível), nunca como aprovado.
/// </summary>
public sealed class IlgpuCudaLifecycleTests
{
    private static GenomeShape Shape => new(InputCount: 2, NodeCount: 1, OutputCount: 1, MaxArity: 2, LevelsBack: 1);

    private static IlgpuBackendSelection SelectCudaOrSkip()
    {
        try
        {
            return IlgpuBackendSelector.SelectFor(IlgpuBackendPolicy.CudaRequired);
        }
        catch (InvalidOperationException ex)
        {
            throw new DynamicSkipException("CUDA indisponível nesta máquina: " + ex.Message);
        }
    }

    private static Genome<double> DoubleGenome(string functionPart)
    {
        var functions = StandardDoubleFunctionSet.Instance;
        return Genome<double>.FromGenes(Shape, [IlgpuTestHelpers.FindFunctionId(functions, functionPart)], [0, 1],
            [0.0], [2], DoubleValueSemantics.Instance, functions);
    }

    [OptionalFact]
    public void Dispose_releases_cuda_accelerator_and_context_and_is_idempotent()
    {
        var selection = SelectCudaOrSkip();

        var functions = StandardDoubleFunctionSet.Instance;
        var genome = DoubleGenome("add");

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

    [OptionalFact]
    public void Launch_failure_after_cuda_selection_is_explicit_and_disposes_buffers_without_fallback()
    {
        var selection = SelectCudaOrSkip();

        var functions = StandardDoubleFunctionSet.Instance;
        var genome = DoubleGenome("add");
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

    [OptionalFact]
    public void Sync_failure_after_cuda_launch_is_explicit_and_disposes_buffers_without_fallback()
    {
        var selection = SelectCudaOrSkip();

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
    /// Workers concorrentes (metade com falha injetada pós-seleção, metade saudável) sobre dois genomas e duas
    /// entradas: a falha de um worker não pode contaminar os demais, e os saudáveis devem coincidir bit a bit
    /// com a linha de base de uma única instância.
    /// </summary>
    [OptionalFact]
    public void Failure_in_one_worker_instance_does_not_contaminate_concurrent_cuda_instances()
    {
        var functions = StandardDoubleFunctionSet.Instance;
        var genomes = new[] { DoubleGenome("add"), DoubleGenome("mul") };
        var inputSets = new[] { new[] { 1.75, -3.5 }, new[] { 0.5, 1e-3 } };
        var workerCount = Math.Max(4, Environment.ProcessorCount);

        var baseline = new double[genomes.Length, inputSets.Length];
        using (var reference = new IlgpuGenomeEvaluator<double>(functions, IlgpuBackendPolicy.CudaRequired))
        {
            for (var g = 0; g < genomes.Length; g++)
            {
                for (var i = 0; i < inputSets.Length; i++)
                {
                    var output = new double[1];
                    reference.Evaluate(genomes[g], inputSets[i], output);
                    Assert.True(reference.LastEvaluationUsedDevice);
                    baseline[g, i] = output[0];
                }
            }
        }

        var selections = new IlgpuBackendSelection[workerCount];
        for (var w = 0; w < workerCount; w++)
        {
            selections[w] = SelectCudaOrSkip();
        }

        var failedExplicitly = new bool[workerCount];
        var mismatches = new int[workerCount];
        Parallel.For(0, workerCount, new ParallelOptions { MaxDegreeOfParallelism = workerCount }, w =>
        {
            var failing = w % 2 == 0;
            var evaluator = failing
                ? new IlgpuGenomeEvaluator<double>(
                    functions, fallback: null, selectionOverride: selections[w], contextFactory: null,
                    acceleratorFactory: null, beforeKernelExecution: null,
                    launchInvoker: _ => throw new InvalidOperationException("falha-worker-" + w))
                : new IlgpuGenomeEvaluator<double>(functions, IlgpuBackendPolicy.CudaRequired);
            using (evaluator)
            {
                for (var g = 0; g < genomes.Length; g++)
                {
                    for (var i = 0; i < inputSets.Length; i++)
                    {
                        var output = new double[1];
                        if (failing)
                        {
                            var error = Assert.Throws<InvalidOperationException>(() => evaluator.Evaluate(genomes[g], inputSets[i], output));
                            Assert.Equal("falha-worker-" + w, error.Message);
                            failedExplicitly[w] = true;
                            continue;
                        }

                        evaluator.Evaluate(genomes[g], inputSets[i], output);
                        if (!evaluator.LastEvaluationUsedDevice
                            || BitConverter.DoubleToInt64Bits(output[0]) != BitConverter.DoubleToInt64Bits(baseline[g, i]))
                        {
                            mismatches[w]++;
                        }
                    }
                }
            }
        });

        for (var w = 0; w < workerCount; w++)
        {
            Assert.Equal(w % 2 == 0, failedExplicitly[w]);
            Assert.Equal(0, mismatches[w]);
        }
    }

    /// <summary>
    /// Determinismo: 1, 2 e N instâncias CUDA independentes, executando simultaneamente, produzem a mesma saída
    /// bit a bit para cada genoma e entrada da linha de base de uma única instância.
    /// </summary>
    [OptionalFact]
    public void Deterministic_output_across_one_two_and_n_concurrent_cuda_worker_instances()
    {
        var functions = StandardDoubleFunctionSet.Instance;
        var genomes = new[] { DoubleGenome("add"), DoubleGenome("mul") };
        var inputSets = new[] { new[] { 1.75, -3.5 }, new[] { 0.5, 1e-3 } };
        var counts = new[] { 1, 2, 4, Math.Max(4, Environment.ProcessorCount) };

        var baseline = new long[genomes.Length, inputSets.Length];
        using (var reference = new IlgpuGenomeEvaluator<double>(functions, IlgpuBackendPolicy.CudaRequired))
        {
            for (var g = 0; g < genomes.Length; g++)
            {
                for (var i = 0; i < inputSets.Length; i++)
                {
                    var output = new double[1];
                    reference.Evaluate(genomes[g], inputSets[i], output);
                    Assert.True(reference.LastEvaluationUsedDevice);
                    baseline[g, i] = BitConverter.DoubleToInt64Bits(output[0]);
                }
            }
        }

        foreach (var workerCount in counts)
        {
            var mismatches = new int[workerCount];
            var deviceUsed = new bool[workerCount];
            Parallel.For(0, workerCount, new ParallelOptions { MaxDegreeOfParallelism = workerCount }, w =>
            {
                using var evaluator = new IlgpuGenomeEvaluator<double>(functions, IlgpuBackendPolicy.CudaRequired);
                for (var g = 0; g < genomes.Length; g++)
                {
                    for (var i = 0; i < inputSets.Length; i++)
                    {
                        var output = new double[1];
                        evaluator.Evaluate(genomes[g], inputSets[i], output);
                        deviceUsed[w] = evaluator.LastEvaluationUsedDevice;
                        if (BitConverter.DoubleToInt64Bits(output[0]) != baseline[g, i])
                        {
                            mismatches[w]++;
                        }
                    }
                }
            });

            for (var w = 0; w < workerCount; w++)
            {
                Assert.True(deviceUsed[w], $"worker {w} de {workerCount} não usou o dispositivo CUDA");
                Assert.Equal(0, mismatches[w]);
            }
        }
    }
}
