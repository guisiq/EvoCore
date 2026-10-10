using System.Collections.Concurrent;
using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.CPU;
using Evo.Core;
using Evo.Core.Functions;
using Evo.Core.Values;
using Evo.Ilgpu;
using Xunit;

namespace Evo.Ilgpu.Tests;

public sealed class IlgpuWorkerCountTests
{
    private sealed class TrackingFitness : IFitnessEvaluator<double>
    {
        private readonly Action<int>? _afterCase;
        private readonly double[] _cases = [1.0, -2.0, 3.5];

        public TrackingFitness(Action<int>? afterCase = null)
        {
            _afterCase = afterCase;
        }

        public int CallCount { get; private set; }

        public FitnessResult Evaluate(Genome<double> genome, IGenomeEvaluator<double> evaluator, CancellationToken cancellationToken)
        {
            cancellationToken.ThrowIfCancellationRequested();
            var output = new double[1];
            var sum = 0.0;
            foreach (var value in _cases)
            {
                evaluator.Evaluate(genome, [value, 1.0], output);
                CallCount++;
                _afterCase?.Invoke(CallCount);
                cancellationToken.ThrowIfCancellationRequested();
                sum += output[0];
            }

            return FitnessResult.Valid(sum);
        }
    }

    private static Genome<double> DoubleAddGenome(IFunctionSet<double> functions)
    {
        var shape = new GenomeShape(InputCount: 2, NodeCount: 1, OutputCount: 1, MaxArity: 2, LevelsBack: 1);
        return Genome<double>.FromGenes(shape, [IlgpuTestHelpers.FindFunctionId(functions, "add")], [0, 1],
            [0.0], [2], DoubleValueSemantics.Instance, functions);
    }

    private static Genome<int[]> IntArrayAddGenome(IFunctionSet<int[]> functions)
    {
        var shape = new GenomeShape(InputCount: 2, NodeCount: 1, OutputCount: 1, MaxArity: 2, LevelsBack: 1);
        return Genome<int[]>.FromGenes(shape, [IlgpuTestHelpers.FindFunctionId(functions, "add")], [0, 1],
            [Array.Empty<int>()], [2], Int32ArrayValueSemantics.Instance, functions);
    }

    [Theory]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(4)]
    public void Results_and_backend_identity_are_identical_across_isolated_workers(int workerCount)
    {
        var functions = StandardDoubleFunctionSet.Instance;
        var genome = DoubleAddGenome(functions);
        var inputs = new[] { 1.5, -2.25 };
        var expected = new double[1];
        new ReferenceGenomeEvaluator<double>(functions).Evaluate(genome, inputs, expected);

        var results = new double[workerCount][];
        var backends = new string[workerCount];
        var devices = new string[workerCount];
        var reasons = new string?[workerCount];
        Parallel.For(0, workerCount, w =>
        {
            using var evaluator = new IlgpuGenomeEvaluator<double>(functions);
            var output = new double[1];
            evaluator.Evaluate(genome, inputs, output);
            results[w] = output;
            backends[w] = evaluator.Backend;
            devices[w] = evaluator.DeviceName;
            reasons[w] = evaluator.LastFallbackReason;
        });

        foreach (var result in results)
        {
            Assert.Equal(expected[0], result[0], 12);
        }

        Assert.All(backends, backend => Assert.Equal(backends[0], backend));
        Assert.All(devices, device => Assert.Equal(devices[0], device));
        Assert.All(reasons, reason => Assert.Equal(reasons[0], reason));
    }

    [Theory]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(4)]
    public void Array_results_are_identical_across_isolated_workers(int workerCount)
    {
        var functions = StandardInt32ArrayFunctionSet.Instance;
        var genome = IntArrayAddGenome(functions);
        var inputs = new[] { new[] { 1, 2, 3 }, new[] { 4, 5, 6 } };
        var expected = new int[1][];
        new ReferenceGenomeEvaluator<int[]>(functions).Evaluate(genome, inputs, expected);

        var results = new ConcurrentBag<int[]>();
        Parallel.For(0, workerCount, _ =>
        {
            using var evaluator = new IlgpuGenomeEvaluator<int[]>(functions);
            var output = new int[1][];
            evaluator.Evaluate(genome, inputs, output);
            results.Add(output[0]);
        });

        Assert.Equal(workerCount, results.Count);
        foreach (var result in results)
        {
            Assert.Equal(expected[0], result);
        }
    }

    [Fact]
    public void Fitness_state_is_isolated_across_workers()
    {
        var functions = StandardDoubleFunctionSet.Instance;
        var genome = DoubleAddGenome(functions);
        var calls = new ConcurrentBag<int>();
        var fitnessValues = new ConcurrentBag<double>();

        Parallel.For(0, 4, _ =>
        {
            using var evaluator = new IlgpuGenomeEvaluator<double>(functions);
            var fitness = new TrackingFitness();
            var result = fitness.Evaluate(genome, evaluator, CancellationToken.None);
            calls.Add(fitness.CallCount);
            fitnessValues.Add(result.Value);
        });

        Assert.All(calls, count => Assert.Equal(3, count));
        Assert.All(fitnessValues, value => Assert.Equal(fitnessValues.First(), value, 12));
    }

    [Fact]
    public void Cancellation_before_fitness_evaluation_is_observed_via_existing_contract_and_resources_are_released()
    {
        var functions = StandardDoubleFunctionSet.Instance;
        var genome = DoubleAddGenome(functions);
        var fitness = new TrackingFitness();
        var cts = new CancellationTokenSource();
        cts.Cancel();
        var evaluator = new IlgpuGenomeEvaluator<double>(functions);
        var accelerator = evaluator.InternalAccelerator;
        var context = evaluator.InternalContext;

        Assert.Throws<OperationCanceledException>(() => fitness.Evaluate(genome, evaluator, cts.Token));
        Assert.Equal(0, fitness.CallCount);
        evaluator.Dispose();

        if (accelerator is not null)
        {
            Assert.True(accelerator.IsDisposed);
        }

        if (context is not null)
        {
            Assert.True(context.IsDisposed);
        }
    }

    [Fact]
    public void Cancellation_during_fitness_evaluation_is_observed_via_existing_contract_and_resources_are_released()
    {
        var functions = StandardDoubleFunctionSet.Instance;
        var genome = DoubleAddGenome(functions);
        using var cts = new CancellationTokenSource();
        var fitness = new TrackingFitness(afterCase: callCount =>
        {
            if (callCount == 1)
            {
                cts.Cancel();
            }
        });

        var evaluator = new IlgpuGenomeEvaluator<double>(functions);
        var accelerator = evaluator.InternalAccelerator;
        var context = evaluator.InternalContext;

        Assert.Throws<OperationCanceledException>(() => fitness.Evaluate(genome, evaluator, cts.Token));
        Assert.Equal(1, fitness.CallCount);
        evaluator.Dispose();

        if (accelerator is not null)
        {
            Assert.True(accelerator.IsDisposed);
        }

        if (context is not null)
        {
            Assert.True(context.IsDisposed);
        }
    }

    private static Genome<double> ProtectedLogGenome(IFunctionSet<double> functions)
    {
        var shape = new GenomeShape(InputCount: 2, NodeCount: 1, OutputCount: 1, MaxArity: 2, LevelsBack: 1);
        return Genome<double>.FromGenes(shape, [IlgpuTestHelpers.FindFunctionId(functions, "log")], [0, 1],
            [0.0], [2], DoubleValueSemantics.Instance, functions);
    }

    /// <summary>
    /// Builds a private (non-shared) ILGPU-CPU selection, resolved from a dedicated <see cref="Context"/>, so this
    /// test never touches the process-wide <see cref="IlgpuBackendSelector.Selection"/> singleton used by other
    /// tests that run in parallel in this assembly.
    /// </summary>
    private static (Context Context, IlgpuBackendSelection Selection) CreatePrivateCpuSelection()
    {
        var context = Context.Create(builder => builder.CPU());
        var device = context.Devices.First(d => d.AcceleratorType == AcceleratorType.CPU);
        using var accelerator = device.CreateAccelerator(context);
        var selection = new IlgpuBackendSelection(
            IlgpuBackendKind.IlgpuCpu,
            0,
            device.Name,
            null,
            device.WarpSize,
            device.MaxNumThreads,
            device.NumMultiprocessors,
            device.MaxNumThreadsPerGroup,
            accelerator is CPUAccelerator cpu ? cpu.NumThreads : 0);
        return (context, selection);
    }

    [Theory]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(4)]
    public void One_worker_device_init_failure_forces_cpu_fallback_for_all_workers_sharing_the_selection(int workerCount)
    {
        var (context, selection) = CreatePrivateCpuSelection();
        using var _ = context;
        var functions = StandardDoubleFunctionSet.Instance;
        var genome = ProtectedLogGenome(functions);
        var inputs = new[] { 7.0, 1.0 };

        // Worker 0 falha a inicialização do dispositivo depois da mesma seleção compartilhada.
        using (var failing = new IlgpuGenomeEvaluator<double>(
            functions,
            fallback: null,
            selectionOverride: selection,
            contextFactory: _ => Context.Create(builder => builder.CPU()),
            acceleratorFactory: (_, _) => throw new InvalidOperationException("falha-init-worker-controlada"),
            beforeKernelExecution: null))
        {
            var output = new double[1];
            failing.Evaluate(genome, inputs, output);
            Assert.Equal("CPU fallback", failing.Backend);
        }

        var backends = new string[workerCount];
        var devices = new string[workerCount];
        for (var w = 0; w < workerCount; w++)
        {
            using var evaluator = new IlgpuGenomeEvaluator<double>(
                functions, fallback: null, selectionOverride: selection,
                contextFactory: _ => context, acceleratorFactory: null, beforeKernelExecution: null);
            var output = new double[1];
            evaluator.Evaluate(genome, inputs, output);
            backends[w] = evaluator.Backend;
            devices[w] = evaluator.DeviceName;
        }

        Assert.All(backends, backend => Assert.Equal("CPU fallback", backend));
        Assert.All(devices, device => Assert.Equal("nenhum", device));
    }

    [Fact]
    public void Backend_selector_skips_kernel_probe_failure_and_uses_next_candidate()
    {
        var openClBroken = new IlgpuBackendSelection(
            IlgpuBackendKind.OpenClDevice,
            0,
            "GPU quebrada",
            null,
            PlatformName: "Platform A",
            PlatformId: "10",
            OpenClDeviceId: "20");
        var cpuGood = new IlgpuBackendSelection(
            IlgpuBackendKind.IlgpuCpu,
            1,
            "CPU boa",
            "placeholder",
            WarpSize: 4,
            MaxNumThreads: 16,
            NumMultiprocessors: 4,
            MaxNumThreadsPerGroup: 4,
            CpuNumThreads: 16);

        var selection = IlgpuBackendSelector.ProbeCandidatesForTests(
            [openClBroken],
            [cpuGood],
            candidate => candidate.Kind == IlgpuBackendKind.OpenClDevice
                ? new InvalidOperationException("kernel-load-failed")
                : null);

        Assert.Equal(IlgpuBackendKind.IlgpuCpu, selection.Kind);
        Assert.Equal("CPU boa", selection.DeviceName);
        Assert.Contains("GPU quebrada", selection.FallbackReason);
    }

    [Theory]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(4)]
    public void Selector_candidate_choice_is_consistent_across_worker_counts(int workerCount)
    {
        var results = new ConcurrentBag<IlgpuBackendSelection>();
        Parallel.For(0, workerCount, _ =>
        {
            var selection = IlgpuBackendSelector.ProbeCandidatesForTests(
                [new IlgpuBackendSelection(IlgpuBackendKind.OpenClDevice, 0, "GPU-1", null, PlatformName: "Platform", PlatformId: "1", OpenClDeviceId: "2")],
                [new IlgpuBackendSelection(IlgpuBackendKind.IlgpuCpu, 1, "CPU-1", "placeholder", WarpSize: 4, MaxNumThreads: 16, NumMultiprocessors: 4, MaxNumThreadsPerGroup: 4, CpuNumThreads: 16)],
                _ => null);
            results.Add(selection);
        });

        Assert.Equal(workerCount, results.Count);
        var first = results.First();
        Assert.All(results, selection =>
        {
            Assert.Equal(first.Kind, selection.Kind);
            Assert.Equal(first.DeviceName, selection.DeviceName);
            Assert.Equal(first.PlatformName, selection.PlatformName);
            Assert.Equal(first.OpenClDeviceId, selection.OpenClDeviceId);
        });
    }
}
