using Evo.Core;
using Evo.Core.Functions;
using Evo.Core.Values;
using Evo.Ilgpu;
using Xunit;

namespace Evo.Ilgpu.Tests;

public sealed class IlgpuCudaBackendTests
{
    private static readonly GenomeShape Shape = new(InputCount: 2, NodeCount: 8, OutputCount: 2, MaxArity: 3, LevelsBack: 8);

    private static Genome<double> RandomDouble(ulong seed, IFunctionSet<double> functions)
    {
        var rng = RandomStream.Create(seed, 0, 0, 0);
        var fg = new int[Shape.NodeCount];
        var cg = new int[Shape.NodeCount * Shape.MaxArity];
        var pg = new double[Shape.NodeCount];
        for (var n = 0; n < Shape.NodeCount; n++)
        {
            fg[n] = rng.NextInt(0, functions.Count);
            for (var a = 0; a < Shape.MaxArity; a++)
            {
                cg[n * Shape.MaxArity + a] = rng.NextInt(0, Shape.InputCount + n);
            }

            pg[n] = rng.NextDouble() * 4 - 2;
        }

        var og = new[]
        {
            rng.NextInt(0, Shape.InputCount + Shape.NodeCount),
            rng.NextInt(0, Shape.InputCount + Shape.NodeCount),
        };
        return Genome<double>.FromGenes(Shape, fg, cg, pg, og, DoubleValueSemantics.Instance, functions);
    }

    private static Genome<int> RandomInt(ulong seed, IFunctionSet<int> functions)
    {
        var rng = RandomStream.Create(seed, 0, 0, 0);
        var fg = new int[Shape.NodeCount];
        var cg = new int[Shape.NodeCount * Shape.MaxArity];
        var pg = new int[Shape.NodeCount];
        for (var n = 0; n < Shape.NodeCount; n++)
        {
            fg[n] = rng.NextInt(0, functions.Count);
            for (var a = 0; a < Shape.MaxArity; a++)
            {
                cg[n * Shape.MaxArity + a] = rng.NextInt(0, Shape.InputCount + n);
            }

            pg[n] = rng.NextInt(-5, 6);
        }

        var og = new[]
        {
            rng.NextInt(0, Shape.InputCount + Shape.NodeCount),
            rng.NextInt(0, Shape.InputCount + Shape.NodeCount),
        };
        return Genome<int>.FromGenes(Shape, fg, cg, pg, og, Int32ValueSemantics.Instance, functions);
    }

    [Fact]
    public void Cuda_identity_is_stable_across_fresh_contexts_and_selects_explicitly()
    {
        IlgpuBackendSelection first;
        try
        {
            first = IlgpuBackendSelector.SelectFor(IlgpuBackendPolicy.CudaRequired);
        }
        catch (InvalidOperationException)
        {
            // Sem CUDA válido nesta máquina: a limitação é registrada em evidence; não é aprovação CUDA.
            return;
        }

        Assert.Equal(IlgpuBackendKind.CudaDevice, first.Kind);
        Assert.False(string.IsNullOrWhiteSpace(first.CudaPciId));
        Assert.NotNull(first.CudaCatalog);
        Assert.Contains(first.CudaCatalog!, entry => entry.StartsWith("validado ", StringComparison.Ordinal));

        // Segundo contexto criado do zero: a identidade física deve ser revalidada e permanecer a mesma.
        var second = IlgpuBackendSelector.SelectCudaByPciId(first.CudaPciId!);
        Assert.Equal(IlgpuBackendKind.CudaDevice, second.Kind);
        Assert.Equal(first.CudaPciId, second.CudaPciId);
        Assert.Equal(first.DeviceName, second.DeviceName);

        var missing = Assert.Throws<InvalidOperationException>(() => IlgpuBackendSelector.SelectCudaByPciId("FFFF:FF:FF"));
        Assert.Contains("ausente ou trocado", missing.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void Cuda_required_matches_reference_for_double_and_int()
    {
        IlgpuBackendSelection selection;
        try
        {
            selection = IlgpuBackendSelector.SelectFor(IlgpuBackendPolicy.CudaRequired);
        }
        catch (InvalidOperationException)
        {
            // Sem CUDA válido nesta máquina: a limitação é registrada em evidence; não é aprovação CUDA.
            return;
        }

        Assert.Equal(IlgpuBackendKind.CudaDevice, selection.Kind);
        Assert.Equal("ILGPU/CUDA", selection.BackendName);

        var doubleFunctions = StandardDoubleFunctionSet.Instance;
        var doubleReference = new ReferenceGenomeEvaluator<double>(doubleFunctions);
        using (var cuda = new IlgpuGenomeEvaluator<double>(doubleFunctions, IlgpuBackendPolicy.CudaRequired))
        {
            var inputs = new[] { 0.5, -1.25 };
            for (var seed = 0UL; seed < 10; seed++)
            {
                var genome = RandomDouble(seed, doubleFunctions);
                var expected = new double[2];
                var actual = new double[2];
                doubleReference.Evaluate(genome, inputs, expected);
                cuda.Evaluate(genome, inputs, actual);
                Assert.True(cuda.LastEvaluationUsedDevice, $"seed={seed} não executou no dispositivo: {cuda.LastFallbackReason}");
                Assert.Equal("ILGPU/CUDA", cuda.Backend);
                for (var i = 0; i < expected.Length; i++)
                {
                    var tolerance = 1e-12 * Math.Max(1, Math.Abs(expected[i]));
                    Assert.True(Math.Abs(expected[i] - actual[i]) <= tolerance, $"seed={seed} out={i} esperado={expected[i]} obtido={actual[i]}");
                }
            }
        }

        var intFunctions = StandardInt32FunctionSet.Instance;
        var intReference = new ReferenceGenomeEvaluator<int>(intFunctions);
        using var cudaInt = new IlgpuGenomeEvaluator<int>(intFunctions, IlgpuBackendPolicy.CudaRequired);
        var intInputs = new[] { 3, -7 };
        for (var seed = 0UL; seed < 10; seed++)
        {
            var genome = RandomInt(seed, intFunctions);
            var expected = new int[2];
            var actual = new int[2];
            intReference.Evaluate(genome, intInputs, expected);
            cudaInt.Evaluate(genome, intInputs, actual);
            Assert.True(cudaInt.LastEvaluationUsedDevice, $"seed={seed} não executou no dispositivo: {cudaInt.LastFallbackReason}");
            Assert.Equal(expected, actual);
        }
    }
}
