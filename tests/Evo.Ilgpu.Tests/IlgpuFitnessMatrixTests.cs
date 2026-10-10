using System.Linq;
using System.Threading;
using Evo.Core;
using Evo.Core.Compiled;
using Evo.Core.Functions;
using Evo.Core.Values;
using Evo.Ilgpu;
using Xunit;

namespace Evo.Ilgpu.Tests;

/// <summary>
/// Matriz de <see cref="FitnessResult"/> (EVO-102, R1-4): compara IsValid e Value entre
/// ReferenceGenomeEvaluator, CompiledGenomeEvaluator e IlgpuGenomeEvaluator em cada linha de
/// <see cref="IlgpuDeviceMatrixTests.Devices"/>, nos quatro tipos (double, int, double[], int[]),
/// usando a mesma implementação de <see cref="IFitnessEvaluator{T}"/> para todos os avaliadores.
/// Cobre fitness válido finito e invalidação por NaN/+Infinity/-Infinity escalar e por elemento de
/// array, sem alterar a política normativa de fitness.
/// </summary>
public sealed class IlgpuFitnessMatrixTests
{
    [Theory]
    [MemberData(nameof(IlgpuDeviceMatrixTests.Devices), MemberType = typeof(IlgpuDeviceMatrixTests))]
    public void FitnessResult_matches_across_reference_compiled_and_device(string label, IlgpuBackendSelection selection)
    {
        AssertDoubleFitnessMatrix(label, selection);
        AssertIntFitnessMatrix(label, selection);
        AssertDoubleArrayFitnessMatrix(label, selection);
        AssertIntArrayFitnessMatrix(label, selection);
    }

    private static void AssertDoubleFitnessMatrix(string label, IlgpuBackendSelection selection)
    {
        var functions = StandardDoubleFunctionSet.Instance;
        var addId = IlgpuTestHelpers.FindFunctionId(functions, "add");
        var shape = new GenomeShape(InputCount: 2, NodeCount: 1, OutputCount: 1, MaxArity: 2, LevelsBack: 1);
        var genome = Genome<double>.FromGenes(shape, [addId], [0, 1], [0.0], [2], DoubleValueSemantics.Instance, functions);
        FitnessResult Score(double[] output) =>
            double.IsNaN(output[0]) || double.IsInfinity(output[0]) ? FitnessResult.Invalid() : FitnessResult.Valid(output[0]);

        foreach (var (name, inputs, expectedValid) in new (string, double[], bool)[]
        {
            ("finito", [3.0, -0.5], true),
            ("NaN", [double.NaN, 1.0], false),
            ("+Infinity", [double.PositiveInfinity, 1.0], false),
            ("-Infinity", [double.NegativeInfinity, 1.0], false),
        })
        {
            var tag = $"{label} double {name}";
            var reference = new ScoredFitness<double>(inputs, Score).Evaluate(genome, new ReferenceGenomeEvaluator<double>(functions), CancellationToken.None);
            Assert.True(expectedValid == reference.IsValid, $"{tag}: referência IsValid");

            var compiled = new ScoredFitness<double>(inputs, Score).Evaluate(genome, new CompiledGenomeEvaluator<double>(functions, DoubleValueSemantics.Instance), CancellationToken.None);
            Assert.True(reference.IsValid == compiled.IsValid, $"{tag}: compiled IsValid");

            using var evaluator = new IlgpuGenomeEvaluator<double>(functions, null, selection, null, null, null);
            var device = new ScoredFitness<double>(inputs, Score).Evaluate(genome, evaluator, CancellationToken.None);
            Assert.True(reference.IsValid == device.IsValid, $"{tag}: device IsValid");

            if (reference.IsValid)
            {
                IlgpuTestHelpers.AssertDoubleClose(reference.Value, compiled.Value, $"{tag} compiled valor");
                IlgpuTestHelpers.AssertDoubleClose(reference.Value, device.Value, $"{tag} device valor");
            }
            else
            {
                Assert.Equal(reference.Value, compiled.Value);
                Assert.Equal(reference.Value, device.Value);
            }

            IlgpuTestHelpers.AssertRealKernelExecution(evaluator, genome, functions);
            Assert.True(selection.BackendName == evaluator.Backend, $"{tag}: backend");
            Assert.True(selection.DiagnosticDeviceName == evaluator.DeviceName, $"{tag}: dispositivo");
        }
    }

    private static void AssertIntFitnessMatrix(string label, IlgpuBackendSelection selection)
    {
        var functions = StandardInt32FunctionSet.Instance;
        var addId = IlgpuTestHelpers.FindFunctionId(functions, "add");
        var shape = new GenomeShape(InputCount: 2, NodeCount: 1, OutputCount: 1, MaxArity: 2, LevelsBack: 1);
        var genome = Genome<int>.FromGenes(shape, [addId], [0, 1], [0], [2], Int32ValueSemantics.Instance, functions);
        static FitnessResult Score(int[] output) => FitnessResult.Valid(output[0]);

        foreach (var (name, inputs) in new (string, int[])[]
        {
            ("finito", [3, -5]),
            ("overflow unchecked", [int.MaxValue, 1]),
        })
        {
            var tag = $"{label} int {name}";
            var reference = new ScoredFitness<int>(inputs, Score).Evaluate(genome, new ReferenceGenomeEvaluator<int>(functions), CancellationToken.None);
            Assert.True(reference.IsValid, $"{tag}: referência IsValid");

            var compiled = new ScoredFitness<int>(inputs, Score).Evaluate(genome, new CompiledGenomeEvaluator<int>(functions, Int32ValueSemantics.Instance), CancellationToken.None);
            Assert.Equal(reference.Value, compiled.Value);

            using var evaluator = new IlgpuGenomeEvaluator<int>(functions, null, selection, null, null, null);
            var device = new ScoredFitness<int>(inputs, Score).Evaluate(genome, evaluator, CancellationToken.None);
            Assert.Equal(reference.Value, device.Value);

            IlgpuTestHelpers.AssertRealKernelExecution(evaluator, genome, functions);
            Assert.True(selection.BackendName == evaluator.Backend, $"{tag}: backend");
            Assert.True(selection.DiagnosticDeviceName == evaluator.DeviceName, $"{tag}: dispositivo");
        }
    }

    private static void AssertDoubleArrayFitnessMatrix(string label, IlgpuBackendSelection selection)
    {
        var functions = StandardDoubleArrayFunctionSet.Instance;
        var addId = IlgpuTestHelpers.FindFunctionId(functions, "add");
        var shape = new GenomeShape(InputCount: 2, NodeCount: 1, OutputCount: 1, MaxArity: 2, LevelsBack: 1);
        var genome = Genome<double[]>.FromGenes(shape, [addId], [0, 1], [Array.Empty<double>()], [2], DoubleArrayValueSemantics.Instance, functions);
        static FitnessResult Score(double[][] output)
        {
            var sum = 0.0;
            foreach (var v in output[0])
            {
                if (double.IsNaN(v) || double.IsInfinity(v))
                {
                    return FitnessResult.Invalid();
                }

                sum += v;
            }

            return FitnessResult.Valid(sum);
        }

        foreach (var (name, inputs, expectedValid) in new (string, double[][], bool)[]
        {
            ("finito", [[1.0, 2.0], [3.0, 4.0]], true),
            ("NaN elemento", [[double.NaN, 2.0], [3.0, 4.0]], false),
            ("+Infinity elemento", [[double.PositiveInfinity, 2.0], [3.0, 4.0]], false),
            ("-Infinity elemento", [[double.NegativeInfinity, 2.0], [3.0, 4.0]], false),
        })
        {
            var tag = $"{label} double[] {name}";
            var reference = new ScoredFitness<double[]>(inputs, Score).Evaluate(genome, new ReferenceGenomeEvaluator<double[]>(functions), CancellationToken.None);
            Assert.True(expectedValid == reference.IsValid, $"{tag}: referência IsValid");

            var compiled = new ScoredFitness<double[]>(inputs, Score).Evaluate(genome, new CompiledGenomeEvaluator<double[]>(functions, DoubleArrayValueSemantics.Instance), CancellationToken.None);
            Assert.True(reference.IsValid == compiled.IsValid, $"{tag}: compiled IsValid");

            using var evaluator = new IlgpuGenomeEvaluator<double[]>(functions, null, selection, null, null, null);
            var device = new ScoredFitness<double[]>(inputs, Score).Evaluate(genome, evaluator, CancellationToken.None);
            Assert.True(reference.IsValid == device.IsValid, $"{tag}: device IsValid");

            if (reference.IsValid)
            {
                IlgpuTestHelpers.AssertDoubleClose(reference.Value, compiled.Value, $"{tag} compiled valor");
                IlgpuTestHelpers.AssertDoubleClose(reference.Value, device.Value, $"{tag} device valor");
            }
            else
            {
                Assert.Equal(reference.Value, compiled.Value);
                Assert.Equal(reference.Value, device.Value);
            }

            IlgpuTestHelpers.AssertRealKernelExecution(evaluator, genome, functions);
            Assert.True(selection.BackendName == evaluator.Backend, $"{tag}: backend");
            Assert.True(selection.DiagnosticDeviceName == evaluator.DeviceName, $"{tag}: dispositivo");
        }
    }

    private static void AssertIntArrayFitnessMatrix(string label, IlgpuBackendSelection selection)
    {
        var functions = StandardInt32ArrayFunctionSet.Instance;
        var addId = IlgpuTestHelpers.FindFunctionId(functions, "add");
        var shape = new GenomeShape(InputCount: 2, NodeCount: 1, OutputCount: 1, MaxArity: 2, LevelsBack: 1);
        var genome = Genome<int[]>.FromGenes(shape, [addId], [0, 1], [Array.Empty<int>()], [2], Int32ArrayValueSemantics.Instance, functions);
        static FitnessResult Score(int[][] output) => FitnessResult.Valid(output[0].Sum());

        var inputs = new[] { new[] { 1, 2 }, new[] { 3, 4 } };
        var tag = $"{label} int[] finito";
        var reference = new ScoredFitness<int[]>(inputs, Score).Evaluate(genome, new ReferenceGenomeEvaluator<int[]>(functions), CancellationToken.None);
        Assert.True(reference.IsValid, $"{tag}: referência IsValid");

        var compiled = new ScoredFitness<int[]>(inputs, Score).Evaluate(genome, new CompiledGenomeEvaluator<int[]>(functions, Int32ArrayValueSemantics.Instance), CancellationToken.None);
        Assert.Equal(reference.Value, compiled.Value);

        using var evaluator = new IlgpuGenomeEvaluator<int[]>(functions, null, selection, null, null, null);
        var device = new ScoredFitness<int[]>(inputs, Score).Evaluate(genome, evaluator, CancellationToken.None);
        Assert.Equal(reference.Value, device.Value);

        IlgpuTestHelpers.AssertRealKernelExecution(evaluator, genome, functions);
        Assert.True(selection.BackendName == evaluator.Backend, $"{tag}: backend");
        Assert.True(selection.DiagnosticDeviceName == evaluator.DeviceName, $"{tag}: dispositivo");
    }

    /// <summary>Avaliador de fitness genérico que delega o cálculo de <see cref="FitnessResult"/> a um
    /// delegado, permitindo reutilizar a mesma política de pontuação entre referência, compiled e ILGPU.</summary>
    private sealed class ScoredFitness<T>(T[] inputs, Func<T[], FitnessResult> score) : IFitnessEvaluator<T>
    {
        public FitnessResult Evaluate(Genome<T> genome, IGenomeEvaluator<T> evaluator, CancellationToken cancellationToken)
        {
            var output = new T[genome.Shape.OutputCount];
            evaluator.Evaluate(genome, inputs, output);
            return score(output);
        }
    }
}
