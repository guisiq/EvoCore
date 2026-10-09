using System.Threading;
using Evo.Core;
using Evo.Core.Functions;
using Evo.Core.Values;

namespace Evo.Cli;

/// <summary>
/// Dataset fixo de regressão simbólica: target(x) = x^4 + x^3 + x^2 + x em 201 pontos uniformes em [-1, 1].
/// Índices pares formam o treino e os ímpares a validação. A seed afeta somente a evolução.
/// </summary>
public static class SymbolicRegressionDataset
{
    /// <summary>Número fixo de pontos.</summary>
    public const int PointCount = 201;

    /// <summary>Entradas <c>x_i = -1 + 2*i/200</c>.</summary>
    public static readonly double[] Inputs = Build();

    /// <summary>Alvo do dataset.</summary>
    public static double Target(double x) => x * x * x * x + x * x * x + x * x + x;

    /// <summary>Índices de treino (pares).</summary>
    public static int[] TrainingIndices { get; } = Indices(0);

    /// <summary>Índices de validação (ímpares).</summary>
    public static int[] ValidationIndices { get; } = Indices(1);

    private static double[] Build()
    {
        var values = new double[PointCount];
        for (var i = 0; i < PointCount; i++)
        {
            values[i] = -1.0 + 2.0 * i / 200.0;
        }

        return values;
    }

    private static int[] Indices(int parity)
    {
        var list = new List<int>();
        for (var i = parity; i < PointCount; i += 2)
        {
            list.Add(i);
        }

        return list.ToArray();
    }
}

/// <summary>Problema <c>double</c> de regressão simbólica com fitness <c>-MSE</c> no treino.</summary>
public sealed class SymbolicRegressionProblem : IGeneticProblem<double>
{
    /// <summary>Shape fixo da demo.</summary>
    public static readonly GenomeShape DemoShape = new(InputCount: 1, NodeCount: 30, OutputCount: 1, MaxArity: 2, LevelsBack: 30);

    /// <inheritdoc />
    public string ProblemId => "symbolic-regression-quartic-v1";

    /// <inheritdoc />
    public GenomeShape Shape => DemoShape;

    /// <inheritdoc />
    public IFunctionSet<double> Functions => StandardDoubleFunctionSet.Instance;

    /// <inheritdoc />
    public IValueSemantics<double> Values => DoubleValueSemantics.Instance;

    /// <inheritdoc />
    public IParameterGeneStrategy<double> Parameters => DoubleParameterStrategy.Instance;

    /// <inheritdoc />
    public IFitnessEvaluator<double> CreateFitnessEvaluator() => new Evaluator();

    /// <summary>MSE de um genoma sobre um conjunto de índices do dataset.</summary>
    public static double Mse(Genome<double> genome, IGenomeEvaluator<double> evaluator, int[] indices)
    {
        var output = new double[1];
        var input = new double[1];
        var sum = 0.0;
        foreach (var index in indices)
        {
            input[0] = SymbolicRegressionDataset.Inputs[index];
            evaluator.Evaluate(genome, input, output);
            var diff = output[0] - SymbolicRegressionDataset.Target(input[0]);
            sum += diff * diff;
        }

        return sum / indices.Length;
    }

    private sealed class Evaluator : IFitnessEvaluator<double>
    {
        private readonly double[] _input = new double[1];
        private readonly double[] _output = new double[1];

        public FitnessResult Evaluate(Genome<double> genome, IGenomeEvaluator<double> evaluator, CancellationToken cancellationToken)
        {
            var sum = 0.0;
            foreach (var index in SymbolicRegressionDataset.TrainingIndices)
            {
                _input[0] = SymbolicRegressionDataset.Inputs[index];
                evaluator.Evaluate(genome, _input, _output);
                var diff = _output[0] - SymbolicRegressionDataset.Target(_input[0]);
                if (double.IsNaN(diff) || double.IsInfinity(diff))
                {
                    return FitnessResult.Invalid();
                }

                sum += diff * diff;
            }

            return FitnessResult.Valid(-sum / SymbolicRegressionDataset.TrainingIndices.Length);
        }
    }
}
