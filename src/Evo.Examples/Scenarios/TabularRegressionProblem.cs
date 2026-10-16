using Evo.Core;
using Evo.Core.Functions;
using Evo.Core.Values;

namespace Evo.Examples.Scenarios;

/// <summary>
/// Problema genérico de regressão tabular: fitness <c>-MSE</c> sobre linhas de um dataset local
/// (features -&gt; alvo escalar), reutilizando o conjunto padrão de funções <c>double</c>.
/// </summary>
public sealed class TabularRegressionProblem : IGeneticProblem<double>
{
    private readonly double[][] _features;
    private readonly double[] _target;

    public TabularRegressionProblem(string problemId, double[][] features, double[] target, int nodeCount, IFunctionSet<double>? functions = null)
    {
        if (features.Length == 0 || features.Length != target.Length)
        {
            throw new ArgumentException("features e target devem ter o mesmo comprimento não vazio.");
        }

        _features = features;
        _target = target;
        ProblemId = problemId;
        Functions = functions ?? StandardDoubleFunctionSet.Instance;
        Shape = new GenomeShape(InputCount: features[0].Length, NodeCount: nodeCount, OutputCount: 1, MaxArity: 2, LevelsBack: nodeCount);
    }

    public string ProblemId { get; }

    public GenomeShape Shape { get; }

    public IFunctionSet<double> Functions { get; }

    public IValueSemantics<double> Values => DoubleValueSemantics.Instance;

    public IParameterGeneStrategy<double> Parameters => DoubleParameterStrategy.Instance;

    public IFitnessEvaluator<double> CreateFitnessEvaluator() => new Evaluator(_features, _target);

    /// <summary>MSE de um genoma avaliado sobre todas as linhas do problema (para relatório fora do fitness).</summary>
    public double Mse(Genome<double> genome, IGenomeEvaluator<double> evaluator)
    {
        var output = new double[1];
        var sum = 0.0;
        for (var i = 0; i < _features.Length; i++)
        {
            evaluator.Evaluate(genome, _features[i], output);
            var diff = output[0] - _target[i];
            sum += diff * diff;
        }

        return sum / _features.Length;
    }

    private sealed class Evaluator : IFitnessEvaluator<double>
    {
        private readonly double[][] _features;
        private readonly double[] _target;
        private readonly double[] _output = new double[1];

        public Evaluator(double[][] features, double[] target)
        {
            _features = features;
            _target = target;
        }

        public FitnessResult Evaluate(Genome<double> genome, IGenomeEvaluator<double> evaluator, CancellationToken cancellationToken)
        {
            var sum = 0.0;
            for (var i = 0; i < _features.Length; i++)
            {
                cancellationToken.ThrowIfCancellationRequested();
                evaluator.Evaluate(genome, _features[i], _output);
                var diff = _output[0] - _target[i];
                if (double.IsNaN(diff) || double.IsInfinity(diff))
                {
                    return FitnessResult.Invalid();
                }

                sum += diff * diff;
            }

            return FitnessResult.Valid(-sum / _features.Length);
        }
    }
}
