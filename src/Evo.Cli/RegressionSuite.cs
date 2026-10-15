using Evo.Core;
using Evo.Core.Functions;
using Evo.Core.Values;

namespace Evo.Cli;

/// <summary>Problema determinístico da suíte exploratória de regressão simbólica.</summary>
public sealed record RegressionSuiteProblem(
    string Id,
    string Expression,
    double InputMinimum,
    double InputMaximum,
    Func<double, double> Target)
{
    public GenomeShape Shape { get; } = new(InputCount: 1, NodeCount: 30, OutputCount: 1, MaxArity: 2, LevelsBack: 30);
    public IFunctionSet<double> Functions => StandardDoubleFunctionSet.Instance;
    public IValueSemantics<double> Values => DoubleValueSemantics.Instance;
    public IParameterGeneStrategy<double> Parameters => DoubleParameterStrategy.Instance;
    public RegressionSuiteDataset Dataset { get; } = RegressionSuiteDataset.Create(InputMinimum, InputMaximum);

    public IGeneticProblem<double> CreateGeneticProblem()
        => new SuiteGeneticProblem(this);

    private sealed class SuiteGeneticProblem(RegressionSuiteProblem definition) : IGeneticProblem<double>
    {
        public string ProblemId => definition.Id;
        public GenomeShape Shape => definition.Shape;
        public IFunctionSet<double> Functions => definition.Functions;
        public IValueSemantics<double> Values => definition.Values;
        public IParameterGeneStrategy<double> Parameters => definition.Parameters;
        public IFitnessEvaluator<double> CreateFitnessEvaluator() => new Evaluator(definition);

        private sealed class Evaluator(RegressionSuiteProblem definition) : IFitnessEvaluator<double>
        {
            private readonly double[] _input = new double[1];
            private readonly double[] _output = new double[1];

            public FitnessResult Evaluate(
                Genome<double> genome,
                IGenomeEvaluator<double> evaluator,
                CancellationToken cancellationToken)
            {
                var sum = 0.0;
                foreach (var value in definition.Dataset.TrainingInputs)
                {
                    cancellationToken.ThrowIfCancellationRequested();
                    _input[0] = value;
                    evaluator.Evaluate(genome, _input, _output);
                    var difference = _output[0] - definition.Target(value);
                    if (!double.IsFinite(difference))
                    {
                        return FitnessResult.Invalid();
                    }

                    sum += difference * difference;
                }

                return FitnessResult.Valid(-sum / definition.Dataset.TrainingInputs.Length);
            }
        }
    }
}

public sealed record RegressionSuiteDataset(
    double[] TrainingInputs,
    double[] ValidationInputs)
{
    public const int PointCount = 201;

    public static RegressionSuiteDataset Create(double minimum, double maximum)
    {
        if (!double.IsFinite(minimum) || !double.IsFinite(maximum) || minimum >= maximum)
        {
            throw new ArgumentOutOfRangeException(nameof(minimum), "The input interval must be finite and increasing.");
        }

        var inputs = new double[PointCount];
        for (var index = 0; index < inputs.Length; index++)
        {
            inputs[index] = minimum + (maximum - minimum) * index / (PointCount - 1);
        }

        return new RegressionSuiteDataset(
            inputs.Where((_, index) => index % 2 == 0).ToArray(),
            inputs.Where((_, index) => index % 2 != 0).ToArray());
    }
}

/// <summary>
/// Subconjunto univariado e representável da suíte Nguyen descrita por McDermott et al.
/// Os domínios seguem a referência; a amostragem foi tornada densa e determinística.
/// </summary>
public static class RegressionSuiteCatalog
{
    public static IReadOnlyList<RegressionSuiteProblem> All { get; } =
    [
        new("nguyen-1", "x^3 + x^2 + x", -1.0, 1.0, x => PowSum(x, 3)),
        new("nguyen-2", "x^4 + x^3 + x^2 + x", -1.0, 1.0, x => PowSum(x, 4)),
        new("nguyen-3", "x^5 + x^4 + x^3 + x^2 + x", -1.0, 1.0, x => PowSum(x, 5)),
        new("nguyen-4", "x^6 + x^5 + x^4 + x^3 + x^2 + x", -1.0, 1.0, x => PowSum(x, 6)),
        new("nguyen-7", "log(x + 1) + log(x^2 + 1)", 0.0, 2.0, x => Math.Log(x + 1.0) + Math.Log((x * x) + 1.0)),
        new("nguyen-8", "sqrt(x)", 0.0, 4.0, Math.Sqrt),
    ];

    private static double PowSum(double x, int highestPower)
    {
        var sum = x;
        var power = x;
        for (var exponent = 2; exponent <= highestPower; exponent++)
        {
            power *= x;
            sum += power;
        }

        return sum;
    }
}

public readonly record struct DagReuseMetrics(
    int ActiveNodes,
    int ReusedActiveNodes,
    int ReuseEdges,
    int SemanticReferences,
    double ReusedActiveNodeRatio,
    double ReuseEdgeRatio)
{
    public static DagReuseMetrics Measure(
        Genome<double> genome,
        IFunctionSet<double> functions)
    {
        var active = genome.Analyze(functions);
        var fanOut = new int[genome.Shape.NodeCount];
        var references = 0;

        foreach (var node in active.ActiveNodes)
        {
            var arity = functions.Get(genome.FunctionGenes[node]).Arity;
            for (var slot = 0; slot < arity; slot++)
            {
                var address = genome.GetConnection(node, slot) - genome.Shape.InputCount;
                if (address >= 0 && address < fanOut.Length)
                {
                    fanOut[address]++;
                    references++;
                }
            }
        }

        foreach (var address in genome.OutputGenes)
        {
            var node = address - genome.Shape.InputCount;
            if (node >= 0 && node < fanOut.Length)
            {
                fanOut[node]++;
                references++;
            }
        }

        var reused = active.ActiveNodes.Count(node => fanOut[node] > 1);
        var reuseEdges = fanOut.Where(count => count > 1).Sum(count => count - 1);
        var reusedNodeRatio = active.ActiveNodeCount == 0 ? 0.0 : (double)reused / active.ActiveNodeCount;
        var reuseEdgeRatio = references == 0 ? 0.0 : (double)reuseEdges / references;
        return new DagReuseMetrics(
            active.ActiveNodeCount,
            reused,
            reuseEdges,
            references,
            reusedNodeRatio,
            reuseEdgeRatio);
    }
}

public readonly record struct DagReuseAverages(
    double ActiveNodes,
    double ReusedActiveNodes,
    double ReuseEdges,
    double SemanticReferences,
    double ReusedActiveNodeRatio,
    double ReuseEdgeRatio)
{
    public static DagReuseAverages From(IReadOnlyList<DagReuseMetrics> metrics)
    {
        if (metrics.Count == 0)
        {
            return new DagReuseAverages();
        }

        return new DagReuseAverages(
            metrics.Average(item => item.ActiveNodes),
            metrics.Average(item => item.ReusedActiveNodes),
            metrics.Average(item => item.ReuseEdges),
            metrics.Average(item => item.SemanticReferences),
            metrics.Average(item => item.ReusedActiveNodeRatio),
            metrics.Average(item => item.ReuseEdgeRatio));
    }
}
