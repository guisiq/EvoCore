namespace Evo.Core;

/// <summary>
/// A problem definition: shape, functions, value semantics, parameter strategy
/// and a factory of per-worker fitness evaluators.
/// </summary>
/// <remarks>
/// Implementations must be immutable and safe for concurrent use. A new problem
/// can be added without changing <c>Evo.Core</c>.
/// </remarks>
/// <typeparam name="T">Logical value produced by a terminal/node.</typeparam>
public interface IGeneticProblem<T>
{
    /// <summary>Stable identifier of the problem, part of the fitness cache key.</summary>
    string ProblemId { get; }

    /// <summary>Fixed genome shape of the problem.</summary>
    GenomeShape Shape { get; }

    /// <summary>Function set used by the problem.</summary>
    IFunctionSet<T> Functions { get; }

    /// <summary>Value semantics of the problem's value domain.</summary>
    IValueSemantics<T> Values { get; }

    /// <summary>Parameter gene strategy of the problem.</summary>
    IParameterGeneStrategy<T> Parameters { get; }

    /// <summary>Creates a fitness evaluator owned by a single worker.</summary>
    IFitnessEvaluator<T> CreateFitnessEvaluator();
}
