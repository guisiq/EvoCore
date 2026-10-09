namespace Evo.Core;

/// <summary>
/// Convenience overloads for consumers that work with arrays or read-only lists.
/// The span-based <see cref="IGenomeEvaluator{T}.Evaluate"/> remains the
/// zero-allocation hot path; these helpers allocate the caller's buffer only.
/// </summary>
public static class GenomeEvaluatorExtensions
{
    /// <summary>Evaluates a genome and returns a freshly allocated output array.</summary>
    public static T[] Evaluate<T>(
        this IGenomeEvaluator<T> evaluator,
        Genome<T> genome,
        T[] inputs)
    {
        ArgumentNullException.ThrowIfNull(evaluator);
        ArgumentNullException.ThrowIfNull(genome);
        ArgumentNullException.ThrowIfNull(inputs);

        var outputs = new T[genome.Shape.OutputCount];
        evaluator.Evaluate(genome, inputs, outputs);
        return outputs;
    }

    /// <summary>Evaluates a genome from a read-only list of inputs.</summary>
    public static T[] Evaluate<T>(
        this IGenomeEvaluator<T> evaluator,
        Genome<T> genome,
        IReadOnlyList<T> inputs)
    {
        ArgumentNullException.ThrowIfNull(evaluator);
        ArgumentNullException.ThrowIfNull(genome);
        ArgumentNullException.ThrowIfNull(inputs);

        var buffer = new T[inputs.Count];
        for (var i = 0; i < inputs.Count; i++)
        {
            buffer[i] = inputs[i];
        }

        return evaluator.Evaluate(genome, buffer);
    }

    /// <summary>Evaluates a genome into a caller-provided output buffer.</summary>
    public static void Evaluate<T>(
        this IGenomeEvaluator<T> evaluator,
        Genome<T> genome,
        T[] inputs,
        T[] outputs)
    {
        ArgumentNullException.ThrowIfNull(evaluator);
        ArgumentNullException.ThrowIfNull(genome);
        ArgumentNullException.ThrowIfNull(inputs);
        ArgumentNullException.ThrowIfNull(outputs);

        evaluator.Evaluate(genome, inputs, outputs);
    }
}
