namespace Evo.Core;

/// <summary>
/// Capacidade opcional (v2, 05-v2-batch-evaluation.md) de avaliar vários genomas em uma chamada.
/// </summary>
/// <remarks>
/// Aditivo: <see cref="IFitnessEvaluator{T}"/> continua sendo o contrato canônico. O engine detecta
/// esta capacidade por tipo; problemas que não a implementam seguem o caminho individual.
/// <see cref="PreferredBatchSize"/> é apenas uma dica. Os resultados devem coincidir, na mesma ordem,
/// com chamadas individuais a <see cref="IFitnessEvaluator{T}.Evaluate"/>, independentemente de como
/// o engine particiona os lotes.
/// </remarks>
/// <typeparam name="T">Valor lógico produzido por terminal/nó.</typeparam>
public interface IBatchFitnessEvaluator<T> : IFitnessEvaluator<T>
{
    /// <summary>Tamanho de lote preferido (dica, mínimo efetivo 1).</summary>
    int PreferredBatchSize { get; }

    /// <summary>
    /// Avalia <paramref name="genomes"/> e escreve o resultado de cada um em <paramref name="results"/>
    /// na mesma posição. Ambos os spans devem ter o mesmo tamanho.
    /// </summary>
    void EvaluateBatch(
        ReadOnlySpan<Genome<T>> genomes,
        IGenomeEvaluator<T> evaluator,
        Span<FitnessResult> results,
        CancellationToken cancellationToken);
}
