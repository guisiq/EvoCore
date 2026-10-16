using Evo.Core;

namespace Evo.ExternalConsumer;

/// <summary>
/// Seletor próprio de consumidor externo: sorteia dois candidatos e mantém o de maior fitness válido,
/// com empate para o menor índice. Usa apenas a API pública de <c>Evo.Core</c>.
/// </summary>
public sealed class ConsumerParentSelector<T> : IParentSelector<T>
{
    /// <inheritdoc />
    public int SelectParentIndex(ParentSelectionContext<T> context, ref RandomStream random)
    {
        ArgumentNullException.ThrowIfNull(context);
        var first = random.NextInt(0, context.Count);
        var second = random.NextInt(0, context.Count);
        var cmp = EvolutionOperators<T>.CompareFitness(context.Fitness[second], context.Fitness[first]);
        if (cmp > 0 || (cmp == 0 && second < first))
        {
            return second;
        }

        return first;
    }
}
