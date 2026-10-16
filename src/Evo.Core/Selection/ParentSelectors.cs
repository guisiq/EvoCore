namespace Evo.Core;

/// <summary>
/// Torneio padrão do engine: sorteia <c>Size</c> candidatos com reposição e mantém o de melhor fitness.
/// Empates vencem o menor índice.
/// </summary>
public sealed class TournamentParentSelector<T> : IParentSelector<T>
{
    /// <summary>Cria o seletor com tamanho de torneio maior ou igual a 1.</summary>
    public TournamentParentSelector(int size)
    {
        if (size < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(size), "Tournament size must be >= 1.");
        }

        Size = size;
    }

    /// <summary>Número de candidatos sorteados por torneio.</summary>
    public int Size { get; }

    /// <inheritdoc />
    public int SelectParentIndex(ParentSelectionContext<T> context, ref RandomStream random)
    {
        ArgumentNullException.ThrowIfNull(context);
        var fitness = context.Fitness;
        var best = random.NextInt(0, context.Count);
        for (var i = 1; i < Size; i++)
        {
            var candidate = random.NextInt(0, context.Count);
            var cmp = EvolutionOperators<T>.CompareFitness(fitness[candidate], fitness[best]);
            if (cmp > 0 || (cmp == 0 && candidate < best))
            {
                best = candidate;
            }
        }

        return best;
    }
}

/// <summary>Seleção uniforme: um índice sorteado igualmente entre todos os indivíduos.</summary>
public sealed class UniformParentSelector<T> : IParentSelector<T>
{
    /// <inheritdoc />
    public int SelectParentIndex(ParentSelectionContext<T> context, ref RandomStream random)
    {
        ArgumentNullException.ThrowIfNull(context);
        return random.NextInt(0, context.Count);
    }
}

/// <summary>Seleção do melhor absoluto por fitness; empates vencem o menor índice. Não consome RNG.</summary>
public sealed class BestParentSelector<T> : IParentSelector<T>
{
    /// <inheritdoc />
    public int SelectParentIndex(ParentSelectionContext<T> context, ref RandomStream random)
    {
        ArgumentNullException.ThrowIfNull(context);
        var fitness = context.Fitness;
        var best = 0;
        for (var i = 1; i < context.Count; i++)
        {
            if (EvolutionOperators<T>.CompareFitness(fitness[i], fitness[best]) > 0)
            {
                best = i;
            }
        }

        return best;
    }
}
