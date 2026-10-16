namespace Evo.Core;

/// <summary>
/// Vista somente leitura da população e do fitness usada por um seletor de pais.
/// Os índices correspondem aos da população recebida pelo engine.
/// </summary>
public sealed class ParentSelectionContext<T>
{
    /// <summary>Cria o contexto validando que população e fitness têm o mesmo tamanho e não estão vazios.</summary>
    public ParentSelectionContext(
        IReadOnlyList<Genome<T>> population,
        IReadOnlyList<FitnessResult> fitness,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(population);
        ArgumentNullException.ThrowIfNull(fitness);
        if (population.Count == 0)
        {
            throw new ArgumentException("Population must not be empty.", nameof(population));
        }

        if (population.Count != fitness.Count)
        {
            throw new ArgumentException("Population and fitness must have the same length.", nameof(fitness));
        }

        ValidateGenomes(population);
        Population = population;
        Fitness = fitness;
        CancellationToken = cancellationToken;
    }

    /// <summary>Token de cancelamento da operação que criou o contexto.</summary>
    public CancellationToken CancellationToken { get; }

    private static void ValidateGenomes(IReadOnlyList<Genome<T>> population)
    {
        var first = population[0] ?? throw new ArgumentException("Population must not contain null genomes.", nameof(population));
        for (var i = 1; i < population.Count; i++)
        {
            var genome = population[i] ?? throw new ArgumentException("Population must not contain null genomes.", nameof(population));
            if (genome.Shape != first.Shape)
            {
                throw new ArgumentException($"Genome at index {i} has a different shape from genome 0.", nameof(population));
            }
        }
    }

    /// <summary>Indivíduos da geração atual, indexados de 0 a Count-1.</summary>
    public IReadOnlyList<Genome<T>> Population { get; }

    /// <summary>Fitness de cada indivíduo, no mesmo índice de <see cref="Population"/>.</summary>
    public IReadOnlyList<FitnessResult> Fitness { get; }

    /// <summary>Tamanho da população.</summary>
    public int Count => Population.Count;
}
