namespace Evo.Core;

/// <summary>
/// Entrada de variação: população e fitness da geração atual, mais as opções da execução.
/// </summary>
public sealed class VariationContext<T>
{
    /// <summary>Cria o contexto validando argumentos e que população e fitness têm o mesmo tamanho.</summary>
    public VariationContext(
        IReadOnlyList<Genome<T>> population,
        IReadOnlyList<FitnessResult> fitness,
        EvolutionOptions options,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(options);
        Population = new ParentSelectionContext<T>(population, fitness, cancellationToken).Population;
        Fitness = fitness;
        Options = options;
        CancellationToken = cancellationToken;
    }

    /// <summary>Token de cancelamento aplicado à criação dos descendentes.</summary>
    public CancellationToken CancellationToken { get; }

    /// <summary>Indivíduos da geração anterior.</summary>
    public IReadOnlyList<Genome<T>> Population { get; }

    /// <summary>Fitness de cada indivíduo da geração anterior.</summary>
    public IReadOnlyList<FitnessResult> Fitness { get; }

    /// <summary>Opções da execução (taxas de crossover e tamanho de torneio padrão).</summary>
    public EvolutionOptions Options { get; }
}
