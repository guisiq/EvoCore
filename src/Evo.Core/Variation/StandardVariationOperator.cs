namespace Evo.Core;

/// <summary>
/// Variação padrão do engine: seleciona pai A com o seletor, com <c>CrossoverRate</c> seleciona pai B
/// e aplica crossover uniforme, depois mutação. Mesma sequência RNG do caminho legado.
/// </summary>
public sealed class StandardVariationOperator<T> : IVariationOperator<T>
{
    private readonly EvolutionOperators<T> _operators;
    private readonly IParentSelector<T>? _selector;

    /// <summary>
    /// Cria a variação padrão. Com <paramref name="selector"/> nulo, usa torneio com o
    /// <c>TournamentSize</c> de cada <see cref="VariationContext{T}"/>, como o engine legado.
    /// </summary>
    public StandardVariationOperator(IGeneticProblem<T> problem, IParentSelector<T>? selector = null)
    {
        _operators = new EvolutionOperators<T>(problem);
        _selector = selector;
    }

    /// <inheritdoc />
    public Genome<T> CreateOffspring(VariationContext<T> context, ref RandomStream random)
    {
        ArgumentNullException.ThrowIfNull(context);
        var selector = _selector ?? new TournamentParentSelector<T>(context.Options.TournamentSize);
        return _operators.BreedWith(selector, context.Population, context.Fitness, context.Options, ref random, context.CancellationToken);
    }
}
