using Evo.Core;

namespace Evo.ExternalConsumer;

/// <summary>
/// Operador de variação próprio de consumidor externo. Gera o descendente com o seletor do consumidor
/// e a rotina de crossover/mutação de <c>EvolutionOperators{T}</c>, sem depender de tipos internos.
/// </summary>
public sealed class ConsumerVariationOperator<T> : IVariationOperator<T>
{
    private readonly EvolutionOperators<T> _operators;
    private readonly IParentSelector<T> _selector;

    /// <summary>Cria o operador com o problema genético e o seletor de pais do consumidor.</summary>
    public ConsumerVariationOperator(IGeneticProblem<T> problem, IParentSelector<T> selector)
    {
        ArgumentNullException.ThrowIfNull(problem);
        ArgumentNullException.ThrowIfNull(selector);
        _operators = new EvolutionOperators<T>(problem);
        _selector = selector;
    }

    /// <inheritdoc />
    public Genome<T> CreateOffspring(VariationContext<T> context, ref RandomStream random)
    {
        ArgumentNullException.ThrowIfNull(context);
        return _operators.BreedWith(_selector, context.Population, context.Fitness, context.Options, ref random);
    }
}
