namespace Evo.Core;

/// <summary>Modo de composição de operadores de variação.</summary>
public enum CompositionMode
{
    /// <summary>Cada estágio recebe o descendente produzido pelo estágio anterior, na ordem declarada.</summary>
    Sequential,

    /// <summary>Cada posição de descendente é atribuída a um operador pelo método dos maiores restos sobre os pesos.</summary>
    Partitioned,
}

/// <summary>
/// Composição de operadores de variação em sequência ou particionada por pesos.
/// A composição é imutável; estado por execução pertence ao contexto recebido.
/// </summary>
public sealed class CompositeVariationOperator<T> : IVariationOperator<T>
{
    private readonly IVariationOperator<T>[] _stages;
    private readonly double[]? _weights;

    /// <summary>Cria a composição validando operadores, modo e pesos antes de qualquer geração.</summary>
    public CompositeVariationOperator(
        CompositionMode mode,
        IReadOnlyList<IVariationOperator<T>> stages,
        IReadOnlyList<double>? weights = null)
    {
        ArgumentNullException.ThrowIfNull(stages);
        if (stages.Count == 0)
        {
            throw new ArgumentException("A composição precisa de ao menos um operador.", nameof(stages));
        }

        for (var i = 0; i < stages.Count; i++)
        {
            if (stages[i] is null)
            {
                throw new ArgumentException($"Operador {i} é nulo.", nameof(stages));
            }
        }

        Mode = mode;
        _stages = stages.ToArray();

        // Rejeita ciclos e reuso da mesma instância em qualquer ponto do grafo de composição,
        // por identidade de referência, antes de qualquer geração.
        var visited = new HashSet<IVariationOperator<T>>(ReferenceEqualityComparer.Instance) { this };
        foreach (var stage in _stages)
        {
            VisitForCycles(stage, visited);
        }

        switch (mode)
        {
            case CompositionMode.Sequential:
                if (weights is not null)
                {
                    throw new ArgumentException("Composição sequencial não aceita pesos.", nameof(weights));
                }

                break;
            case CompositionMode.Partitioned:
                if (weights is null || weights.Count != _stages.Length)
                {
                    throw new ArgumentException("Composição particionada exige um peso por operador.", nameof(weights));
                }

                // Valida pesos finitos e positivos antes de qualquer geração.
                PartitionAllocator.LargestRemainder(weights, 0);
                _weights = weights.ToArray();
                break;
            default:
                throw new ArgumentOutOfRangeException(nameof(mode));
        }
    }

    /// <summary>Modo de composição.</summary>
    public CompositionMode Mode { get; }

    /// <inheritdoc />
    public Genome<T> CreateOffspring(VariationContext<T> context, ref RandomStream random) =>
        CreatePositionOffspring(context, 0, 1, ref random);

    /// <summary>
    /// Cria o descendente da posição <paramref name="position"/> de um lote de <paramref name="total"/> posições.
    /// No modo particionado, a posição é atribuída ao operador pela alocação de maiores restos do lote.
    /// No modo sequencial, todos os estágios são aplicados em ordem ao mesmo descendente.
    /// </summary>
    public Genome<T> CreatePositionOffspring(VariationContext<T> context, int position, int total, ref RandomStream random)
    {
        ArgumentNullException.ThrowIfNull(context);
        ArgumentOutOfRangeException.ThrowIfNegative(position);
        if (total <= position)
        {
            throw new ArgumentOutOfRangeException(nameof(total), "A posição deve ser menor que o total do lote.");
        }

        if (Mode == CompositionMode.Partitioned)
        {
            var branch = BranchFor(position, total);
            return _stages[branch].CreateOffspring(context, ref random);
        }

        Genome<T>? current = null;
        for (var s = 0; s < _stages.Length; s++)
        {
            var stageContext = s == 0
                ? context
                : new VariationContext<T>(
                    new[] { current! },
                    new[] { FitnessResult.Invalid() },
                    context.Options,
                    context.CancellationToken);
            current = _stages[s].CreateOffspring(stageContext, ref random);
        }

        return current!;
    }

    private static void VisitForCycles(IVariationOperator<T> op, HashSet<IVariationOperator<T>> visited)
    {
        if (!visited.Add(op))
        {
            throw new ArgumentException(
                "Composição contém ciclo ou reusa a mesma instância de operador em mais de um ponto do grafo.",
                nameof(op));
        }

        if (op is CompositeVariationOperator<T> nested)
        {
            foreach (var child in nested._stages)
            {
                VisitForCycles(child, visited);
            }
        }
    }

    /// <summary>Operador (índice na lista) responsável pela posição de um lote particionado.</summary>
    public int BranchFor(int position, int total)
    {
        if (Mode != CompositionMode.Partitioned || _weights is null)
        {
            return 0;
        }

        var allocation = PartitionAllocator.LargestRemainder(_weights, total);
        var cumulative = 0;
        for (var b = 0; b < allocation.Length; b++)
        {
            cumulative += allocation[b];
            if (position < cumulative)
            {
                return b;
            }
        }

        throw new InvalidOperationException("Alocação não cobre a posição solicitada.");
    }
}
