namespace Evo.Core;

/// <summary>
/// Alocação inteira de posições de descendentes por pesos, pelo método dos maiores restos.
/// </summary>
public static class PartitionAllocator
{
    /// <summary>
    /// Distribui <paramref name="total"/> posições entre partes com pesos finitos e estritamente positivos.
    /// Cada parte recebe o piso da cota ideal; as posições restantes vão para os maiores restos, com empate
    /// pelo menor índice da parte. A soma das alocações é exatamente <paramref name="total"/>.
    /// </summary>
    public static int[] LargestRemainder(IReadOnlyList<double> weights, int total)
    {
        ArgumentNullException.ThrowIfNull(weights);
        ArgumentOutOfRangeException.ThrowIfNegative(total);
        if (weights.Count == 0)
        {
            throw new ArgumentException("A partição precisa de ao menos um peso.", nameof(weights));
        }

        var sum = 0.0;
        for (var i = 0; i < weights.Count; i++)
        {
            var w = weights[i];
            if (!double.IsFinite(w) || w <= 0)
            {
                throw new ArgumentException($"Peso {i} deve ser finito e estritamente positivo.", nameof(weights));
            }

            sum += w;
        }

        var allocation = new int[weights.Count];
        var remainders = new double[weights.Count];
        var assigned = 0;
        for (var i = 0; i < weights.Count; i++)
        {
            var quota = total * (weights[i] / sum);
            var floor = (int)Math.Floor(quota);
            allocation[i] = floor;
            remainders[i] = quota - floor;
            assigned += floor;
        }

        var order = Enumerable.Range(0, weights.Count).ToArray();
        Array.Sort(order, (x, y) =>
        {
            var cmp = remainders[y].CompareTo(remainders[x]);
            return cmp != 0 ? cmp : x.CompareTo(y);
        });

        for (var k = 0; k < total - assigned; k++)
        {
            allocation[order[k]]++;
        }

        return allocation;
    }
}
