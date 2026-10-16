namespace Evo.Core;

/// <summary>
/// Contrato público de variação: produz um descendente validado a partir do contexto da geração.
/// Implementações devem ser determinísticas dado o <see cref="RandomStream"/> recebido.
/// </summary>
public interface IVariationOperator<T>
{
    /// <summary>Gera um descendente.</summary>
    Genome<T> CreateOffspring(VariationContext<T> context, ref RandomStream random);
}
