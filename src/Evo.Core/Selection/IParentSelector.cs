namespace Evo.Core;

/// <summary>
/// Contrato público de seleção de pais. Implementações devem ser determinísticas dado o
/// <see cref="RandomStream"/> recebido e devem retornar um índice válido de
/// <see cref="ParentSelectionContext{T}.Population"/>.
/// </summary>
public interface IParentSelector<T>
{
    /// <summary>Escolhe o índice de um pai na população do contexto.</summary>
    int SelectParentIndex(ParentSelectionContext<T> context, ref RandomStream random);
}
