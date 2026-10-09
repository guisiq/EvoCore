using Evo.Core;

namespace Evo.Core.Tests;

// Descritores de fixtures de EVO-011. Dados puros, independentes de Genome<T>.
// Endereços: entradas [0, InputCount); nó n = InputCount + n. Todas as fixtures usam InputCount = 1.
public static class DagFunctions
{
    public const int Add = 0;
    public const int Negate = 1;
    public const int Constant = 2;

    public static int Arity(int function) => function switch
    {
        Add => 2,
        Negate => 1,
        Constant => 0,
        _ => throw new ArgumentOutOfRangeException(nameof(function)),
    };
}

// Function: id da função; Parameter: inteiro usado como gene de parâmetro (Constant); Args: conexões efetivas (aridade).
public sealed record DagNode(int Function, int Parameter, int[] Args);

// Expectativas declaradas à mão. UseCounts[a] = consumidores ativos que referenciam o endereço a mais as referências de output.
public sealed record DagExpectation(int[] ActiveNodes, int Depth, int[] UsedInputs, int[] UseCounts);

public sealed record DagFixture(
    string Name,
    int InputCount,
    int MaxArity,
    int LevelsBack,
    int[] Outputs,
    DagNode[] Nodes,
    int VectorLength,
    DagExpectation Expected)
{
    public int NodeCount => Nodes.Length;
    public int OutputCount => Outputs.Length;
    public GenomeShape Shape => new(InputCount, NodeCount, OutputCount, MaxArity, LevelsBack);

    public int[] FunctionGenes() => Nodes.Select(n => n.Function).ToArray();

    public int[] ConnectionGenes()
    {
        var genes = new int[NodeCount * MaxArity];
        for (var node = 0; node < NodeCount; node++)
        {
            var args = Nodes[node].Args;
            for (var slot = 0; slot < MaxArity; slot++)
            {
                genes[node * MaxArity + slot] = slot < args.Length ? args[slot] : 0;
            }
        }

        return genes;
    }
}

public static class DagFixtureMatrix
{
    private const int Neg = DagFunctions.Negate;
    private const int Add = DagFunctions.Add;
    private const int Const = DagFunctions.Constant;

    private static DagNode N(int function, int parameter, params int[] args) => new(function, parameter, args);

    private static DagFixture Fx(
        string name, int maxArity, int levelsBack, int[] outputs, DagNode[] nodes,
        int[] activeNodes, int depth, int[] usedInputs, int[] useCounts, int vectorLength = 3) =>
        new(name, 1, maxArity, levelsBack, outputs, nodes, vectorLength,
            new DagExpectation(activeNodes, depth, usedInputs, useCounts));

    private static DagNode[] Chain3() => [N(Neg, 0, 0), N(Neg, 0, 1), N(Neg, 0, 2)];

    private static DagNode[] Diamond() => [N(Neg, 0, 0), N(Neg, 0, 0), N(Add, 0, 1, 2)];

    public static readonly IReadOnlyList<DagFixture> All =
    [
        // Cadeia: variação de LevelsBack (1, intermediário, igual e maior que NodeCount = 3).
        Fx("chain_M1_LB1", 1, 1, [3], Chain3(), [0, 1, 2], 3, [0], [1, 1, 1, 1]),
        Fx("chain_M1_LB2", 1, 2, [3], Chain3(), [0, 1, 2], 3, [0], [1, 1, 1, 1]),
        Fx("chain_M1_LB3", 1, 3, [3], Chain3(), [0, 1, 2], 3, [0], [1, 1, 1, 1]),
        Fx("chain_M1_LB5", 1, 5, [3], Chain3(), [0, 1, 2], 3, [0], [1, 1, 1, 1]),
        // Nó único e cadeia de 5 nós (profundidade igual a NodeCount).
        Fx("single_node_M1_LB1", 1, 1, [1], [N(Neg, 0, 0)], [0], 1, [0], [1, 1]),
        Fx("chain5_M1_LB5", 1, 5, [5],
            [N(Neg, 0, 0), N(Neg, 0, 1), N(Neg, 0, 2), N(Neg, 0, 3), N(Neg, 0, 4)],
            [0, 1, 2, 3, 4], 5, [0], [1, 1, 1, 1, 1, 1]),
        // Fan-out com argumento repetido: n1 = Add(n0, n0).
        Fx("fan_out_M2_LB2", 2, 2, [2], [N(Neg, 0, 0), N(Add, 0, 1, 1)], [0, 1], 2, [0], [1, 2, 1]),
        // Diamante em MaxArity 2, 3 (igual a NodeCount) e 4 (maior que NodeCount).
        Fx("diamond_M2_LB3", 2, 3, [3], Diamond(), [0, 1, 2], 2, [0], [2, 1, 1, 1]),
        Fx("diamond_M3_LB3", 3, 3, [3], Diamond(), [0, 1, 2], 2, [0], [2, 1, 1, 1]),
        Fx("diamond_M4_LB3", 4, 3, [3], Diamond(), [0, 1, 2], 2, [0], [2, 1, 1, 1]),
        // OutputCount intermediário (2) e igual a NodeCount (3) sobre o diamante.
        Fx("diamond_O2_M2_LB3", 2, 3, [3, 1], Diamond(), [0, 1, 2], 2, [0], [2, 2, 1, 1]),
        Fx("diamond_O3_M2_LB3", 2, 3, [3, 1, 2], Diamond(), [0, 1, 2], 2, [0], [2, 2, 2, 1]),
        // Diamantes encadeados: n2 = Add(n0, n1), n1 = Add(n0, n0).
        Fx("chained_diamonds_M2_LB3", 2, 3, [3],
            [N(Neg, 0, 0), N(Add, 0, 1, 1), N(Add, 0, 1, 2)], [0, 1, 2], 3, [0], [1, 3, 1, 1]),
        // Consumidores em níveis diferentes: in0 é usado no nível 1 (n0) e no nível 3 (n2).
        Fx("consumers_different_levels_M2_LB3", 2, 3, [3],
            [N(Neg, 0, 0), N(Neg, 0, 1), N(Add, 0, 0, 2)], [0, 1, 2], 3, [0], [2, 1, 1, 1]),
        // Outputs repetidos e output direto de entrada.
        Fx("outputs_repeated_M1_LB1", 1, 1, [1, 1, 1], [N(Neg, 0, 0)], [0], 1, [0], [1, 3]),
        Fx("output_direct_input_M1_LB1", 1, 1, [0, 1], [N(Neg, 0, 0)], [0], 1, [0], [2, 1]),
        // Nó de aridade zero (profundidade 0) consumido por Add.
        Fx("arity_zero_source_M2_LB2", 2, 2, [2], [N(Const, 5), N(Add, 0, 1, 0)], [0, 1], 1, [0], [1, 1, 1]),
        // Nó inativo com o mesmo gene de um nó ativo não entra no grafo ativo.
        Fx("inactive_nodes_M2_LB2", 2, 2, [1], [N(Neg, 0, 0), N(Neg, 0, 0)], [0], 1, [0], [1, 1, 0]),
        // Mesmo diamante com arrays de comprimento zero.
        Fx("diamond_M2_LB3_vetores_vazios", 2, 3, [3], Diamond(), [0, 1, 2], 2, [0], [2, 1, 1, 1], 0),
    ];
}
