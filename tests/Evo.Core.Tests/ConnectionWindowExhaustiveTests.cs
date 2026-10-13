using Evo.Core.Functions;
using Evo.Core.Values;
using Xunit;

namespace Evo.Core.Tests;

// Oráculo escrito a partir da regra de 02-dag-e-evolucao.md, sem reutilizar Genome<T>.
public sealed class ConnectionWindowExhaustiveTests
{
    private static bool OracleAccepts(int inputCount, int levelsBack, int node, int address)
    {
        if (address < 0)
        {
            return false;
        }

        if (address < inputCount)
        {
            return true;
        }

        var target = address - inputCount;
        return target < node && target >= node - levelsBack;
    }

    [Theory]
    [InlineData(1, 3, 2, 1)]
    [InlineData(1, 3, 2, 2)]
    [InlineData(2, 3, 2, 1)]
    [InlineData(2, 4, 3, 2)]
    [InlineData(1, 4, 2, 3)]
    public void Connection_addresses_match_oracle_exhaustively(int inputCount, int nodeCount, int maxArity, int levelsBack)
    {
        var shape = new GenomeShape(inputCount, nodeCount, OutputCount: 1, maxArity, levelsBack);
        var functions = StandardDoubleFunctionSet.Instance;
        var arity = functions.Get(0).Arity;
        Assert.InRange(arity, 0, maxArity);

        for (var node = 0; node < nodeCount; node++)
        {
            for (var address = -1; address <= inputCount + nodeCount; address++)
            {
                // Base válido: todos os slots apontam para a primeira entrada.
                var cg = new int[nodeCount * maxArity];
                var slot = 0;
                var expectedOk = OracleAccepts(inputCount, levelsBack, node, address);
                for (var n = 0; n < nodeCount; n++)
                {
                    for (var a = 0; a < maxArity; a++)
                    {
                        cg[n * maxArity + a] = 0;
                    }
                }

                // Usa somente os slots efetivamente consumidos pela aridade da função 0.
                slot = node * maxArity;
                if (arity > 0)
                {
                    cg[slot] = address;
                }

                var fg = new int[nodeCount];
                var pg = new double[nodeCount];
                var og = new[] { inputCount };

                var ok = true;
                try
                {
                    _ = Genome<double>.FromGenes(shape, fg, cg, pg, og, DoubleValueSemantics.Instance, functions);
                }
                catch (InvalidGenomeException)
                {
                    ok = false;
                }

                if (arity == 0)
                {
                    // Sem consumo de slot: nenhuma conexão deve ser rejeitada por endereço.
                    Assert.True(ok, $"arity 0 must ignore addresses (node {node}, address {address}).");
                }
                else
                {
                    Assert.True(
                        ok == expectedOk,
                        $"shape I={inputCount} N={nodeCount} M={maxArity} LB={levelsBack} node={node} addr={address}: expected {expectedOk}, got {ok}.");
                }
            }
        }
    }
}
