using Evo.Core.Functions;
using Evo.Core.Values;
using Xunit;

namespace Evo.Core.Tests;

// Oráculo estrutural escrito a partir de 02-dag-e-evolucao.md, sem chamar Genome<T>.Analyze.
// Enumera todos os genomas de shapes pequenos (função 0 com aridade 2, saída única) e compara
// conjunto ativo, profundidade, entradas usadas e ordem topológica.
public sealed class ActiveGraphOracleTests
{
    private static List<int> ValidAddresses(int inputCount, int node, int levelsBack)
    {
        var list = new List<int>();
        for (var i = 0; i < inputCount; i++)
        {
            list.Add(i);
        }

        for (var t = Math.Max(0, node - levelsBack); t < node; t++)
        {
            list.Add(inputCount + t);
        }

        return list;
    }

    private static (int[] ActiveNodes, int Depth, int[] Inputs) Oracle(
        int inputCount, int nodeCount, int maxArity, int[] cg, int[] outputs)
    {
        const int Arity = 2;
        var active = new bool[nodeCount];
        var stack = new Stack<int>();
        foreach (var addr in outputs)
        {
            if (addr >= inputCount)
            {
                stack.Push(addr - inputCount);
            }
        }

        while (stack.Count > 0)
        {
            var n = stack.Pop();
            if (active[n])
            {
                continue;
            }

            active[n] = true;
            for (var s = 0; s < Arity; s++)
            {
                var a = cg[n * maxArity + s];
                if (a >= inputCount)
                {
                    stack.Push(a - inputCount);
                }
            }
        }

        var depth = new int[nodeCount];
        var known = new bool[nodeCount];
        int DepthOf(int addr)
        {
            if (addr < inputCount)
            {
                return 0;
            }

            var n = addr - inputCount;
            if (known[n])
            {
                return depth[n];
            }

            var best = 0;
            for (var s = 0; s < Arity; s++)
            {
                best = Math.Max(best, DepthOf(cg[n * maxArity + s]));
            }

            known[n] = true;
            depth[n] = best + 1;
            return depth[n];
        }

        var outDepth = 0;
        var inputs = new SortedSet<int>();
        foreach (var addr in outputs)
        {
            outDepth = Math.Max(outDepth, DepthOf(addr));
            if (addr < inputCount)
            {
                inputs.Add(addr);
            }
        }

        var nodes = new List<int>();
        for (var n = 0; n < nodeCount; n++)
        {
            if (!active[n])
            {
                continue;
            }

            nodes.Add(n);
            for (var s = 0; s < Arity; s++)
            {
                var a = cg[n * maxArity + s];
                if (a < inputCount)
                {
                    inputs.Add(a);
                }
            }
        }

        return (nodes.ToArray(), outDepth, inputs.ToArray());
    }

    [Theory]
    [InlineData(1, 3, 2, 3)]
    [InlineData(2, 3, 2, 1)]
    public void Active_graph_matches_structural_oracle_for_all_small_genomes(int inputCount, int nodeCount, int maxArity, int levelsBack)
    {
        var shape = new GenomeShape(inputCount, nodeCount, 1, maxArity, levelsBack);
        var functions = StandardDoubleFunctionSet.Instance;
        Assert.Equal(2, functions.Get(0).Arity);

        var slotChoices = new List<int>[nodeCount * maxArity];
        for (var s = 0; s < slotChoices.Length; s++)
        {
            slotChoices[s] = ValidAddresses(inputCount, s / maxArity, levelsBack);
        }

        var outputChoices = Enumerable.Range(0, inputCount + nodeCount).ToArray();
        var index = new int[slotChoices.Length];
        var checkedCount = 0;

        while (true)
        {
            var cg = new int[slotChoices.Length];
            for (var s = 0; s < cg.Length; s++)
            {
                cg[s] = slotChoices[s][index[s]];
            }

            foreach (var output in outputChoices)
            {
                var outputs = new[] { output };
                var fg = new int[nodeCount];
                var pg = new double[nodeCount];
                var genome = Genome<double>.FromGenes(shape, fg, cg, pg, outputs, DoubleValueSemantics.Instance, functions);
                var actual = genome.Analyze(functions);
                var expected = Oracle(inputCount, nodeCount, maxArity, cg, outputs);

                Assert.Equal(expected.ActiveNodes, actual.ActiveNodes);
                Assert.Equal(expected.ActiveNodes.Length, actual.ActiveNodeCount);
                Assert.Equal(expected.Depth, actual.ActiveDepth);
                Assert.Equal(expected.Inputs, actual.UsedInputs);
                checkedCount++;
            }

            // Incremento de contador mista sobre as escolhas válidas de cada slot.
            var pos = 0;
            while (pos < index.Length)
            {
                index[pos]++;
                if (index[pos] < slotChoices[pos].Count)
                {
                    break;
                }

                index[pos] = 0;
                pos++;
            }

            if (pos == index.Length)
            {
                break;
            }
        }

        Assert.True(checkedCount > 0);
    }
}
