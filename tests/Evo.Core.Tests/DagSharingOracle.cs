namespace Evo.Core.Tests;

public sealed record DagOracleResult(int[] ActiveNodes, int Depth, int[] UsedInputs, int[] UseCounts);

// Oráculo independente de Genome<T>.Analyze: percurso próprio sobre o descritor.
public static class DagSharingOracle
{
    public static DagOracleResult Compute(DagFixture f)
    {
        var inputs = f.InputCount;
        var nodes = f.NodeCount;
        var active = new bool[nodes];

        // Fecho ativo por DFS a partir dos outputs, seguindo apenas as conexões efetivas (aridade).
        var stack = new Stack<int>();
        foreach (var output in f.Outputs)
        {
            stack.Push(output);
        }

        while (stack.Count > 0)
        {
            var address = stack.Pop();
            if (address < inputs)
            {
                continue;
            }

            var node = address - inputs;
            if (active[node])
            {
                continue;
            }

            active[node] = true;
            var arity = DagFunctions.Arity(f.Nodes[node].Function);
            for (var slot = 0; slot < arity; slot++)
            {
                stack.Push(f.Nodes[node].Args[slot]);
            }
        }

        // Janela: cada aresta ativa de n para t deve ter t < n e n - t <= LevelsBack.
        for (var node = 0; node < nodes; node++)
        {
            if (!active[node])
            {
                continue;
            }

            var arity = DagFunctions.Arity(f.Nodes[node].Function);
            for (var slot = 0; slot < arity; slot++)
            {
                var address = f.Nodes[node].Args[slot];
                if (address < inputs)
                {
                    continue;
                }

                var target = address - inputs;
                if (target >= node || node - target > f.LevelsBack)
                {
                    throw new InvalidOperationException($"Fixture {f.Name}: aresta {node}->{target} viola LevelsBack={f.LevelsBack}.");
                }
            }
        }

        // Profundidade em arestas (02 §3.1): entrada = 0, aridade zero = 0, demais = 1 + máximo dos argumentos.
        var depth = new int[nodes];
        for (var node = 0; node < nodes; node++)
        {
            if (!active[node])
            {
                continue;
            }

            var arity = DagFunctions.Arity(f.Nodes[node].Function);
            var max = -1;
            for (var slot = 0; slot < arity; slot++)
            {
                max = Math.Max(max, DepthOf(f.Nodes[node].Args[slot], depth, inputs));
            }

            depth[node] = arity == 0 ? 0 : max + 1;
        }

        var maxDepth = 0;
        var usedInputs = new SortedSet<int>();
        var useCounts = new int[inputs + nodes];
        foreach (var output in f.Outputs)
        {
            maxDepth = Math.Max(maxDepth, DepthOf(output, depth, inputs));
            useCounts[output]++;
            if (output < inputs)
            {
                usedInputs.Add(output);
            }
        }

        for (var node = 0; node < nodes; node++)
        {
            if (!active[node])
            {
                continue;
            }

            var arity = DagFunctions.Arity(f.Nodes[node].Function);
            for (var slot = 0; slot < arity; slot++)
            {
                var address = f.Nodes[node].Args[slot];
                useCounts[address]++;
                if (address < inputs)
                {
                    usedInputs.Add(address);
                }
            }
        }

        var activeNodes = Enumerable.Range(0, nodes).Where(n => active[n]).ToArray();
        return new DagOracleResult(activeNodes, maxDepth, usedInputs.ToArray(), useCounts);
    }

    // Avaliação escalar de referência (entradas = inputValue) para comparar valores nos quatro domínios.
    public static int[] EvaluateScalar(DagFixture f, int inputValue)
    {
        var memo = new Dictionary<int, int>();
        int Value(int address)
        {
            if (address < f.InputCount)
            {
                return inputValue;
            }

            var node = address - f.InputCount;
            if (memo.TryGetValue(node, out var cached))
            {
                return cached;
            }

            var spec = f.Nodes[node];
            var result = spec.Function switch
            {
                DagFunctions.Add => Value(spec.Args[0]) + Value(spec.Args[1]),
                DagFunctions.Negate => -Value(spec.Args[0]),
                _ => spec.Parameter,
            };
            memo[node] = result;
            return result;
        }

        return f.Outputs.Select(Value).ToArray();
    }

    private static int DepthOf(int address, int[] depth, int inputs) =>
        address < inputs ? 0 : depth[address - inputs];
}
