using Evo.Core;
using Evo.Core.Values;
using Xunit;

namespace Evo.Core.Tests;

public sealed partial class DagSharingContractTests
{
    public static IEnumerable<object[]> FixtureNames() =>
        DagFixtureMatrix.All.Select(f => new object[] { f.Name });

    public static IEnumerable<object[]> FixtureDomainCases() =>
        from f in DagFixtureMatrix.All
        from domain in new[] { "double", "int", "double[]", "int[]" }
        select new object[] { f.Name, domain };

    [Theory]
    [MemberData(nameof(FixtureNames))]
    public void Fixture_declared_expectations_match_independent_oracle(string name)
    {
        var f = DagFixtureMatrix.All.Single(x => x.Name == name);
        var o = DagSharingOracle.Compute(f);
        Assert.Equal(f.Expected.ActiveNodes, o.ActiveNodes);
        Assert.Equal(f.Expected.Depth, o.Depth);
        Assert.Equal(f.Expected.UsedInputs, o.UsedInputs);
        Assert.Equal(f.Expected.UseCounts, o.UseCounts);
    }

    [Theory]
    [MemberData(nameof(FixtureDomainCases))]
    public void Every_fixture_matches_oracle_and_genome_analyze_in_each_domain(string name, string domain)
    {
        var f = DagFixtureMatrix.All.Single(x => x.Name == name);
        switch (domain)
        {
            case "double":
                CheckDomain(f, DoubleFunctionSet.Instance, DoubleValueSemantics.Instance, d => (double)d);
                break;
            case "int":
                CheckDomain(
                    f,
                    new OpsFunctionSet<int>((a, b) => a + b, a => -a, p => p),
                    Int32ValueSemantics.Instance,
                    d => d);
                break;
            case "double[]":
                CheckDomain(
                    f,
                    new OpsFunctionSet<double[]>(
                        (a, b) => a.Zip(b, (x, y) => x + y).ToArray(),
                        a => a.Select(x => -x).ToArray(),
                        p => (double[])p.Clone()),
                    DoubleArrayValueSemantics.Instance,
                    d => Enumerable.Repeat((double)d, f.VectorLength).ToArray());
                break;
            case "int[]":
                CheckDomain(
                    f,
                    new OpsFunctionSet<int[]>(
                        (a, b) => a.Zip(b, (x, y) => x + y).ToArray(),
                        a => a.Select(x => -x).ToArray(),
                        p => (int[])p.Clone()),
                    Int32ArrayValueSemantics.Instance,
                    d => Enumerable.Repeat(d, f.VectorLength).ToArray());
                break;
            default:
                throw new ArgumentOutOfRangeException(nameof(domain));
        }
    }

    private static void CheckDomain<T>(DagFixture f, IFunctionSet<T> set, IValueSemantics<T> values, Func<int, T> lift)
    {
        const int inputValue = 7;
        var genome = Genome<T>.FromGenes(
            f.Shape,
            f.FunctionGenes(),
            f.ConnectionGenes(),
            f.Nodes.Select(n => lift(n.Parameter)).ToArray(),
            f.Outputs,
            values,
            set);

        var oracle = DagSharingOracle.Compute(f);
        var graph = genome.Analyze(set);
        Assert.Equal(oracle.ActiveNodes, graph.ActiveNodes);
        Assert.Equal(oracle.ActiveNodes.Length, graph.ActiveNodeCount);
        Assert.Equal(oracle.Depth, graph.ActiveDepth);
        Assert.Equal(oracle.UsedInputs, graph.UsedInputs);

        // Avaliação própria com o mesmo tipo T: valores de arrays devem fluir pelos nós compartilhados.
        var memo = new Dictionary<int, T>();
        T Value(int address)
        {
            if (address < f.InputCount)
            {
                return lift(inputValue);
            }

            var node = address - f.InputCount;
            if (memo.TryGetValue(node, out var cached))
            {
                return cached;
            }

            var spec = f.Nodes[node];
            var arity = DagFunctions.Arity(spec.Function);
            var args = new T[arity];
            for (var slot = 0; slot < arity; slot++)
            {
                args[slot] = Value(spec.Args[slot]);
            }

            var result = set.Get(spec.Function).Invoke(args, lift(spec.Parameter));
            memo[node] = result;
            return result;
        }

        var expected = DagSharingOracle.EvaluateScalar(f, inputValue);
        for (var i = 0; i < f.Outputs.Length; i++)
        {
            Assert.True(values.ValueEquals(Value(f.Outputs[i]), lift(expected[i])), $"{f.Name} output {i}");
        }

        _ = genome;
    }

    [Theory]
    [InlineData("self", 1, 1, 0, 1, "self or future")]
    [InlineData("future", 2, 2, 0, 2, "self or future")]
    [InlineData("inexistent", 1, 1, 0, -1, "must be >= 0")]
    [InlineData("window", 3, 1, 2, 1, "violates LevelsBack=1")]
    [InlineData("levelsBack<=0", 2, 0, 1, 1, "LevelsBack must be > 0, got 0.")]
    public void Invalid_connection_addresses_fail_with_identifiable_error(
        string kind, int nodeCount, int levelsBack, int badNode, int badAddress, string fragment)
    {
        // Negate chain-like shape: every node takes address 0 except badNode, which takes badAddress.
        var connections = new int[nodeCount];
        for (var node = 0; node < nodeCount; node++)
        {
            connections[node] = node == badNode ? badAddress : 0;
        }

        var functions = Enumerable.Repeat(DagFunctions.Negate, nodeCount).ToArray();
        var parameters = new double[nodeCount];
        var shape = new GenomeShape(InputCount: 1, NodeCount: nodeCount, OutputCount: 1, MaxArity: 1, LevelsBack: levelsBack);
        var ex = Assert.Throws<InvalidGenomeException>(() =>
            Genome<double>.FromGenes(shape, functions, connections, parameters, [nodeCount], DoubleValueSemantics.Instance, DoubleFunctionSet.Instance));
        // LevelsBack<=0 is rejected at shape level, before any node is checked.
        if (kind != "levelsBack<=0")
        {
            Assert.Contains($"Node {badNode} ", ex.Message);
        }

        Assert.True(ex.Message.Contains(fragment, StringComparison.Ordinal), $"{kind}: {ex.Message}");
    }
}
