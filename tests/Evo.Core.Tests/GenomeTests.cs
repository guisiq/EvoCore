using System.Security.Cryptography;
using Evo.Core;
using Evo.Core.Values;
using Xunit;

namespace Evo.Core.Tests;

public sealed class GenomeTests
{
    private static readonly GenomeShape Shape = new(InputCount: 2, NodeCount: 4, OutputCount: 2, MaxArity: 2, LevelsBack: 4);

    private static Genome<double> NewGenome(GenomeShape? shape = null) => new(shape ?? Shape);

    private static byte[] Fingerprint(Genome<double> genome) =>
        genome.ComputePhenotypeFingerprint(DoubleFunctionSet.Instance, DoubleValueSemantics.Instance);

    [Fact]
    public void Layout_uses_node_major_connection_index()
    {
        var genome = NewGenome();
        genome.SetConnectionGene(2, 1, 3);

        Assert.Equal(4 * 2, genome.ConnectionGenes.Length);
        Assert.Equal(3, genome.ConnectionGenes[2 * 2 + 1]);
        Assert.Equal(4, genome.FunctionGenes.Length);
        Assert.Equal(4, genome.ParameterGenes.Length);
        Assert.Equal(2, genome.OutputGenes.Length);
    }

    [Theory]
    [InlineData(0, 0, 2)] // self (node 0 is address 2)
    [InlineData(0, 0, 3)] // future node
    [InlineData(2, 0, 4)] // node 2 pointing to itself (address 4 = node 2)
    [InlineData(3, 0, -1)] // negative address
    [InlineData(3, 0, 99)] // out of range
    public void Invalid_connections_fail_with_identifiable_error(int node, int slot, int address)
    {
        var genome = NewGenome();
        var ex = Assert.Throws<InvalidGenomeException>(() => genome.SetConnectionGene(node, slot, address));
        Assert.False(string.IsNullOrWhiteSpace(ex.Message));
    }

    [Fact]
    public void LevelsBack_limits_reachable_predecessors()
    {
        var shape = new GenomeShape(InputCount: 1, NodeCount: 4, OutputCount: 1, MaxArity: 1, LevelsBack: 1);
        var genome = new Genome<double>(shape);

        genome.SetConnectionGene(3, 0, 1 + 2); // node 3 -> node 2 (within 1 level)
        var ex = Assert.Throws<InvalidGenomeException>(() => genome.SetConnectionGene(3, 0, 1 + 0));
        Assert.Contains("LevelsBack", ex.Message);
    }

    [Fact]
    public void Invalid_shape_and_gene_values_are_rejected()
    {
        Assert.Throws<InvalidGenomeException>(() => new Genome<double>(Shape with { LevelsBack = 0 }));
        Assert.Throws<InvalidGenomeException>(() => new Genome<double>(Shape with { NodeCount = 0 }));
        var genome = NewGenome();
        Assert.Throws<InvalidGenomeException>(() => genome.SetFunctionGene(0, -1));
        Assert.Throws<InvalidGenomeException>(() => genome.SetOutputGene(0, 99));
    }

    [Fact]
    public void Changing_only_inactive_genes_preserves_fingerprint()
    {
        var genome = NewGenome();
        genome.SetFunctionGene(0, 0); // Add (arity 2)
        genome.SetConnectionGene(0, 0, 0);
        genome.SetConnectionGene(0, 1, 1);
        genome.SetOutputGene(0, 2); // output = node 0
        genome.SetOutputGene(1, 0);
        var before = Fingerprint(genome);

        genome.SetFunctionGene(3, 2); // node 3 inactive: Constant (arity 0)
        genome.SetParameterGene(3, 7.5);
        genome.SetConnectionGene(3, 1, 4); // inactive slot of inactive node
        Assert.Equal(before, Fingerprint(genome));
    }

    [Fact]
    public void Changing_an_active_gene_changes_fingerprint()
    {
        var genome = NewGenome();
        genome.SetFunctionGene(0, 0);
        genome.SetConnectionGene(0, 0, 0);
        genome.SetConnectionGene(0, 1, 1);
        genome.SetOutputGene(0, 2);
        genome.SetOutputGene(1, 0);
        var before = Fingerprint(genome);

        genome.SetConnectionGene(0, 1, 0);
        Assert.NotEqual(before, Fingerprint(genome));
    }

    [Fact]
    public void Shared_nodes_repeated_arguments_and_outputs_are_counted_once()
    {
        // n0 = Add(in0, in0); n1 = Add(n0, n0); n2 = Add(n0, in1) [inactive]; outputs: n1, n1.
        var shape = new GenomeShape(InputCount: 2, NodeCount: 3, OutputCount: 2, MaxArity: 2, LevelsBack: 3);
        var genome = new Genome<double>(shape);
        genome.SetFunctionGene(0, 0);
        genome.SetConnectionGene(0, 0, 0);
        genome.SetConnectionGene(0, 1, 0);
        genome.SetFunctionGene(1, 0);
        genome.SetConnectionGene(1, 0, 2); // n0
        genome.SetConnectionGene(1, 1, 2); // n0 again
        genome.SetFunctionGene(2, 0);
        genome.SetConnectionGene(2, 0, 2);
        genome.SetConnectionGene(2, 1, 1);
        genome.SetOutputGene(0, 3); // n1
        genome.SetOutputGene(1, 3); // n1 again

        var graph = genome.Analyze(DoubleFunctionSet.Instance);
        Assert.Equal(2, graph.ActiveNodeCount);
        Assert.Equal(new[] { 0, 1 }, graph.ActiveNodes);
        Assert.Equal(new[] { 0 }, graph.UsedInputs);
        Assert.Equal(2, graph.ActiveDepth);
    }

    [Fact]
    public void Reconvergence_at_different_depths_uses_longest_path()
    {
        // n0 = Negate(in0); n1 = Negate(n0); n2 = Add(in0, n1); output = n2 and in0 deep path.
        var shape = new GenomeShape(InputCount: 1, NodeCount: 3, OutputCount: 1, MaxArity: 2, LevelsBack: 3);
        var genome = new Genome<double>(shape);
        genome.SetFunctionGene(0, 1); // Negate
        genome.SetConnectionGene(0, 0, 0);
        genome.SetFunctionGene(1, 1);
        genome.SetConnectionGene(1, 0, 1); // n0
        genome.SetFunctionGene(2, 0); // Add
        genome.SetConnectionGene(2, 0, 0); // in0 (shallow path)
        genome.SetConnectionGene(2, 1, 1 + 1); // n1 (deep path)
        genome.SetOutputGene(0, 3); // n2

        var graph = genome.Analyze(DoubleFunctionSet.Instance);
        Assert.Equal(3, graph.ActiveNodeCount);
        Assert.Equal(3, graph.ActiveDepth);
    }

    [Fact]
    public void Ten_thousand_generated_valid_genomes_have_no_cycles()
    {
        var random = new Random(12345);
        var shape = new GenomeShape(InputCount: 3, NodeCount: 20, OutputCount: 2, MaxArity: 2, LevelsBack: 5);
        var functions = DoubleFunctionSet.Instance;
        for (var g = 0; g < 10_000; g++)
        {
            var genome = new Genome<double>(shape);
            for (var node = 0; node < shape.NodeCount; node++)
            {
                genome.SetFunctionGene(node, random.Next(functions.Count));
                genome.SetParameterGene(node, random.NextDouble());
                for (var slot = 0; slot < shape.MaxArity; slot++)
                {
                    genome.SetConnectionGene(node, slot, RandomAddress(random, shape, node));
                }
            }

            for (var o = 0; o < shape.OutputCount; o++)
            {
                genome.SetOutputGene(o, random.Next(shape.InputCount + shape.NodeCount));
            }

            genome.Validate(functions);
            var graph = genome.Analyze(functions);
            foreach (var node in graph.ActiveNodes)
            {
                Assert.True(node < shape.NodeCount);
            }
        }
    }

    [Fact]
    public void Array_parameters_use_deep_clone_and_content_fingerprint()
    {
        var shape = new GenomeShape(InputCount: 1, NodeCount: 1, OutputCount: 1, MaxArity: 1, LevelsBack: 1);
        var values = DoubleArrayValueSemantics.Instance;
        var functions = new DoubleArrayFunctionSet();
        var parameter = new[] { 1.0, 2.0 };
        var genome = Genome<double[]>.FromGenes(
            shape,
            new[] { 0 },
            new[] { 0 },
            new[] { parameter },
            new[] { 1 },
            values,
            functions);

        parameter[0] = 99.0; // caller mutation must not reach the genome
        var clone = genome.Clone(values);
        Assert.Equal(1.0, genome.ParameterGenes[0][0]);
        Assert.NotSame(genome.ParameterGenes[0], clone.ParameterGenes[0]);
        Assert.True(values.ValueEquals(genome.ParameterGenes[0], clone.ParameterGenes[0]));

        var before = genome.ComputePhenotypeFingerprint(functions, values);
        var changed = genome.Clone(values);
        changed.SetParameterGene(0, new[] { 1.0, 3.0 });
        Assert.NotEqual(before, changed.ComputePhenotypeFingerprint(functions, values));
    }

    private static int RandomAddress(Random random, GenomeShape shape, int node)
    {
        var earliest = Math.Max(0, node - shape.LevelsBack);
        var nodeChoices = node - earliest;
        var choice = random.Next(shape.InputCount + nodeChoices);
        return choice < shape.InputCount ? choice : shape.InputCount + earliest + (choice - shape.InputCount);
    }
}

internal sealed class DoubleFunctionSet : IFunctionSet<double>
{
    public static readonly DoubleFunctionSet Instance = new();
    private static readonly IFunction<double>[] Functions =
    {
        new Fn("Add", 2, (a, _) => a[0] + a[1]),
        new Fn("Negate", 1, (a, _) => -a[0]),
        new Fn("Constant", 0, (_, p) => p),
    };

    public string Id => "test-double";
    public int Count => Functions.Length;
    public IFunction<double> Get(int functionId) => Functions[functionId];

    private sealed class Fn(string id, int arity, Func<double[], double, double> body) : IFunction<double>
    {
        public string Id => id;
        public int Arity => arity;
        public double Invoke(ReadOnlySpan<double> arguments, double parameter) => body(arguments.ToArray(), parameter);
    }
}

internal sealed class DoubleArrayFunctionSet : IFunctionSet<double[]>
{
    public string Id => "test-double-array";
    public int Count => 1;
    public IFunction<double[]> Get(int functionId) => new Fn();

    private sealed class Fn : IFunction<double[]>
    {
        public string Id => "Constant";
        public int Arity => 0;
        public double[] Invoke(ReadOnlySpan<double[]> arguments, double[] parameter) => (double[])parameter.Clone();
    }
}
