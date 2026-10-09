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

    [Fact]
    public void Arity_zero_nodes_are_depth_zero_sources()
    {
        var shape = new GenomeShape(InputCount: 1, NodeCount: 2, OutputCount: 1, MaxArity: 2, LevelsBack: 2);

        var constantOutput = new Genome<double>(shape);
        constantOutput.SetFunctionGene(0, 2); // Constant
        constantOutput.SetParameterGene(0, 1.0);
        constantOutput.SetOutputGene(0, 1); // output = node 0
        Assert.Equal(0, constantOutput.Analyze(DoubleFunctionSet.Instance).ActiveDepth);

        var addConstant = new Genome<double>(shape);
        addConstant.SetFunctionGene(0, 2); // Constant
        addConstant.SetParameterGene(0, 1.0);
        addConstant.SetFunctionGene(1, 0); // Add(node 0, in0)
        addConstant.SetConnectionGene(1, 0, 1);
        addConstant.SetConnectionGene(1, 1, 0);
        addConstant.SetOutputGene(0, 2); // output = node 1
        var graph = addConstant.Analyze(DoubleFunctionSet.Instance);
        Assert.Equal(1, graph.ActiveDepth);
        Assert.Equal(2, graph.ActiveNodeCount);
        Assert.Equal(new[] { 0 }, graph.UsedInputs);
    }

    [Fact]
    public void Shared_topology_is_counted_once_for_double()
    {
        AssertSharedTopology(new OpsFunctionSet<double>((a, b) => a + b, a => -a, p => p), DoubleValueSemantics.Instance, () => 1.0);
    }

    [Fact]
    public void Shared_topology_is_counted_once_for_int()
    {
        AssertSharedTopology(new OpsFunctionSet<int>((a, b) => a + b, a => -a, p => p), Int32ValueSemantics.Instance, () => 1);
    }

    [Fact]
    public void Shared_topology_is_counted_once_for_double_array()
    {
        AssertSharedTopology(
            new OpsFunctionSet<double[]>(
                (a, b) => a.Zip(b, (x, y) => x + y).ToArray(),
                a => a.Select(x => -x).ToArray(),
                p => (double[])p.Clone()),
            DoubleArrayValueSemantics.Instance,
            () => new[] { 1.0, 2.0 });
    }

    [Fact]
    public void Shared_topology_is_counted_once_for_int_array()
    {
        AssertSharedTopology(
            new OpsFunctionSet<int[]>(
                (a, b) => a.Zip(b, (x, y) => x + y).ToArray(),
                a => a.Select(x => -x).ToArray(),
                p => (int[])p.Clone()),
            Int32ArrayValueSemantics.Instance,
            () => new[] { 1, 2 });
    }

    [Fact]
    public void Import_accepts_valid_genes()
    {
        var shape = new GenomeShape(InputCount: 1, NodeCount: 2, OutputCount: 1, MaxArity: 1, LevelsBack: 2);
        ImportDouble(shape, new[] { 1, 1 }, new[] { 0, 1 }, new[] { 0.0, 0.0 }, new[] { 2 });
    }

    [Fact]
    public void Import_rejects_invalid_connections()
    {
        var shape = new GenomeShape(InputCount: 1, NodeCount: 2, OutputCount: 1, MaxArity: 1, LevelsBack: 2);
        var par = new[] { 0.0, 0.0 };
        var outs = new[] { 2 };
        Assert.Throws<InvalidGenomeException>(() => ImportDouble(shape, new[] { 1, 1 }, new[] { 1, 1 }, par, outs)); // self
        Assert.Throws<InvalidGenomeException>(() => ImportDouble(shape, new[] { 1, 1 }, new[] { 2, 1 }, par, outs)); // future
        Assert.Throws<InvalidGenomeException>(() => ImportDouble(shape, new[] { 1, 1 }, new[] { -1, 1 }, par, outs)); // negative
        Assert.Throws<InvalidGenomeException>(() => ImportDouble(shape, new[] { 1, 1 }, new[] { 5, 1 }, par, outs)); // out of range

        var levelsShape = new GenomeShape(InputCount: 1, NodeCount: 3, OutputCount: 1, MaxArity: 1, LevelsBack: 1);
        var levelsPar = new[] { 0.0, 0.0, 0.0 };
        ImportDouble(levelsShape, new[] { 1, 1, 1 }, new[] { 0, 1, 2 }, levelsPar, new[] { 3 });
        Assert.Throws<InvalidGenomeException>(() => ImportDouble(levelsShape, new[] { 1, 1, 1 }, new[] { 0, 1, 1 }, levelsPar, new[] { 3 }));
    }

    [Fact]
    public void Import_rejects_invalid_outputs_functions_and_vector_lengths()
    {
        var shape = new GenomeShape(InputCount: 1, NodeCount: 2, OutputCount: 1, MaxArity: 1, LevelsBack: 2);
        var par = new[] { 0.0, 0.0 };
        var conn = new[] { 0, 1 };
        Assert.Throws<InvalidGenomeException>(() => ImportDouble(shape, new[] { 1, 1 }, conn, par, new[] { 4 })); // output out of range
        Assert.Throws<InvalidGenomeException>(() => ImportDouble(shape, new[] { 1, 1 }, conn, par, new[] { -1 }));
        Assert.Throws<InvalidGenomeException>(() => ImportDouble(shape, new[] { 1, 5 }, conn, par, new[] { 2 })); // function id outside set
        Assert.Throws<InvalidGenomeException>(() => ImportDouble(shape, new[] { 1, -1 }, conn, par, new[] { 2 }));
        Assert.Throws<InvalidGenomeException>(() => ImportDouble(shape, new[] { 1 }, conn, par, new[] { 2 })); // FunctionGenes length
        Assert.Throws<InvalidGenomeException>(() => ImportDouble(shape, new[] { 1, 1 }, new[] { 0 }, par, new[] { 2 })); // ConnectionGenes length
        Assert.Throws<InvalidGenomeException>(() => ImportDouble(shape, new[] { 1, 1 }, conn, new[] { 0.0 }, new[] { 2 })); // ParameterGenes length
        Assert.Throws<InvalidGenomeException>(() => ImportDouble(shape, new[] { 1, 1 }, conn, par, new[] { 2, 2 })); // OutputGenes length
    }

    [Fact]
    public void Fingerprint_for_int_tracks_only_active_genes()
    {
        AssertFingerprintBehaviour(
            new OpsFunctionSet<int>((a, b) => a + b, a => -a, p => p),
            Int32ValueSemantics.Instance,
            () => 1,
            () => 2);
    }

    [Fact]
    public void Fingerprint_for_int_array_tracks_only_active_genes_with_deep_equality()
    {
        AssertFingerprintBehaviour(
            new OpsFunctionSet<int[]>(
                (a, b) => a.Zip(b, (x, y) => x + y).ToArray(),
                a => a.Select(x => -x).ToArray(),
                p => (int[])p.Clone()),
            Int32ArrayValueSemantics.Instance,
            () => new[] { 1, 2 },
            () => new[] { 1, 3 });
    }

    private static void ImportDouble(GenomeShape shape, int[] functionGenes, int[] connectionGenes, double[] parameterGenes, int[] outputGenes) =>
        Genome<double>.FromGenes(
            shape,
            functionGenes,
            connectionGenes,
            parameterGenes,
            outputGenes,
            DoubleValueSemantics.Instance,
            DoubleFunctionSet.Instance);

    // Topology: n0=Add(in0,in0); n1=Add(n0,n0); n2=Negate(n1); n3=Add(n0,n2); n4=Add(in1,in1) inactive.
    // Outputs: n3, n1, n3. Expected: 4 unique active nodes, UsedInputs {0}, ActiveDepth 4 (in0->n0->n1->n2->n3).
    private static void AssertSharedTopology<T>(IFunctionSet<T> functions, IValueSemantics<T> values, Func<T> sample)
    {
        var shape = new GenomeShape(InputCount: 2, NodeCount: 5, OutputCount: 3, MaxArity: 2, LevelsBack: 5);
        var genome = new Genome<T>(shape);
        for (var node = 0; node < shape.NodeCount; node++)
        {
            genome.SetParameterGene(node, sample());
        }

        genome.SetFunctionGene(0, 0);
        genome.SetConnectionGene(0, 0, 0);
        genome.SetConnectionGene(0, 1, 0);
        genome.SetFunctionGene(1, 0);
        genome.SetConnectionGene(1, 0, 2);
        genome.SetConnectionGene(1, 1, 2);
        genome.SetFunctionGene(2, 1);
        genome.SetConnectionGene(2, 0, 3);
        genome.SetFunctionGene(3, 0);
        genome.SetConnectionGene(3, 0, 2);
        genome.SetConnectionGene(3, 1, 4);
        genome.SetFunctionGene(4, 0);
        genome.SetConnectionGene(4, 0, 1);
        genome.SetConnectionGene(4, 1, 1);
        genome.SetOutputGene(0, 5);
        genome.SetOutputGene(1, 3);
        genome.SetOutputGene(2, 5);
        genome.Validate(functions);

        var graph = genome.Analyze(functions);
        Assert.Equal(4, graph.ActiveNodeCount);
        Assert.Equal(new[] { 0, 1, 2, 3 }, graph.ActiveNodes);
        Assert.Equal(new[] { 0 }, graph.UsedInputs);
        Assert.Equal(4, graph.ActiveDepth);

        var before = genome.ComputePhenotypeFingerprint(functions, values);
        genome.SetFunctionGene(4, 2);
        genome.SetParameterGene(4, sample());
        Assert.Equal(before, genome.ComputePhenotypeFingerprint(functions, values));
    }

    // Topology: n0 = Add(in0, in1) active (output); n1 = Negate(in0) and n2 = Constant inactive in the baseline.
    private static void AssertFingerprintBehaviour<T>(IFunctionSet<T> functions, IValueSemantics<T> values, Func<T> p1, Func<T> p2)
    {
        var shape = new GenomeShape(InputCount: 2, NodeCount: 3, OutputCount: 1, MaxArity: 2, LevelsBack: 3);

        Genome<T> Build(T activeParameter, bool alternateInactive)
        {
            var genome = new Genome<T>(shape);
            genome.SetFunctionGene(0, 0);
            genome.SetConnectionGene(0, 0, 0);
            genome.SetConnectionGene(0, 1, 1);
            genome.SetParameterGene(0, activeParameter);
            genome.SetOutputGene(0, 2);
            genome.SetFunctionGene(1, alternateInactive ? 0 : 1);
            genome.SetConnectionGene(1, 0, 0);
            genome.SetConnectionGene(1, 1, alternateInactive ? 1 : 0);
            genome.SetParameterGene(1, alternateInactive ? p2() : p1());
            genome.SetFunctionGene(2, alternateInactive ? 0 : 2);
            genome.SetParameterGene(2, alternateInactive ? p2() : p1());
            return genome;
        }

        byte[] Fingerprint(Genome<T> genome) => genome.ComputePhenotypeFingerprint(functions, values);

        var baseline = Build(p1(), false);
        var before = Fingerprint(baseline);

        Assert.Equal(before, Fingerprint(Build(p1(), true)));
        Assert.Equal(before, Fingerprint(Build(p1(), false)));
        Assert.NotEqual(before, Fingerprint(Build(p2(), false)));

        var connectionChanged = Build(p1(), false);
        connectionChanged.SetConnectionGene(0, 1, 0);
        Assert.NotEqual(before, Fingerprint(connectionChanged));
    }

    private static int RandomAddress(Random random, GenomeShape shape, int node)
    {
        var earliest = Math.Max(0, node - shape.LevelsBack);
        var nodeChoices = node - earliest;
        var choice = random.Next(shape.InputCount + nodeChoices);
        return choice < shape.InputCount ? choice : shape.InputCount + earliest + (choice - shape.InputCount);
    }
}

internal sealed class OpsFunctionSet<T> : IFunctionSet<T>
{
    private readonly IFunction<T>[] _functions;

    public OpsFunctionSet(Func<T, T, T> add, Func<T, T> negate, Func<T, T> copy)
    {
        _functions =
        [
            new Fn("Add", 2, (a, _) => add(a[0], a[1])),
            new Fn("Negate", 1, (a, _) => negate(a[0])),
            new Fn("Constant", 0, (_, p) => copy(p)),
        ];
    }

    public string Id => "test-ops";
    public int Count => _functions.Length;
    public IFunction<T> Get(int functionId) => _functions[functionId];

    private sealed class Fn(string id, int arity, Func<T[], T, T> body) : IFunction<T>
    {
        public string Id => id;
        public int Arity => arity;
        public T Invoke(ReadOnlySpan<T> arguments, T parameter) => body(arguments.ToArray(), parameter);
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
