using Evo.Core;
using Evo.Core.Values;
using Xunit;

namespace Evo.Core.Tests;

// Fixtures: DoubleFunctionSet ids 0=Add(2), 1=Negate(1), 2=Constant(0).
// Expectations are computed by hand (see evidence/002a-dag-sharing-contract/implementation.md)
// and are independent of Genome<T>.Analyze. Addresses: inputs [0, InputCount), node n = InputCount + n.
public sealed class DagSharingContractTests
{
    private const int Add = 0;
    private const int Negate = 1;
    private const int Constant = 2;

    private static Genome<double> BuildDouble(GenomeShape shape, int[] fn, int[] conn, double[] p, int[] outs) =>
        Genome<double>.FromGenes(shape, fn, conn, p, outs, DoubleValueSemantics.Instance, DoubleFunctionSet.Instance);

    private static void AssertGraph(Genome<double> g, int activeNodes, int depth, int[] usedInputs, int[] nodes)
    {
        var graph = g.Analyze(DoubleFunctionSet.Instance);
        Assert.Equal(activeNodes, graph.ActiveNodeCount);
        Assert.Equal(depth, graph.ActiveDepth);
        Assert.Equal(usedInputs, graph.UsedInputs);
        Assert.Equal(nodes, graph.ActiveNodes);
    }

    [Theory]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(3)]
    [InlineData(5)]
    public void Chain_is_valid_for_levels_back_1_intermediate_equal_and_greater_than_node_count(int levelsBack)
    {
        // in0 -> n0 -> n1 -> n2 (output). Every edge has distance 1, so any LevelsBack >= 1 is valid.
        var shape = new GenomeShape(InputCount: 1, NodeCount: 3, OutputCount: 1, MaxArity: 2, LevelsBack: levelsBack);
        var g = BuildDouble(shape, [Negate, Negate, Negate], [0, 0, 1, 0, 2, 0], [0, 0, 0], [3]);
        AssertGraph(g, 3, 3, [0], [0, 1, 2]);
    }

    [Fact]
    public void Fan_out_counts_shared_node_once_with_repeated_argument()
    {
        // n0 = Negate(in0); n1 = Add(n0, n0); output n1.
        var shape = new GenomeShape(InputCount: 1, NodeCount: 2, OutputCount: 1, MaxArity: 2, LevelsBack: 2);
        var g = BuildDouble(shape, [Negate, Add], [0, 0, 1, 1], [0, 0], [2]);
        AssertGraph(g, 2, 2, [0], [0, 1]);
    }

    [Fact]
    public void Diamond_shares_one_source_between_two_branches()
    {
        // n0 = Negate(in0); n1 = Negate(in0); n2 = Add(n0, n1); output n2.
        var shape = new GenomeShape(InputCount: 1, NodeCount: 3, OutputCount: 1, MaxArity: 2, LevelsBack: 3);
        var g = BuildDouble(shape, [Negate, Negate, Add], [0, 0, 0, 0, 1, 2], [0, 0, 0], [3]);
        AssertGraph(g, 3, 2, [0], [0, 1, 2]);
    }

    [Fact]
    public void Chained_diamonds_report_longest_path_not_tree_expansion()
    {
        // n0 = Negate(in0); n1 = Add(n0, n0); n2 = Add(n0, n1); output n2.
        // Tree expansion would count n0 three times; the DAG counts 3 distinct addresses, depth 3.
        var shape = new GenomeShape(InputCount: 1, NodeCount: 3, OutputCount: 1, MaxArity: 2, LevelsBack: 3);
        var g = BuildDouble(shape, [Negate, Add, Add], [0, 0, 1, 1, 1, 2], [0, 0, 0], [3]);
        AssertGraph(g, 3, 3, [0], [0, 1, 2]);
    }

    [Fact]
    public void Output_may_address_an_input_directly_alongside_a_node()
    {
        // outputs = [in0, n0], n0 = Negate(in0). Depth is the maximum over outputs.
        var shape = new GenomeShape(InputCount: 1, NodeCount: 1, OutputCount: 2, MaxArity: 1, LevelsBack: 1);
        var g = BuildDouble(shape, [Negate], [0], [0], [0, 1]);
        AssertGraph(g, 1, 1, [0], [0]);
    }

    [Fact]
    public void Output_repeated_addresses_the_same_node_without_double_counting()
    {
        var shape = new GenomeShape(InputCount: 1, NodeCount: 1, OutputCount: 3, MaxArity: 1, LevelsBack: 1);
        var g = BuildDouble(shape, [Negate], [0], [0], [1, 1, 1]);
        AssertGraph(g, 1, 1, [0], [0]);
    }

    [Fact]
    public void Arity_zero_node_is_a_depth_zero_source()
    {
        // n0 = Constant(2.5); n1 = Add(n0, in0); output n1.
        var shape = new GenomeShape(InputCount: 1, NodeCount: 2, OutputCount: 1, MaxArity: 2, LevelsBack: 2);
        var g = BuildDouble(shape, [Constant, Add], [0, 0, 1, 0], [2.5, 0], [2]);
        AssertGraph(g, 2, 1, [0], [0, 1]);
    }

    [Fact]
    public void Inactive_nodes_and_extra_slots_do_not_enter_the_active_graph()
    {
        // n0 = Negate(in0) is the output; n1 = Negate(in0) is inactive.
        // Slot 1 of Negate (arity 1) is an inactive slot and must be ignored.
        var shape = new GenomeShape(InputCount: 1, NodeCount: 2, OutputCount: 1, MaxArity: 2, LevelsBack: 2);
        var g = BuildDouble(shape, [Negate, Negate], [0, 0, 0, 0], [0, 0], [1]);
        AssertGraph(g, 1, 1, [0], [0]);
    }

    [Fact]
    public void Same_expression_at_distinct_addresses_is_counted_twice()
    {
        // n0 and n1 are both Negate(in0): identical expressions, distinct addresses.
        var shape = new GenomeShape(InputCount: 1, NodeCount: 3, OutputCount: 1, MaxArity: 2, LevelsBack: 3);
        var g = BuildDouble(shape, [Negate, Negate, Add], [0, 0, 0, 0, 1, 2], [0, 0, 0], [3]);
        var graph = g.Analyze(DoubleFunctionSet.Instance);
        Assert.Equal(3, graph.ActiveNodeCount);
    }

    [Fact]
    public void Levels_back_window_rejects_reference_beyond_the_window()
    {
        // LevelsBack = 1: n2 may reference n1 (address 2) but not n0 (address 1).
        var shape = new GenomeShape(InputCount: 1, NodeCount: 3, OutputCount: 1, MaxArity: 1, LevelsBack: 1);
        var ex = Assert.Throws<InvalidGenomeException>(() =>
            BuildDouble(shape, [Negate, Negate, Negate], [0, 1, 1], [0, 0, 0], [3]));
        Assert.Contains("LevelsBack", ex.Message);
    }

    [Theory]
    [InlineData(new[] { 3 }, "self")]
    [InlineData(new[] { 4 }, "future")]
    [InlineData(new[] { 99 }, "out of range")]
    [InlineData(new[] { -1 }, "negative")]
    public void Invalid_connection_addresses_fail_with_identifiable_error(int[] connection, string label)
    {
        // Single Negate node n0 (address 1). Address 1 is self, 2+ is future, 99 out of range, -1 negative.
        var shape = new GenomeShape(InputCount: 1, NodeCount: 1, OutputCount: 1, MaxArity: 1, LevelsBack: 1);
        var ex = Assert.Throws<InvalidGenomeException>(() =>
            BuildDouble(shape, [Negate], connection, [0], [1]));
        Assert.False(string.IsNullOrWhiteSpace(ex.Message), label);
    }

    [Fact]
    public void Invalid_output_address_is_rejected()
    {
        var shape = new GenomeShape(InputCount: 1, NodeCount: 1, OutputCount: 1, MaxArity: 1, LevelsBack: 1);
        Assert.Throws<InvalidGenomeException>(() => BuildDouble(shape, [Negate], [0], [0], [42]));
    }

    [Fact]
    public void Fingerprint_distinguishes_shared_from_expanded_structure()
    {
        // Shared: n0 feeds both args of n1. Expanded: n1 gets two separate Negate nodes of the same input.
        var sharedShape = new GenomeShape(InputCount: 1, NodeCount: 2, OutputCount: 1, MaxArity: 2, LevelsBack: 2);
        var shared = BuildDouble(sharedShape, [Negate, Add], [0, 0, 1, 1], [0, 0], [2]);
        var expandedShape = new GenomeShape(InputCount: 1, NodeCount: 3, OutputCount: 1, MaxArity: 2, LevelsBack: 3);
        var expanded = BuildDouble(expandedShape, [Negate, Negate, Add], [0, 0, 0, 0, 1, 2], [0, 0, 0], [3]);

        var fpShared = shared.ComputePhenotypeFingerprint(DoubleFunctionSet.Instance, DoubleValueSemantics.Instance);
        var fpExpanded = expanded.ComputePhenotypeFingerprint(DoubleFunctionSet.Instance, DoubleValueSemantics.Instance);
        Assert.NotEqual(fpShared, fpExpanded);
    }

    [Fact]
    public void Int_domain_shares_one_node_across_two_consumers()
    {
        // n0 = Negate(in0); n1 = Add(n0, n0); output n1.
        var set = new OpsFunctionSet<int>(
            add: (a, b) => a + b,
            negate: a => -a,
            copy: p => p);
        var shape = new GenomeShape(InputCount: 1, NodeCount: 2, OutputCount: 1, MaxArity: 2, LevelsBack: 2);
        var g = Genome<int>.FromGenes(shape, [1, 0], [0, 0, 1, 1], [0, 0], [2], Int32ValueSemantics.Instance, set);
        var graph = g.Analyze(set);
        Assert.Equal(2, graph.ActiveNodeCount);
        Assert.Equal(2, graph.ActiveDepth);
        Assert.Equal(new[] { 0 }, graph.UsedInputs);
    }

    [Fact]
    public void Double_array_domain_handles_empty_and_non_empty_constants()
    {
        // Arity-zero nodes only; output is a constant node. Empty arrays are valid values.
        var set = new DoubleArrayFunctionSet();
        var shape = new GenomeShape(InputCount: 1, NodeCount: 2, OutputCount: 1, MaxArity: 1, LevelsBack: 2);
        var g = Genome<double[]>.FromGenes(
            shape,
            [0, 0],
            [0, 0],
            [Array.Empty<double>(), new[] { 1.0, 2.0 }],
            [2],
            DoubleArrayValueSemantics.Instance,
            set);
        var graph = g.Analyze(set);
        Assert.Equal(1, graph.ActiveNodeCount);
        Assert.Equal(0, graph.ActiveDepth);
        Assert.Equal(new[] { 1 }, graph.ActiveNodes);
    }
}
