using Evo.Core.Functions;
using Xunit;

namespace Evo.Core.Tests;

public sealed class ReferenceEvaluationTests
{
    private const double Tolerance = 1e-9;

    // ---------- Tabelas de funções escalares double ----------

    [Theory]
    [InlineData(0, 1.5, 2.25, 3.75)]
    [InlineData(1, 5.0, 7.0, -2.0)]
    [InlineData(2, 3.0, -4.0, -12.0)]
    [InlineData(3, 7.0, 2.0, 3.5)]
    [InlineData(3, 7.0, 0.0, 7.0)]
    [InlineData(3, 7.0, 1e-13, 7.0)]
    [InlineData(3, -1.0, -1e-12, -1.0)]
    [InlineData(4, 2.0, 0.0, -2.0)]
    [InlineData(5, -3.0, 0.0, 3.0)]
    [InlineData(6, -3.0, 0.0, 9.0)]
    [InlineData(7, -16.0, 0.0, 4.0)]
    [InlineData(9, 2.0, -1.0, -1.0)]
    [InlineData(10, 2.0, -1.0, 2.0)]
    [InlineData(11, 2.0, 1.0, 1.0)]
    [InlineData(11, 1.0, 2.0, 0.0)]
    public void Double_function_table_matches_spec(int functionId, double a, double b, double expected)
    {
        var set = StandardDoubleFunctionSet.Instance;
        var function = set.Get(functionId);
        var args = function.Arity == 1 ? new[] { a } : new[] { a, b };
        Assert.Equal(expected, function.Invoke(args, 0.0), Tolerance);
    }

    [Fact]
    public void Double_protected_log_uses_abs_plus_epsilon()
    {
        var set = StandardDoubleFunctionSet.Instance;
        Assert.Equal(Math.Log(1.0 + 1e-12), set.Get(8).Invoke([1.0], 0.0), 15);
        Assert.Equal(Math.Log(2.0 + 1e-12), set.Get(8).Invoke([-2.0], 0.0), 15);
    }

    [Fact]
    public void Double_protected_sqrt_uses_abs()
    {
        Assert.Equal(3.0, StandardDoubleFunctionSet.Instance.Get(7).Invoke([-9.0], 0.0));
    }

    [Fact]
    public void Double_select_uses_condition_greater_than_zero()
    {
        var select = StandardDoubleFunctionSet.Instance.Get(12);
        Assert.Equal(10.0, select.Invoke([1.0, 10.0, 20.0], 0.0));
        Assert.Equal(20.0, select.Invoke([0.0, 10.0, 20.0], 0.0));
        Assert.Equal(20.0, select.Invoke([-1.0, 10.0, 20.0], 0.0));
    }

    [Fact]
    public void Double_constant_returns_parameter_and_has_zero_arity()
    {
        var constant = StandardDoubleFunctionSet.Instance.Get(13);
        Assert.Equal(0, constant.Arity);
        Assert.Equal(4.25, constant.Invoke([], 4.25));
    }

    [Fact]
    public void Double_nan_and_infinity_follow_protected_rules()
    {
        var set = StandardDoubleFunctionSet.Instance;
        Assert.True(double.IsNaN(set.Get(3).Invoke([double.NaN, 1.0], 0.0)));
        Assert.True(double.IsNaN(set.Get(3).Invoke([1.0, double.NaN], 0.0)));
        Assert.Equal(0.0, set.Get(3).Invoke([1.0, double.PositiveInfinity], 0.0));
        Assert.Equal(double.PositiveInfinity, set.Get(3).Invoke([double.PositiveInfinity, 2.0], 0.0));
        Assert.True(double.IsNaN(set.Get(7).Invoke([double.NaN], 0.0)));
        Assert.True(double.IsNaN(set.Get(8).Invoke([double.NaN], 0.0)));
        Assert.True(double.IsNaN(set.Get(11).Invoke([double.NaN, 1.0], 0.0)));
    }

    // ---------- Tabelas de funções escalares int ----------

    [Theory]
    [InlineData(0, 2, 3, 5)]
    [InlineData(1, 2, 3, -1)]
    [InlineData(2, 4, -3, -12)]
    [InlineData(3, 7, 2, 3)]
    [InlineData(3, 7, 0, 7)]
    [InlineData(3, -7, 2, -3)]
    [InlineData(3, 7, -2, -3)]
    [InlineData(4, 5, 0, -5)]
    [InlineData(5, -5, 0, 5)]
    [InlineData(6, -4, 0, 16)]
    [InlineData(7, -2, 9, -2)]
    [InlineData(8, 2, 9, 9)]
    [InlineData(9, 2, 9, 0)]
    [InlineData(10, 2, 9, 9)]
    [InlineData(9, 3, 2, 1)]
    [InlineData(9, 2, 3, 0)]
    public void Int32_function_table_matches_spec(int functionId, int a, int b, int expected)
    {
        var function = StandardInt32FunctionSet.Instance.Get(functionId);
        var args = function.Arity switch
        {
            1 => new[] { a },
            3 => new[] { a, b, 9 },
            _ => new[] { a, b },
        };
        Assert.Equal(expected, function.Invoke(args, 0));
    }

    [Fact]
    public void Int32_select_uses_condition_greater_than_zero()
    {
        var select = StandardInt32FunctionSet.Instance.Get(10);
        Assert.Equal(1, select.Invoke([1, 1, 2], 0));
        Assert.Equal(2, select.Invoke([0, 1, 2], 0));
        Assert.Equal(2, select.Invoke([-5, 1, 2], 0));
    }

    [Fact]
    public void Int32_set_has_no_sqrt_or_log_and_constant_is_last()
    {
        var set = StandardInt32FunctionSet.Instance;
        Assert.Equal(12, set.Count);
        Assert.Equal("Constant", set.Get(11).Id);
        Assert.Equal(0, set.Get(11).Arity);
        Assert.Equal(7, set.Get(11).Invoke([], 7));
    }

    [Fact]
    public void Int32_division_corner_cases()
    {
        var div = StandardInt32FunctionSet.Instance.Get(3);
        Assert.Equal(7, div.Invoke([7, 0], 0));
        Assert.Equal(int.MaxValue, div.Invoke([int.MinValue, -1], 0));
        Assert.Equal(int.MinValue, div.Invoke([int.MinValue, 1], 0));
    }

    [Fact]
    public void Int32_overflow_is_unchecked_and_abs_min_value_saturates()
    {
        var set = StandardInt32FunctionSet.Instance;
        Assert.Equal(int.MinValue, set.Get(0).Invoke([int.MaxValue, 1], 0));
        Assert.Equal(int.MaxValue, set.Get(1).Invoke([int.MinValue, 1], 0));
        Assert.Equal(-2, set.Get(2).Invoke([int.MaxValue, 2], 0));
        Assert.Equal(int.MinValue, set.Get(4).Invoke([int.MinValue], 0));
        Assert.Equal(0, set.Get(6).Invoke([int.MinValue], 0));
        Assert.Equal(int.MaxValue, set.Get(5).Invoke([int.MinValue], 0));
    }

    // ---------- Arrays ----------

    [Fact]
    public void Double_array_add_is_element_wise()
    {
        var add = StandardDoubleArrayFunctionSet.Instance.Get(0);
        Assert.Equal(new[] { 5.0, 7.0, 9.0 }, add.Invoke([new[] { 1.0, 2.0, 3.0 }, new[] { 4.0, 5.0, 6.0 }], []));
    }

    [Fact]
    public void Array_operands_of_different_length_fail_clearly()
    {
        var add = StandardInt32ArrayFunctionSet.Instance.Get(0);
        var error = Assert.Throws<ArgumentException>(() =>
            add.Invoke([new[] { 1, 2 }, new[] { 1 }], []));
        Assert.Contains("same length", error.Message);
    }

    [Fact]
    public void Int32_array_select_and_protected_divide_are_element_wise()
    {
        var set = StandardInt32ArrayFunctionSet.Instance;
        Assert.Equal(new[] { 4, 9 }, set.Get(3).Invoke([new[] { 8, 9 }, new[] { 2, 0 }], []));
        Assert.Equal(new[] { 10, 20 }, set.Get(10).Invoke([new[] { 1, 0 }, new[] { 10, 10 }, new[] { 20, 20 }], []));
    }

    [Fact]
    public void Array_constant_returns_deep_clone_of_parameter()
    {
        var constant = StandardDoubleArrayFunctionSet.Instance.Get(13);
        var parameter = new[] { 1.0, 2.0 };
        var result = constant.Invoke([], parameter);
        Assert.Equal(parameter, result);
        Assert.NotSame(parameter, result);
        result[0] = 99.0;
        Assert.Equal(1.0, parameter[0]);
    }

    // ---------- Avaliador de referência ----------

    [Fact]
    public void Direct_input_to_output_and_repeated_evaluation_is_deterministic()
    {
        var genome = new Genome<int>(new GenomeShape(1, 1, 2, 1, 1));
        genome.SetFunctionGene(0, 4);
        genome.SetConnectionGene(0, 0, 0);
        genome.SetOutputGene(0, 0);
        genome.SetOutputGene(1, 1);

        var evaluator = new ReferenceGenomeEvaluator<int>(StandardInt32FunctionSet.Instance);
        var first = evaluator.Evaluate(genome, [5]);
        var second = evaluator.Evaluate(genome, [5]);
        Assert.Equal(new[] { 5, -5 }, first);
        Assert.Equal(first, second);
    }

    [Fact]
    public void Array_outputs_are_not_corrupted_by_later_calls()
    {
        var genome = new Genome<double[]>(new GenomeShape(1, 1, 2, 1, 1));
        genome.SetFunctionGene(0, 4);
        genome.SetConnectionGene(0, 0, 0);
        genome.SetOutputGene(0, 1);
        genome.SetOutputGene(1, 1);

        var evaluator = new ReferenceGenomeEvaluator<double[]>(StandardDoubleArrayFunctionSet.Instance);
        var first = evaluator.Evaluate(genome, [new[] { 1.0, 2.0 }]);
        var snapshot = (double[])first[0].Clone();
        evaluator.Evaluate(genome, [new[] { 7.0, 8.0 }]);
        Assert.Equal(snapshot, first[0]);
        Assert.Equal(new[] { -1.0, -2.0 }, first[1]);
    }

    [Fact]
    public void Evaluator_rejects_wrong_input_count()
    {
        var genome = new Genome<double>(new GenomeShape(2, 1, 1, 2, 1));
        genome.SetFunctionGene(0, 13);
        genome.SetOutputGene(0, 2);
        var evaluator = new ReferenceGenomeEvaluator<double>(StandardDoubleFunctionSet.Instance);
        Assert.Throws<ArgumentException>(() => evaluator.Evaluate(genome, [1.0]));
    }

    // ---------- EVO-011: execução única por endereço ativo ----------

    [Fact]
    public void Every_fixture_executes_each_active_address_once_in_double_domain()
    {
        foreach (var fixture in DagFixtureMatrix.All)
        {
            CheckFixture<double>(fixture, StandardDoubleFunctionSet.Instance, v => v, (a, b) => a == b, 0);
        }
    }

    [Fact]
    public void Every_fixture_executes_each_active_address_once_in_int_domain()
    {
        foreach (var fixture in DagFixtureMatrix.All)
        {
            CheckFixture<int>(fixture, StandardInt32FunctionSet.Instance, v => v, (a, b) => a == b, 0);
        }
    }

    [Fact]
    public void Every_fixture_executes_each_active_address_once_in_double_array_domain()
    {
        foreach (var fixture in DagFixtureMatrix.All)
        {
            var length = fixture.VectorLength;
            CheckFixture<double[]>(
                fixture,
                StandardDoubleArrayFunctionSet.Instance,
                v => Enumerable.Repeat((double)v, length).ToArray(),
                (a, b) => a.AsSpan().SequenceEqual(b),
                length);
        }
    }

    [Fact]
    public void Every_fixture_executes_each_active_address_once_in_int_array_domain()
    {
        foreach (var fixture in DagFixtureMatrix.All)
        {
            var length = fixture.VectorLength;
            CheckFixture<int[]>(
                fixture,
                StandardInt32ArrayFunctionSet.Instance,
                v => Enumerable.Repeat(v, length).ToArray(),
                (a, b) => a.AsSpan().SequenceEqual(b),
                length);
        }
    }

    private static void CheckFixture<T>(
        DagFixture fixture,
        IFunctionSet<T> standard,
        Func<int, T> lift,
        Func<T, T, bool> equal,
        int vectorLength)
    {
        var counting = new CountingFunctionSet<T>(fixture, standard);
        var genome = BuildGenome(fixture, lift);
        var evaluator = new ReferenceGenomeEvaluator<T>(counting);
        var active = DagSharingOracle.Compute(fixture).ActiveNodes;

        foreach (var input in new[] { 3, -5 })
        {
            Array.Clear(counting.Counts);
            var outputs = evaluator.Evaluate(genome, [lift(input)]);
            var expected = DagSharingOracle.EvaluateScalar(fixture, input);

            for (var node = 0; node < fixture.NodeCount; node++)
            {
                var expectedCount = active.Contains(node) ? 1 : 0;
                Assert.True(
                    counting.Counts[node] == expectedCount,
                    $"{fixture.Name} (vector {vectorLength}, input {input}): node {node} executed {counting.Counts[node]} times, expected {expectedCount}.");
            }

            for (var output = 0; output < outputs.Length; output++)
            {
                Assert.True(
                    equal(outputs[output], lift(expected[output])),
                    $"{fixture.Name}: output {output} differs from oracle for input {input}.");
            }
        }
    }

    private static Genome<T> BuildGenome<T>(DagFixture fixture, Func<int, T> lift)
    {
        var genome = new Genome<T>(fixture.Shape);
        for (var node = 0; node < fixture.NodeCount; node++)
        {
            var description = fixture.Nodes[node];
            genome.SetFunctionGene(node, node);
            genome.SetParameterGene(node, lift(description.Parameter));
            for (var slot = 0; slot < description.Args.Length; slot++)
            {
                genome.SetConnectionGene(node, slot, description.Args[slot]);
            }
        }

        for (var output = 0; output < fixture.OutputCount; output++)
        {
            genome.SetOutputGene(output, fixture.Outputs[output]);
        }

        return genome;
    }

    /// <summary>
    /// Maps each node to its own function gene (the node index) so that execution
    /// counts can be attributed to addresses. The operation is taken from the fixture.
    /// </summary>
    private sealed class CountingFunctionSet<T> : IFunctionSet<T>
    {
        private readonly DagFixture _fixture;
        private readonly IFunctionSet<T> _standard;
        private readonly IFunction<T>[] _functions;

        public CountingFunctionSet(DagFixture fixture, IFunctionSet<T> standard)
        {
            _fixture = fixture;
            _standard = standard;
            Counts = new int[fixture.NodeCount];
            _functions = new IFunction<T>[fixture.NodeCount];
            for (var node = 0; node < fixture.NodeCount; node++)
            {
                _functions[node] = new CountingFunction(this, node, StandardIndex(fixture.Nodes[node].Function));
            }
        }

        public string Id => "counting-" + _standard.Id;

        public int Count => _functions.Length;

        public int[] Counts { get; }

        public IFunction<T> Get(int functionId) => _functions[functionId];

        private int StandardIndex(int fixtureFunction) => fixtureFunction switch
        {
            DagFunctions.Add => 0,
            DagFunctions.Negate => 4,
            DagFunctions.Constant => _standard.Count - 1,
            _ => throw new ArgumentOutOfRangeException(nameof(fixtureFunction)),
        };

        private sealed class CountingFunction : IFunction<T>
        {
            private readonly CountingFunctionSet<T> _owner;
            private readonly int _node;
            private readonly IFunction<T> _inner;

            public CountingFunction(CountingFunctionSet<T> owner, int node, int standardIndex)
            {
                _owner = owner;
                _node = node;
                _inner = owner._standard.Get(standardIndex);
            }

            public string Id => _inner.Id;

            public int Arity => _inner.Arity;

            public T Invoke(ReadOnlySpan<T> arguments, T parameter)
            {
                _owner.Counts[_node]++;
                return _inner.Invoke(arguments, parameter);
            }
        }
    }
}
