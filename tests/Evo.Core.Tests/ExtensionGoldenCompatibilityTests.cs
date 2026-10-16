using Evo.Core.Functions;
using Evo.Core.Values;
using Xunit;

namespace Evo.Core.Tests;

public sealed class ExtensionGoldenCompatibilityTests
{
    private static readonly GenomeShape Shape = new(1, 8, 1, 2, 3);
    private static readonly int[] Seeds = { 1, 7, 42, 2024 };

    private static readonly int[] WorkerCounts = { 1, 2, Math.Max(2, Environment.ProcessorCount) };

    [Fact]
    public void Golden_double()
    {
        AssertGolden(
            () => new EvolutionTestProblem<double>(Shape, StandardDoubleFunctionSet.Instance, DoubleValueSemantics.Instance,
                DoubleParameterStrategy.Instance, () => new QuadraticFitness()),
            StandardDoubleFunctionSet.Instance,
            DoubleValueSemantics.Instance);
    }

    [Fact]
    public void Golden_int()
    {
        AssertGolden(
            () => new EvolutionTestProblem<int>(Shape, StandardInt32FunctionSet.Instance, Int32ValueSemantics.Instance,
                IntParameterStrategy.Instance,
                () => new ScalarFitness<int>(new[] { -2, -1, 0, 1, 2 }, x => x * x, (a, b) => Math.Abs((double)a - b))),
            StandardInt32FunctionSet.Instance,
            Int32ValueSemantics.Instance);
    }

    [Fact]
    public void Golden_double_array()
    {
        var inputs = new[] { new[] { -1.0, 0.5 }, new[] { 1.0, 2.0 }, new[] { 0.0, -2.0 } };
        AssertGolden(
            () => new EvolutionTestProblem<double[]>(Shape, StandardDoubleArrayFunctionSet.Instance, DoubleArrayValueSemantics.Instance,
                new DoubleArrayParameterStrategy(2),
                () => new ScalarFitness<double[]>(inputs, v => v.Select(e => e * e).ToArray(), (a, b) => a.Zip(b, (p, q) => Math.Abs(p - q)).Sum())),
            StandardDoubleArrayFunctionSet.Instance,
            DoubleArrayValueSemantics.Instance);
    }

    [Fact]
    public void Golden_int_array()
    {
        var inputs = new[] { new[] { -1, 2 }, new[] { 1, 0 }, new[] { 3, -2 } };
        AssertGolden(
            () => new EvolutionTestProblem<int[]>(Shape, StandardInt32ArrayFunctionSet.Instance, Int32ArrayValueSemantics.Instance,
                new IntArrayParameterStrategy(2),
                () => new ScalarFitness<int[]>(inputs, v => v.Select(e => e * e).ToArray(), (a, b) => a.Zip(b, (p, q) => Math.Abs((double)p - q)).Sum())),
            StandardInt32ArrayFunctionSet.Instance,
            Int32ArrayValueSemantics.Instance);
    }

    private static void AssertGolden<T>(Func<EvolutionTestProblem<T>> problemFactory, IFunctionSet<T> functions, IValueSemantics<T> values)
    {
        foreach (var seed in Seeds)
        {
            AssertOperatorsMatchLegacy(problemFactory(), functions, values, seed);
            AssertEngineStableAcrossWorkers(problemFactory, functions, values, seed);
        }
    }

    private static void AssertOperatorsMatchLegacy<T>(EvolutionTestProblem<T> problem, IFunctionSet<T> functions, IValueSemantics<T> values, int seed)
    {
        var operators = new EvolutionOperators<T>(problem);
        const int count = 8;
        var population = new Genome<T>[count];
        var fitness = new FitnessResult[count];
        for (var i = 0; i < count; i++)
        {
            var init = RandomStream.Create((ulong)seed, 0UL, (ulong)i, 1UL);
            population[i] = operators.CreateInitial(ref init);
            fitness[i] = new FitnessResult(i % 3, true);
        }

        var options = new EvolutionOptions { TournamentSize = 3, CrossoverRate = 0.5, Seed = (ulong)seed };
        var variation = new StandardVariationOperator<T>(problem);
        var context = new VariationContext<T>(population, fitness, options);
        for (var k = 0; k < 10; k++)
        {
            var legacyRng = RandomStream.Create((ulong)seed, 1UL, (ulong)k, 1UL);
            var contractRng = legacyRng;
            var legacy = operators.Breed(population, fitness, options, ref legacyRng);
            var contract = variation.CreateOffspring(context, ref contractRng);

            Assert.Equal(legacy.ComputePhenotypeFingerprint(functions, values), contract.ComputePhenotypeFingerprint(functions, values));
            Assert.Equal(legacyRng, contractRng);
        }
    }

    private static void AssertEngineStableAcrossWorkers<T>(Func<EvolutionTestProblem<T>> problemFactory, IFunctionSet<T> functions, IValueSemantics<T> values, int seed)
    {
        (List<string> Digests, EvolutionResult<T> Result) Run(int workers)
        {
            var digests = new List<string>();
            var engine = new EvolutionEngine<T>(progress => digests.Add(progress.PopulationDigest));
            var options = new EvolutionOptions
            {
                Seed = (ulong)seed,
                PopulationSize = 10,
                MaxGenerations = 3,
                TournamentSize = 3,
                CrossoverRate = 0.5,
                WorkerCount = workers,
            };
            return (digests, engine.Run(problemFactory(), options));
        }

        var (baseDigests, baseResult) = Run(WorkerCounts[0]);
        var baseFingerprint = baseResult.BestGenome.ComputePhenotypeFingerprint(functions, values);
        foreach (var workers in WorkerCounts.Skip(1))
        {
            var (digests, result) = Run(workers);
            Assert.Equal(baseDigests, digests);
            Assert.Equal(baseResult.BestFitness, result.BestFitness);
            Assert.Equal(baseResult.GenerationsRun, result.GenerationsRun);
            Assert.Equal(baseFingerprint, result.BestGenome.ComputePhenotypeFingerprint(functions, values));
        }
    }
}
