using System.Linq;
using System.Reflection;
using Evo.Core;
using Xunit;

namespace Evo.Core.Tests;

/// <summary>
/// Smoke tests over the assembly and the frozen public contracts of
/// 01-escopo-e-api.md.
/// </summary>
public sealed class PublicContractsTests
{
    private static readonly Assembly CoreAssembly = typeof(Genome<>).Assembly;

    [Fact]
    public void CoreAssemblyIsLoadable()
    {
        Assert.Equal("Evo.Core", CoreAssembly.GetName().Name);
    }

    [Fact]
    public void CoreDoesNotReferenceCliOrExamples()
    {
        var referenced = CoreAssembly
            .GetReferencedAssemblies()
            .Select(a => a.Name)
            .ToArray();

        Assert.DoesNotContain("Evo.Cli", referenced);
        Assert.DoesNotContain("Evo.Examples", referenced);
        Assert.DoesNotContain("Evo.Benchmarks", referenced);
    }

    [Theory]
    [InlineData(typeof(IValueSemantics<>))]
    [InlineData(typeof(IParameterGeneStrategy<>))]
    [InlineData(typeof(IFunction<>))]
    [InlineData(typeof(IFunctionSet<>))]
    [InlineData(typeof(IGenomeEvaluator<>))]
    [InlineData(typeof(IFitnessEvaluator<>))]
    [InlineData(typeof(IGeneticProblem<>))]
    [InlineData(typeof(IEvolutionEngine<>))]
    public void FrozenContractIsGenericPublicInterface(Type contract)
    {
        Assert.True(contract.IsInterface);
        Assert.True(contract.IsPublic);
        Assert.Single(contract.GetGenericArguments());
    }

    [Fact]
    public void GenomeShapeCarriesFrozenMembers()
    {
        var shape = new GenomeShape(1, 100, 1, 3, 100);

        Assert.Equal(1, shape.InputCount);
        Assert.Equal(100, shape.NodeCount);
        Assert.Equal(1, shape.OutputCount);
        Assert.Equal(3, shape.MaxArity);
        Assert.Equal(100, shape.LevelsBack);
        Assert.Equal(shape, new GenomeShape(1, 100, 1, 3, 100));
    }

    [Fact]
    public void BatchFitnessEvaluatorIsNotPresentInV1()
    {
        Assert.DoesNotContain(
            CoreAssembly.GetExportedTypes(),
            t => t.Name.StartsWith("IBatchFitnessEvaluator", StringComparison.Ordinal));
    }

    [Fact]
    public void EvolutionOptionsUseNormativeDemoDefaults()
    {
        var options = new EvolutionOptions();

        Assert.Equal(500, options.PopulationSize);
        Assert.Equal(200, options.MaxGenerations);
        Assert.Equal(5, options.TournamentSize);
        Assert.Equal(1, options.EliteCount);
        Assert.Equal(0.50, options.CrossoverRate);
        Assert.Equal(0.50, options.UniformCrossoverGeneProbability);
        Assert.Equal(42UL, options.Seed);
        Assert.Equal(0, options.WorkerCount);
        Assert.Null(options.TimeLimit);
    }

    [Fact]
    public void RandomStreamVersionIsFrozen()
    {
        Assert.Equal("rng/xoshiro256ss-sha256-key/v1", RandomStream.Version);
    }

    [Fact]
    public void InvalidFitnessIsDistinguishable()
    {
        Assert.False(FitnessResult.Invalid().IsValid);
        Assert.True(FitnessResult.Valid(-0.5).IsValid);
    }
}
