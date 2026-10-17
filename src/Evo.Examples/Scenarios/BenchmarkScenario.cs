namespace Evo.Examples.Scenarios;

/// <summary>Resultado padronizado de um cenário: gates (corretude) separados de medições (tempo/alocação).</summary>
public sealed record BenchmarkScenarioResult(
    string ScenarioId,
    bool GatesPassed,
    IReadOnlyDictionary<string, string> Gates,
    IReadOnlyDictionary<string, string> Measurements,
    IReadOnlyDictionary<string, string> RawData,
    IReadOnlyDictionary<string, string> Parameters,
    string Summary);

/// <summary>Contrato público de um cenário de benchmark/exemplo registrado por ID estável.</summary>
public interface IBenchmarkScenario
{
    /// <summary>ID estável e único do cenário, conforme a matriz normativa de EVO-305.</summary>
    string Id { get; }

    /// <summary>Descrição curta do cenário para listagem via CLI.</summary>
    string Description { get; }

    /// <summary>Executa o cenário e devolve gates, medições e dados brutos.</summary>
    BenchmarkScenarioResult Run(CancellationToken cancellationToken);
}

/// <summary>Registro explícito e estável dos cenários disponíveis, por ID.</summary>
public static class BenchmarkScenarioRegistry
{
    private static readonly Dictionary<string, IBenchmarkScenario> ById = BuildRegistry();

    /// <summary>Todos os cenários registrados, na ordem de registro.</summary>
    public static IReadOnlyList<IBenchmarkScenario> All { get; } = [.. ById.Values];

    public static IBenchmarkScenario Resolve(string id) =>
        ById.TryGetValue(id, out var scenario)
            ? scenario
            : throw new KeyNotFoundException($"cenário não registrado: '{id}'.");

    private static Dictionary<string, IBenchmarkScenario> BuildRegistry()
    {
        var scenarios = new IBenchmarkScenario[]
        {
            new VariationStrategiesBostonScenario(),
            new InitializationStrategiesDiabetesScenario(),
            new PopulationScaleWineScenario(),
            new FunctionSetComplexityBostonScenario(),
            new MaximumConfigurationWineScenario(),
            new SeedRepeatabilityDiabetesScenario(),
            new EvaluatorThroughputSyntheticScenario(),
            new EvaluatorParityMatrixScenario(),
            new BinaryOperatorParityScenario(),
            new ParameterizedOperatorParityScenario(),
            new DeepChainParityScenario(),
            new BalancedParameterParityScenario(),
            new ChainDagBenchmarkScenario(),
            new FanOutDagBenchmarkScenario(),
            new DiamondDagBenchmarkScenario(),
            new LevelsDiamondDagBenchmarkScenario(),
            new IlgpuDagBenchmarkScenario(),
        };

        var map = new Dictionary<string, IBenchmarkScenario>(StringComparer.Ordinal);
        foreach (var scenario in scenarios)
        {
            if (!map.TryAdd(scenario.Id, scenario))
            {
                throw new InvalidOperationException($"cenário duplicado: '{scenario.Id}'.");
            }
        }

        return map;
    }
}
