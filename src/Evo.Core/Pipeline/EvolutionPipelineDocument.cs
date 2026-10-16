using System.Text.Json;

namespace Evo.Core.Pipeline;

/// <summary>
/// Carrega um pipeline <c>evo-pipeline/v1</c> a partir de JSON. Todo o arquivo é validado antes de a definição ser
/// construída; a carga não cria nem altera arquivos, então um erro nunca deixa efeitos parciais.
/// Campos desconhecidos e IDs não registrados são rejeitados.
/// </summary>
public static class EvolutionPipelineDocument
{
    public const string Schema = "evo-pipeline/v1";

    private static readonly string[] RootFields = ["schema", "pipelineId", "initialState", "maxTransitions", "states"];
    private static readonly string[] StateFields = ["stateId", "problem", "variation", "options", "stopCondition", "decision"];
    private static readonly string[] OptionFields =
    [
        "populationSize", "maxGenerations", "tournamentSize", "eliteCount", "crossoverRate",
        "uniformCrossoverGeneProbability", "seed", "workerCount", "targetFitness", "stagnationGenerations", "timeLimitSeconds",
    ];

    public static EvolutionPipelineDefinition<T> Load<T>(string json, EvolutionPipelineRegistry<T> registry)
    {
        ArgumentNullException.ThrowIfNull(json);
        ArgumentNullException.ThrowIfNull(registry);

        using var document = JsonDocument.Parse(json);
        var root = document.RootElement;
        RequireObject(root, "raiz");
        RejectUnknown(root, RootFields, "raiz");

        var schema = RequireString(root, "schema", "raiz");
        if (schema != Schema)
        {
            throw new InvalidDataException($"Schema inválido: '{schema}' (esperado '{Schema}').");
        }

        var pipelineId = RequireString(root, "pipelineId", "raiz");
        var initialState = RequireString(root, "initialState", "raiz");
        var maxTransitions = root.TryGetProperty("maxTransitions", out var mt) ? RequireInt(mt, "maxTransitions") : 0;
        if (!root.TryGetProperty("states", out var statesElement) || statesElement.ValueKind != JsonValueKind.Array)
        {
            throw new InvalidDataException("'states' deve ser um array.");
        }

        var states = new List<EvolutionPipelineState<T>>();
        foreach (var element in statesElement.EnumerateArray())
        {
            states.Add(LoadState(element, registry));
        }

        return new EvolutionPipelineDefinition<T>(pipelineId, initialState, states, maxTransitions);
    }

    private static EvolutionPipelineState<T> LoadState<T>(JsonElement element, EvolutionPipelineRegistry<T> registry)
    {
        RequireObject(element, "estado");
        RejectUnknown(element, StateFields, "estado");
        var stateId = RequireString(element, "stateId", "estado");
        var where = $"estado '{stateId}'";

        var problem = registry.ResolveProblem(RequireString(element, "problem", where));
        var variation = OptionalString(element, "variation", where) is { } variationId
            ? registry.ResolveVariation(variationId)
            : null;

        var options = element.TryGetProperty("options", out var optionsElement)
            ? LoadOptions(optionsElement, where)
            : new EvolutionOptions();

        var stopId = OptionalString(element, "stopCondition", where);
        var decisionId = OptionalString(element, "decision", where);

        return new EvolutionPipelineState<T>(
            stateId,
            problem,
            options,
            variation,
            stopId,
            stopId is null ? null : registry.ResolveStopCondition(stopId),
            decisionId,
            decisionId is null ? null : registry.ResolveDecision(decisionId));
    }

    private static EvolutionOptions LoadOptions(JsonElement element, string where)
    {
        RequireObject(element, $"options de {where}");
        RejectUnknown(element, OptionFields, $"options de {where}");
        var d = new EvolutionOptions();
        var values = new Dictionary<string, JsonElement>(StringComparer.Ordinal);
        foreach (var property in element.EnumerateObject())
        {
            values[property.Name] = property.Value;
        }

        int Int(string name, int fallback) => values.TryGetValue(name, out var v) ? RequireInt(v, name) : fallback;
        double Dbl(string name, double fallback) => values.TryGetValue(name, out var v) ? RequireDouble(v, name) : fallback;
        double? OptDbl(string name) => values.TryGetValue(name, out var v) ? RequireDouble(v, name) : null;
        int? OptInt(string name) => values.TryGetValue(name, out var v) ? RequireInt(v, name) : null;

        var seed = d.Seed;
        if (values.TryGetValue("seed", out var seedValue))
        {
            seed = seedValue.ValueKind == JsonValueKind.Number && seedValue.TryGetUInt64(out var s) ? s : throw Invalid("seed");
        }

        return new EvolutionOptions
        {
            PopulationSize = Int("populationSize", d.PopulationSize),
            MaxGenerations = Int("maxGenerations", d.MaxGenerations),
            TournamentSize = Int("tournamentSize", d.TournamentSize),
            EliteCount = Int("eliteCount", d.EliteCount),
            CrossoverRate = Dbl("crossoverRate", d.CrossoverRate),
            UniformCrossoverGeneProbability = Dbl("uniformCrossoverGeneProbability", d.UniformCrossoverGeneProbability),
            Seed = seed,
            WorkerCount = Int("workerCount", d.WorkerCount),
            TargetFitness = OptDbl("targetFitness") ?? d.TargetFitness,
            StagnationGenerations = OptInt("stagnationGenerations") ?? d.StagnationGenerations,
            TimeLimit = values.TryGetValue("timeLimitSeconds", out var tl)
                ? TimeSpan.FromSeconds(RequireDouble(tl, "timeLimitSeconds"))
                : d.TimeLimit,
        };
    }

    private static void RequireObject(JsonElement element, string where)
    {
        if (element.ValueKind != JsonValueKind.Object)
        {
            throw new InvalidDataException($"{where} deve ser um objeto JSON.");
        }
    }

    private static void RejectUnknown(JsonElement element, string[] allowed, string where)
    {
        foreach (var property in element.EnumerateObject())
        {
            if (Array.IndexOf(allowed, property.Name) < 0)
            {
                throw new InvalidDataException($"Campo desconhecido em {where}: '{property.Name}'.");
            }
        }
    }

    private static string RequireString(JsonElement element, string name, string where)
    {
        if (!element.TryGetProperty(name, out var value) || value.ValueKind != JsonValueKind.String)
        {
            throw new InvalidDataException($"Campo obrigatório ausente ou inválido em {where}: '{name}'.");
        }

        var text = value.GetString()!;
        if (name != "schema")
        {
            IdRules.Validate(text, name);
        }

        return text;
    }

    private static string? OptionalString(JsonElement element, string name, string where)
    {
        if (!element.TryGetProperty(name, out var value) || value.ValueKind == JsonValueKind.Null)
        {
            return null;
        }

        if (value.ValueKind != JsonValueKind.String)
        {
            throw new InvalidDataException($"Campo '{name}' em {where} deve ser string ou null.");
        }

        var text = value.GetString()!;
        IdRules.Validate(text, name);
        return text;
    }

    private static int RequireInt(JsonElement value, string name) =>
        value.ValueKind == JsonValueKind.Number && value.TryGetInt32(out var i) ? i : throw Invalid(name);

    private static double RequireDouble(JsonElement value, string name) =>
        value.ValueKind == JsonValueKind.Number && value.TryGetDouble(out var d) ? d : throw Invalid(name);

    private static InvalidDataException Invalid(string name) =>
        new($"Valor inválido para '{name}'.");
}
