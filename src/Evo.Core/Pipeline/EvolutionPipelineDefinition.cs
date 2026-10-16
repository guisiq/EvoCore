namespace Evo.Core.Pipeline;

/// <summary>
/// Estado imutável do pipeline: problema, opções nativas, variação opcional e políticas de parada/decisão.
/// Os IDs de política são persistidos na identidade efetiva; delegados nunca são serializados.
/// </summary>
public sealed record EvolutionPipelineState<T>(
    string StateId,
    IGeneticProblem<T> Problem,
    EvolutionOptions Options,
    IVariationOperator<T>? Variation = null,
    string? StopConditionId = null,
    Predicate<EvolutionPipelineObservation<T>>? StopCondition = null,
    string? DecisionId = null,
    Func<EvolutionPipelineObservation<T>, EvolutionPipelineDecision>? Decide = null);

/// <summary>Definição imutável e validada de um pipeline evolutivo.</summary>
public sealed class EvolutionPipelineDefinition<T>
{
    private readonly Dictionary<string, EvolutionPipelineState<T>> _byId = new(StringComparer.Ordinal);

    public EvolutionPipelineDefinition(string pipelineId, string initialStateId, IEnumerable<EvolutionPipelineState<T>> states, int maxTransitions = 0)
    {
        IdRules.Validate(pipelineId, nameof(pipelineId));
        IdRules.Validate(initialStateId, nameof(initialStateId));
        ArgumentNullException.ThrowIfNull(states);
        if (maxTransitions < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(maxTransitions), "MaxTransitions must be >= 0.");
        }

        foreach (var state in states)
        {
            ArgumentNullException.ThrowIfNull(state);
            IdRules.Validate(state.StateId, nameof(states));
            ArgumentNullException.ThrowIfNull(state.Problem);
            ArgumentNullException.ThrowIfNull(state.Options);
            if ((state.StopCondition is null) != (state.StopConditionId is null))
            {
                throw new ArgumentException($"Estado '{state.StateId}': StopCondition e StopConditionId devem ser fornecidos juntos.", nameof(states));
            }

            if ((state.Decide is null) != (state.DecisionId is null))
            {
                throw new ArgumentException($"Estado '{state.StateId}': Decide e DecisionId devem ser fornecidos juntos.", nameof(states));
            }

            if (state.StopConditionId is not null)
            {
                IdRules.Validate(state.StopConditionId, nameof(states));
            }

            if (state.DecisionId is not null)
            {
                IdRules.Validate(state.DecisionId, nameof(states));
            }

            if (!_byId.TryAdd(state.StateId, state))
            {
                throw new ArgumentException($"Estado duplicado: '{state.StateId}'.", nameof(states));
            }
        }

        if (!_byId.ContainsKey(initialStateId))
        {
            throw new ArgumentException($"Estado inicial inexistente: '{initialStateId}'.", nameof(initialStateId));
        }

        PipelineId = pipelineId;
        InitialStateId = initialStateId;
        MaxTransitions = maxTransitions;
    }

    public string PipelineId { get; }

    public string InitialStateId { get; }

    /// <summary>Limite global de transições, para impedir ciclos sem limite.</summary>
    public int MaxTransitions { get; }

    public EvolutionPipelineState<T> GetState(string stateId) =>
        _byId.TryGetValue(stateId, out var state)
            ? state
            : throw new InvalidOperationException($"Estado inexistente: '{stateId}'.");

    public IReadOnlyCollection<EvolutionPipelineState<T>> States => _byId.Values;
}

/// <summary>Registro explícito por ID de problemas, variações e políticas usados por arquivos <c>evo-pipeline/v1</c>.</summary>
public sealed class EvolutionPipelineRegistry<T>
{
    private readonly Dictionary<string, Func<IGeneticProblem<T>>> _problems = new(StringComparer.Ordinal);
    private readonly Dictionary<string, Func<IVariationOperator<T>>> _variations = new(StringComparer.Ordinal);
    private readonly Dictionary<string, Predicate<EvolutionPipelineObservation<T>>> _stops = new(StringComparer.Ordinal);
    private readonly Dictionary<string, Func<EvolutionPipelineObservation<T>, EvolutionPipelineDecision>> _decisions = new(StringComparer.Ordinal);

    public void RegisterProblem(string id, Func<IGeneticProblem<T>> factory) =>
        Add(_problems, id, factory, "problema");

    public void RegisterVariation(string id, Func<IVariationOperator<T>> factory) =>
        Add(_variations, id, factory, "variação");

    public void RegisterStopCondition(string id, Predicate<EvolutionPipelineObservation<T>> condition) =>
        Add(_stops, id, condition, "StopCondition");

    public void RegisterDecision(string id, Func<EvolutionPipelineObservation<T>, EvolutionPipelineDecision> decision) =>
        Add(_decisions, id, decision, "decisão");

    internal IGeneticProblem<T> ResolveProblem(string id) => Resolve(_problems, id, "problema")();

    internal IVariationOperator<T> ResolveVariation(string id) => Resolve(_variations, id, "variação")();

    internal Predicate<EvolutionPipelineObservation<T>> ResolveStopCondition(string id) => Resolve(_stops, id, "StopCondition");

    internal Func<EvolutionPipelineObservation<T>, EvolutionPipelineDecision> ResolveDecision(string id) => Resolve(_decisions, id, "decisão");

    private static void Add<TValue>(Dictionary<string, TValue> map, string id, TValue value, string kind)
    {
        IdRules.Validate(id, nameof(id));
        ArgumentNullException.ThrowIfNull(value);
        if (!map.TryAdd(id, value))
        {
            throw new ArgumentException($"{kind} já registrado: '{id}'.", nameof(id));
        }
    }

    private static TValue Resolve<TValue>(Dictionary<string, TValue> map, string id, string kind) =>
        map.TryGetValue(id, out var value)
            ? value
            : throw new InvalidDataException($"{kind} não registrado: '{id}'.");
}

/// <summary>Identidade efetiva: IDs de avaliação e estratégia, e parâmetros normalizados.</summary>
internal static class EvolutionPipelineIdentity
{
    public static string EvaluationId<T>(IGeneticProblem<T> problem) =>
        string.Join('|', problem.ProblemId, problem.Functions.Id, problem.Values.ValueTypeId);

    public static string? VariationStrategyId<T>(IVariationOperator<T>? variation) =>
        variation is null ? null : new EvolutionEngine<T>(variation).VariationStrategyId;
}
