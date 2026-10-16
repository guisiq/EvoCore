using System.Text.Json;
using Evo.Core.Persistence;

namespace Evo.Core.Pipeline;

/// <summary>
/// Executa um <see cref="EvolutionPipelineDefinition{T}"/> sobre o engine. Cada fronteira de geração gera observação,
/// evento e checkpoint antes de a decisão ser aplicada; transições e paradas cancelam o segmento atual e recomeçam
/// (ou encerram) entre gerações. Sem definição de pipeline, o engine legado não é alterado.
/// </summary>
public sealed class EvolutionPipelineRunner<T>
{
    private const string EffectiveSchema = "evo-pipeline-effective/v1";

    private static readonly JsonSerializerOptions Json = new()
    {
        PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
        WriteIndented = true,
    };

    public EvolutionPipelineResult<T> Run(
        EvolutionPipelineDefinition<T> definition,
        string runId,
        string? evidenceDirectory = null,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(definition);
        IdRules.Validate(runId, nameof(runId));
        var effective = Describe(definition);
        var store = evidenceDirectory is null ? null : new FileEvolutionEvidenceStore(evidenceDirectory);
        store?.CreateRun(
            new EvolutionPipelineManifest(
                FileEvolutionEvidenceStore.ManifestSchema,
                definition.PipelineId,
                runId,
                effective.Sha256,
                DateTime.UtcNow.ToString("O")),
            effective.Json);
        var session = new Session(definition.PipelineId, runId, store, 0, 0);
        return Drive(definition, session, definition.InitialStateId, null, false, 0, cancellationToken);
    }

    /// <summary>
    /// Inicia uma execução comum do pipeline a partir de uma população inicial explícita e não avaliada
    /// (por exemplo, a saída de um particionamento de população). A população entra pelo caminho normal do
    /// engine: geração lógica, lineage e tempo ativo começam em zero; nenhum estado de avaliador é compartilhado
    /// com a origem da população. Fitness/digest são recomputados pelo engine antes da primeira fronteira.
    /// </summary>
    public EvolutionPipelineResult<T> RunFromInitialPopulation(
        EvolutionPipelineDefinition<T> definition,
        string runId,
        EvolutionState<T> initialState,
        string? evidenceDirectory = null,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(definition);
        ArgumentNullException.ThrowIfNull(initialState);
        IdRules.Validate(runId, nameof(runId));
        var effective = Describe(definition);
        var store = evidenceDirectory is null ? null : new FileEvolutionEvidenceStore(evidenceDirectory);
        store?.CreateRun(
            new EvolutionPipelineManifest(
                FileEvolutionEvidenceStore.ManifestSchema,
                definition.PipelineId,
                runId,
                effective.Sha256,
                DateTime.UtcNow.ToString("O")),
            effective.Json);
        var session = new Session(definition.PipelineId, runId, store, 0, 0);
        return Drive(definition, session, definition.InitialStateId, initialState, false, 0, cancellationToken);
    }

    /// <summary>
    /// Retoma uma evidência existente. Valida manifesto, identidade efetiva e checkpoint do último evento antes de continuar.
    /// Um pipeline cujo último evento é <c>Stop</c> já está concluído e é recusado.
    /// </summary>
    public EvolutionPipelineResult<T> Resume(
        EvolutionPipelineDefinition<T> definition,
        string runId,
        string evidenceDirectory,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(definition);
        IdRules.Validate(runId, nameof(runId));
        var effective = Describe(definition);
        var store = new FileEvolutionEvidenceStore(evidenceDirectory);
        var manifest = store.ReadManifest();
        if (manifest.Schema != FileEvolutionEvidenceStore.ManifestSchema
            || manifest.PipelineId != definition.PipelineId
            || manifest.RunId != runId
            || manifest.EffectiveParametersSha256 != effective.Sha256)
        {
            throw new InvalidDataException("Manifesto incompatível com pipeline, run ou parâmetros efetivos.");
        }

        if (store.ReadEffectiveParameters() != effective.Json)
        {
            throw new InvalidDataException("parameters efetivos gravados divergem da configuração atual.");
        }

        var events = store.ReadEvents();
        var transitions = events.Count(e => e.DecisionKind == nameof(EvolutionPipelineDecisionKind.Transition));
        var session = new Session(definition.PipelineId, runId, store, events.Count == 0 ? 0 : events[^1].Sequence, transitions);
        if (events.Count == 0)
        {
            return Drive(definition, session, definition.InitialStateId, null, false, 0, cancellationToken);
        }

        var last = events[^1];
        if (last.DecisionKind == nameof(EvolutionPipelineDecisionKind.Stop))
        {
            throw new InvalidOperationException("O pipeline já foi concluído nesta evidência.");
        }

        if (last.DecisionKind == nameof(EvolutionPipelineDecisionKind.Transition))
        {
            var source = definition.GetState(last.StateId);
            var sourceState = LoadCheckpoint(store, source, last);
            var target = definition.GetState(last.TargetStateId!);
            var rebased = Rebase(sourceState, last.EvaluationId, target, cancellationToken);
            return Drive(definition, session, target.StateId, rebased, false, last.Generation, cancellationToken);
        }

        var state = definition.GetState(last.StateId);
        var resume = LoadCheckpoint(store, state, last);
        return Drive(definition, session, state.StateId, resume, true, last.StateStartGeneration, cancellationToken);
    }

    private EvolutionPipelineResult<T> Drive(
        EvolutionPipelineDefinition<T> definition,
        Session session,
        string stateId,
        EvolutionState<T>? resume,
        bool alreadyRecorded,
        int stateStart,
        CancellationToken cancellationToken)
    {
        while (true)
        {
            var state = definition.GetState(stateId);
            var segment = RunSegment(definition, state, session, resume, alreadyRecorded, stateStart, cancellationToken);
            var result = segment.Result;

            if (segment.Fired is { Kind: EvolutionPipelineDecisionKind.Transition } transition)
            {
                var last = segment.Last!;
                var target = definition.GetState(transition.TargetStateId!);
                resume = Rebase(last, EvolutionPipelineIdentity.EvaluationId(state.Problem), target, cancellationToken);
                stateStart = last.Generation;
                stateId = target.StateId;
                session.Transitions++;
                alreadyRecorded = false;
                continue;
            }

            if (segment.Fired is { Kind: EvolutionPipelineDecisionKind.Stop } stop)
            {
                return new EvolutionPipelineResult<T>(result.BestGenome, result.BestFitness, result.GenerationsRun, stateId, session.Transitions,
                    EvolutionPipelineTermination.PolicyStop, result.Termination, stop.Reason!);
            }

            if (segment.NativeTermination is { } native)
            {
                return new EvolutionPipelineResult<T>(result.BestGenome, result.BestFitness, result.GenerationsRun, stateId, session.Transitions,
                    EvolutionPipelineTermination.Native, native, "terminação nativa: " + native);
            }

            return new EvolutionPipelineResult<T>(result.BestGenome, result.BestFitness, result.GenerationsRun, stateId, session.Transitions,
                EvolutionPipelineTermination.Cancellation, result.Termination, "cancelamento");
        }
    }

    private static Segment RunSegment(
        EvolutionPipelineDefinition<T> definition,
        EvolutionPipelineState<T> state,
        Session session,
        EvolutionState<T>? resume,
        bool alreadyRecorded,
        int stateStart,
        CancellationToken cancellationToken)
    {
        using var cts = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);
        var segment = new Segment();
        var digest = string.Empty;
        var skipFirstRecord = alreadyRecorded;
        var evaluationId = EvolutionPipelineIdentity.EvaluationId(state.Problem);
        var strategyId = EvolutionPipelineIdentity.VariationStrategyId(state.Variation);

        void OnCheckpoint(EvolutionState<T> s)
        {
            if (cancellationToken.IsCancellationRequested)
            {
                return;
            }

            var skipRecord = skipFirstRecord;
            skipFirstRecord = false;
            segment.Last = s;
            var sequence = skipRecord ? session.Sequence : session.Sequence + 1;
            var observation = new EvolutionPipelineObservation<T>(
                session.PipelineId,
                session.RunId,
                state.StateId,
                evaluationId,
                strategyId,
                s.Generation,
                s.Generation - stateStart,
                s.BestFitness,
                s.BestGenome?.Clone(state.Problem.Values),
                Array.AsReadOnly((FitnessResult[])s.Fitness.Clone()),
                digest,
                s.LastImprovement,
                s.ActiveElapsed,
                sequence,
                session.Transitions);

            var native = NativeTermination(state.Options, s);
            var decision = EvolutionPipelineDecision.Continue;
            if (native is null)
            {
                if (state.StopCondition is { } condition && condition(observation))
                {
                    decision = EvolutionPipelineDecision.Stop($"StopCondition '{state.StopConditionId}' satisfeita");
                }
                else if (state.Decide is { } decide)
                {
                    decision = decide(observation)
                        ?? throw new InvalidOperationException($"Estado '{state.StateId}': a decisão retornou nula.");
                }
            }

            if (decision.Kind == EvolutionPipelineDecisionKind.Transition)
            {
                if (session.Transitions >= definition.MaxTransitions)
                {
                    throw new InvalidOperationException($"Limite de transições excedido ({definition.MaxTransitions}).");
                }

                var target = definition.GetState(decision.TargetStateId!);
                EnsureCompatible(state.Problem, target.Problem);
            }

            if (!skipRecord)
            {
                session.Sequence++;
                string? checkpointSha = null;
                if (session.Store is { } store)
                {
                    store.AppendGeneration(state.StateId, s.Generation, s.Generation - stateStart, s.BestFitness, digest);
                    checkpointSha = store.SaveCheckpoint(state.StateId, path =>
                        CheckpointFile.Save(path, state.Problem, state.Options, s, strategyId));
                }

                var reason = native is { } n ? "terminação nativa: " + n : decision.Reason ?? decision.Kind.ToString();
                var kind = native is not null ? EvolutionPipelineDecisionKind.Stop : decision.Kind;
                var evt = new EvolutionPipelineEvent(
                    session.Sequence,
                    session.PipelineId,
                    session.RunId,
                    state.StateId,
                    s.Generation,
                    stateStart,
                    evaluationId,
                    strategyId,
                    digest,
                    kind.ToString(),
                    decision.TargetStateId,
                    reason,
                    native?.ToString(),
                    checkpointSha,
                    DateTime.UtcNow.ToString("O"));
                session.Store?.AppendEvent(evt);
            }

            if (native is not null)
            {
                segment.NativeTermination = native;
                return;
            }

            if (decision.Kind != EvolutionPipelineDecisionKind.Continue)
            {
                segment.Fired = decision;
                cts.Cancel();
            }
        }

        var engine = state.Variation is null
            ? new EvolutionEngine<T>(progress: p => digest = p.PopulationDigest, checkpoint: OnCheckpoint)
            : new EvolutionEngine<T>(state.Variation, progress: p => digest = p.PopulationDigest, checkpoint: OnCheckpoint);

        segment.Result = resume is null
            ? engine.Run(state.Problem, state.Options, cts.Token)
            : engine.Resume(state.Problem, state.Options, resume, cts.Token);
        return segment;
    }

    /// <summary>Terminação nativa, avaliada na mesma ordem do engine (sem a checagem de cancelamento).</summary>
    private static TerminationReason? NativeTermination(EvolutionOptions options, EvolutionState<T> s)
    {
        if (options.TargetFitness is { } target && s.BestFitness.IsValid && s.BestFitness.Value >= target)
        {
            return TerminationReason.TargetFitness;
        }

        if (options.StagnationGenerations is { } stagnation && s.Generation - s.LastImprovement >= stagnation)
        {
            return TerminationReason.Stagnation;
        }

        if (s.Generation >= options.MaxGenerations)
        {
            return TerminationReason.MaxGenerations;
        }

        if (options.TimeLimit is { } limit && s.ActiveElapsed >= limit)
        {
            return TerminationReason.TimeLimit;
        }

        return null;
    }

    private static void EnsureCompatible(IGeneticProblem<T> from, IGeneticProblem<T> to)
    {
        if (!from.Shape.Equals(to.Shape) || from.Functions.Id != to.Functions.Id || from.Values.ValueTypeId != to.Values.ValueTypeId)
        {
            throw new InvalidOperationException("Transição rejeitada: shape, function set ou ValueTypeId incompatíveis.");
        }
    }

    /// <summary>
    /// Prepara o estado da próxima execução. Se a identidade de avaliação muda, todo fitness é reavaliado e o melhor é descartado;
    /// caso contrário, população, fitness e melhor são preservados.
    /// </summary>
    private static EvolutionState<T> Rebase(EvolutionState<T> last, string fromEvaluationId, EvolutionPipelineState<T> target, CancellationToken cancellationToken)
    {
        var problem = target.Problem;
        var values = problem.Values;
        var population = last.Population.Select(g => g.Clone(values)).ToArray();
        if (EvolutionPipelineIdentity.EvaluationId(problem) == fromEvaluationId)
        {
            return new EvolutionState<T>(last.Generation, population, (FitnessResult[])last.Fitness.Clone(),
                last.BestGenome?.Clone(values), last.BestFitness, last.LastImprovement, last.ActiveElapsed, last.ElapsedKnown);
        }

        var fitnessEvaluator = problem.CreateFitnessEvaluator();
        var genomeEvaluator = new ReferenceGenomeEvaluator<T>(problem.Functions);
        var fitness = new FitnessResult[population.Length];
        for (var i = 0; i < population.Length; i++)
        {
            fitness[i] = fitnessEvaluator.Evaluate(population[i], genomeEvaluator, cancellationToken);
        }

        return new EvolutionState<T>(last.Generation, population, fitness, null, FitnessResult.Invalid(),
            last.Generation, last.ActiveElapsed, last.ElapsedKnown);
    }

    private static EvolutionState<T> LoadCheckpoint(FileEvolutionEvidenceStore store, EvolutionPipelineState<T> state, EvolutionPipelineEvent evt)
    {
        var path = store.CheckpointPath(state.StateId);
        if (evt.CheckpointSha256 is null || !File.Exists(path) || FileEvolutionEvidenceStore.Sha256File(path) != evt.CheckpointSha256)
        {
            throw new InvalidDataException($"Checkpoint do estado '{state.StateId}' ausente ou não corresponde ao último evento.");
        }

        return CheckpointFile.Load(path, state.Problem, state.Options, EvolutionPipelineIdentity.VariationStrategyId(state.Variation));
    }

    private static (string Json, string Sha256) Describe(EvolutionPipelineDefinition<T> definition)
    {
        var states = definition.States
            .OrderBy(s => s.StateId, StringComparer.Ordinal)
            .Select(s => new EffectiveState(
                s.StateId,
                s.Problem.ProblemId,
                EvolutionPipelineIdentity.EvaluationId(s.Problem),
                EvolutionPipelineIdentity.VariationStrategyId(s.Variation),
                s.StopConditionId,
                s.DecisionId,
                EffectiveOptions.From(s.Options)))
            .ToArray();
        var document = new EffectiveDocument(EffectiveSchema, definition.PipelineId, definition.InitialStateId, definition.MaxTransitions, states);
        var json = JsonSerializer.Serialize(document, Json) + "\n";
        return (json, FileEvolutionEvidenceStore.Sha256Text(json));
    }

    private sealed record EffectiveDocument(string Schema, string PipelineId, string InitialState, int MaxTransitions, EffectiveState[] States);

    private sealed record EffectiveState(
        string Id,
        string ProblemId,
        string EvaluationId,
        string? VariationStrategyId,
        string? StopCondition,
        string? Decision,
        EffectiveOptions Options);

    private sealed record EffectiveOptions(
        int PopulationSize,
        int MaxGenerations,
        int TournamentSize,
        int EliteCount,
        double CrossoverRate,
        double UniformCrossoverGeneProbability,
        ulong Seed,
        int WorkerCount,
        double? TargetFitness,
        int? StagnationGenerations,
        double? TimeLimitSeconds)
    {
        public static EffectiveOptions From(EvolutionOptions o) => new(
            o.PopulationSize, o.MaxGenerations, o.TournamentSize, o.EliteCount, o.CrossoverRate,
            o.UniformCrossoverGeneProbability, o.Seed, o.WorkerCount, o.TargetFitness, o.StagnationGenerations,
            o.TimeLimit?.TotalSeconds);
    }

    private sealed class Segment
    {
        public EvolutionResult<T> Result { get; set; } = null!;

        public EvolutionState<T>? Last { get; set; }

        public EvolutionPipelineDecision? Fired { get; set; }

        public TerminationReason? NativeTermination { get; set; }
    }

    private sealed class Session(string pipelineId, string runId, FileEvolutionEvidenceStore? store, long sequence, int transitions)
    {
        public string PipelineId { get; } = pipelineId;

        public string RunId { get; } = runId;

        public FileEvolutionEvidenceStore? Store { get; } = store;

        public long Sequence { get; set; } = sequence;

        public int Transitions { get; set; } = transitions;
    }
}
