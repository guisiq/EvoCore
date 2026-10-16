namespace Evo.Core.Pipeline;

/// <summary>Tipo de decisão tomada em uma fronteira de geração.</summary>
public enum EvolutionPipelineDecisionKind
{
    /// <summary>Continua a execução do estado atual.</summary>
    Continue,

    /// <summary>Transiciona para outro estado registrado, entre gerações.</summary>
    Transition,

    /// <summary>Encerra o pipeline com razão persistida.</summary>
    Stop,
}

/// <summary>Decisão imutável produzida pela política do estado. A razão é obrigatória para transição e parada.</summary>
public sealed class EvolutionPipelineDecision
{
    private EvolutionPipelineDecision(EvolutionPipelineDecisionKind kind, string? targetStateId, string? reason)
    {
        Kind = kind;
        TargetStateId = targetStateId;
        Reason = reason;
    }

    public EvolutionPipelineDecisionKind Kind { get; }

    public string? TargetStateId { get; }

    public string? Reason { get; }

    public static EvolutionPipelineDecision Continue { get; } = new(EvolutionPipelineDecisionKind.Continue, null, null);

    public static EvolutionPipelineDecision Transition(string targetStateId, string reason)
    {
        IdRules.Validate(targetStateId, nameof(targetStateId));
        ValidateReason(reason);
        return new(EvolutionPipelineDecisionKind.Transition, targetStateId, reason);
    }

    public static EvolutionPipelineDecision Stop(string reason)
    {
        ValidateReason(reason);
        return new(EvolutionPipelineDecisionKind.Stop, null, reason);
    }

    private static void ValidateReason(string reason)
    {
        if (string.IsNullOrWhiteSpace(reason))
        {
            throw new ArgumentException("A razão da decisão é obrigatória.", nameof(reason));
        }
    }
}

/// <summary>Regras de identificadores seguros para arquivos e JSON.</summary>
internal static class IdRules
{
    public static bool IsSafe(string value)
    {
        if (value.Length == 0)
        {
            return false;
        }

        foreach (var c in value)
        {
            if (!(char.IsAsciiLetterOrDigit(c) || c is '_' or '-' or '.'))
            {
                return false;
            }
        }

        return true;
    }

    public static void Validate(string value, string name)
    {
        if (value is null || !IsSafe(value))
        {
            throw new ArgumentException("Identificador inválido (use letras ASCII, dígitos, '_', '-' ou '.').", name);
        }
    }
}

/// <summary>
/// Observação somente leitura de uma fronteira de geração. O melhor genoma é uma cópia e a lista de fitness é imutável.
/// </summary>
public sealed record EvolutionPipelineObservation<T>(
    string PipelineId,
    string RunId,
    string StateId,
    string EvaluationId,
    string? VariationStrategyId,
    int Generation,
    int StateGeneration,
    FitnessResult BestFitness,
    Genome<T>? BestGenome,
    IReadOnlyList<FitnessResult> Fitness,
    string PopulationDigest,
    int LastImprovement,
    TimeSpan ActiveElapsed,
    long EventSequence,
    int Transitions);

/// <summary>Evento persistido em <c>events.jsonl</c>, com sequência monotônica.</summary>
public sealed record EvolutionPipelineEvent(
    long Sequence,
    string PipelineId,
    string RunId,
    string StateId,
    int Generation,
    int StateStartGeneration,
    string EvaluationId,
    string? VariationStrategyId,
    string PopulationDigest,
    string DecisionKind,
    string? TargetStateId,
    string Reason,
    string? NativeTermination,
    string? CheckpointSha256,
    string TimestampUtc);

/// <summary>Como o pipeline terminou.</summary>
public enum EvolutionPipelineTermination
{
    /// <summary>Terminação nativa do engine (alvo, estagnação, gerações, tempo).</summary>
    Native,

    /// <summary>Parada por <c>StopCondition</c> ou decisão de política.</summary>
    PolicyStop,

    /// <summary>Cancelamento externo.</summary>
    Cancellation,
}

/// <summary>Resultado do pipeline.</summary>
public sealed record EvolutionPipelineResult<T>(
    Genome<T> BestGenome,
    FitnessResult BestFitness,
    int GenerationsRun,
    string FinalStateId,
    int Transitions,
    EvolutionPipelineTermination Termination,
    TerminationReason? NativeTermination,
    string Reason);
