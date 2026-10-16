namespace Evo.Core.Pipeline;

/// <summary>Modo de pertencimento de um indivíduo às saídas de um particionamento.</summary>
public enum PopulationPartitionMode
{
    /// <summary>Padrão: cada indivíduo da origem pertence a exatamente uma saída (cardinalidade preservada).</summary>
    DisjointExhaustive,

    /// <summary>Cada indivíduo pertence a zero ou uma saída; descartes são explícitos.</summary>
    DisjointSelected,

    /// <summary>Um indivíduo pode pertencer a mais de uma saída, somente por configuração explícita da estratégia.</summary>
    Overlapping,
}

/// <summary>
/// Snapshot imutável do genótipo de uma população-fonte: shape, value type, function set e lineage.
/// Nunca contém fitness, cache ou estado de avaliador, e nunca aceita fitness pré-calculada.
/// </summary>
public sealed class PopulationSnapshot<T>
{
    private readonly Genome<T>[] _genomes;

    private PopulationSnapshot(GenomeShape shape, string valueTypeId, string functionSetId, Genome<T>[] genomes, IReadOnlyList<string> lineage)
    {
        Shape = shape;
        ValueTypeId = valueTypeId;
        FunctionSetId = functionSetId;
        _genomes = genomes;
        Lineage = lineage;
    }

    public GenomeShape Shape { get; }

    public string ValueTypeId { get; }

    public string FunctionSetId { get; }

    /// <summary>Lineage somente leitura: identificadores de origem (execução/checkpoint) desta população.</summary>
    public IReadOnlyList<string> Lineage { get; }

    public int Count => _genomes.Length;

    /// <summary>Devolve um clone profundo e independente do genoma na posição <paramref name="index"/>.</summary>
    public Genome<T> GenomeAt(int index, IValueSemantics<T> values) => _genomes[index].Clone(values);

    /// <summary>Constrói o snapshot a partir de uma população explícita, clonando cada genoma profundamente.</summary>
    public static PopulationSnapshot<T> FromPopulation(
        IReadOnlyList<Genome<T>> genomes,
        IGeneticProblem<T> problem,
        IReadOnlyList<string>? lineage = null)
    {
        ArgumentNullException.ThrowIfNull(genomes);
        ArgumentNullException.ThrowIfNull(problem);
        if (genomes.Count == 0)
        {
            throw new ArgumentException("População de origem vazia.", nameof(genomes));
        }

        var clones = new Genome<T>[genomes.Count];
        for (var i = 0; i < genomes.Count; i++)
        {
            clones[i] = genomes[i].Clone(problem.Values);
        }

        return new PopulationSnapshot<T>(problem.Shape, problem.Values.ValueTypeId, problem.Functions.Id, clones, lineage ?? Array.Empty<string>());
    }

    /// <summary>Constrói o snapshot a partir de um <see cref="EvolutionState{T}"/> validado (checkpoint em memória).</summary>
    public static PopulationSnapshot<T> FromState(EvolutionState<T> state, IGeneticProblem<T> problem, IReadOnlyList<string>? lineage = null)
    {
        ArgumentNullException.ThrowIfNull(state);
        return FromPopulation(state.Population, problem, lineage);
    }

    /// <summary>Constrói o snapshot a partir de um arquivo de checkpoint, carregando apenas o genótipo.</summary>
    public static PopulationSnapshot<T> FromCheckpoint(
        string path,
        IGeneticProblem<T> problem,
        EvolutionOptions options,
        string? variationStrategyId = null,
        IReadOnlyList<string>? lineage = null)
    {
        var state = Persistence.CheckpointFile.Load(path, problem, options, variationStrategyId);
        return FromState(state, problem, lineage);
    }
}

/// <summary>Descrição de uma saída declarada: ID estável, seed explícita e diretório de evidência próprios.</summary>
public sealed class PopulationOutputSpec
{
    public PopulationOutputSpec(string outputId, ulong seed, string evidenceDirectory)
    {
        IdRules.Validate(outputId, nameof(outputId));
        ArgumentException.ThrowIfNullOrWhiteSpace(evidenceDirectory);
        OutputId = outputId;
        Seed = seed;
        EvidenceDirectory = evidenceDirectory;
    }

    public string OutputId { get; }

    public ulong Seed { get; }

    public string EvidenceDirectory { get; }
}

/// <summary>Pedido de particionamento: snapshot imutável da origem, saídas ordenadas, modo de pertencimento e seed.</summary>
public sealed class PopulationPartitionRequest<T>
{
    public PopulationPartitionRequest(
        PopulationSnapshot<T> source,
        IReadOnlyList<PopulationOutputSpec> outputs,
        PopulationPartitionMode mode,
        ulong seed)
    {
        ArgumentNullException.ThrowIfNull(source);
        ArgumentNullException.ThrowIfNull(outputs);
        if (outputs.Count == 0)
        {
            throw new ArgumentException("É necessário declarar ao menos uma saída.", nameof(outputs));
        }

        var ids = new HashSet<string>(StringComparer.Ordinal);
        foreach (var output in outputs)
        {
            ArgumentNullException.ThrowIfNull(output);
            if (!ids.Add(output.OutputId))
            {
                throw new ArgumentException($"OutputId duplicado: '{output.OutputId}'.", nameof(outputs));
            }
        }

        Source = source;
        Outputs = outputs;
        Mode = mode;
        Seed = seed;
    }

    public PopulationSnapshot<T> Source { get; }

    public IReadOnlyList<PopulationOutputSpec> Outputs { get; }

    public PopulationPartitionMode Mode { get; }

    public ulong Seed { get; }
}

/// <summary>Atribuição produzida para uma saída: índices (0-based) da origem, na ordem de atribuição.</summary>
public sealed record PopulationPartition<T>(PopulationOutputSpec Output, IReadOnlyList<int> SourceIndices);

/// <summary>
/// Estratégia de particionamento registrada, com identidade estável (<see cref="Id"/>) e configuração própria.
/// Políticas puras configuradas por código podem, em vez disso, usar
/// <c>Func&lt;PopulationPartitionRequest&lt;T&gt;, IReadOnlyList&lt;PopulationPartition&lt;T&gt;&gt;&gt;</c>.
/// </summary>
public interface IPopulationPartitioner<T>
{
    string Id { get; }

    IReadOnlyList<PopulationPartition<T>> Partition(PopulationPartitionRequest<T> request);
}

/// <summary>
/// Validação comum dos três modos normativos de pertencimento, independente da estratégia concreta.
/// Toda implementação de <see cref="IPopulationPartitioner{T}"/> (ou política <c>Func</c>) deve chamar
/// <see cref="Validate{T}"/> antes de devolver as partições ao chamador.
/// </summary>
public static class PopulationPartitionValidation
{
    public static void Validate<T>(PopulationPartitionRequest<T> request, IReadOnlyList<PopulationPartition<T>> partitions)
    {
        ArgumentNullException.ThrowIfNull(request);
        ArgumentNullException.ThrowIfNull(partitions);
        if (partitions.Count == 0)
        {
            throw new InvalidOperationException("Particionamento não produziu nenhuma saída.");
        }

        if (partitions.Count != request.Outputs.Count)
        {
            throw new InvalidOperationException("Número de partições diverge do número de saídas declaradas.");
        }

        var seenIds = new HashSet<string>(StringComparer.Ordinal);
        var membership = new int[request.Source.Count];
        for (var p = 0; p < partitions.Count; p++)
        {
            var partition = partitions[p] ?? throw new InvalidOperationException("Partição nula.");
            if (partition.Output.OutputId != request.Outputs[p].OutputId)
            {
                throw new InvalidOperationException("Ordem/identidade de saída da partição diverge das saídas declaradas.");
            }

            if (!seenIds.Add(partition.Output.OutputId))
            {
                throw new InvalidOperationException($"OutputId duplicado: '{partition.Output.OutputId}'.");
            }

            if (request.Mode != PopulationPartitionMode.DisjointSelected && (partition.SourceIndices is null || partition.SourceIndices.Count == 0))
            {
                throw new InvalidOperationException($"Saída '{partition.Output.OutputId}' não recebeu indivíduos.");
            }

            var seenInPartition = new HashSet<int>();
            foreach (var index in partition.SourceIndices ?? Array.Empty<int>())
            {
                if (index < 0 || index >= request.Source.Count)
                {
                    throw new InvalidOperationException($"Índice fora de faixa na saída '{partition.Output.OutputId}': {index}.");
                }

                if (!seenInPartition.Add(index))
                {
                    throw new InvalidOperationException($"Saída '{partition.Output.OutputId}' atribui o índice {index} mais de uma vez.");
                }

                membership[index]++;
            }
        }

        switch (request.Mode)
        {
            case PopulationPartitionMode.DisjointExhaustive:
                for (var i = 0; i < membership.Length; i++)
                {
                    if (membership[i] != 1)
                    {
                        throw new InvalidOperationException(
                            $"DisjointExhaustive exige exatamente uma saída por indivíduo; índice {i} tem {membership[i]}.");
                    }
                }

                break;
            case PopulationPartitionMode.DisjointSelected:
                for (var i = 0; i < membership.Length; i++)
                {
                    if (membership[i] > 1)
                    {
                        throw new InvalidOperationException(
                            $"DisjointSelected exige no máximo uma saída por indivíduo; índice {i} tem {membership[i]}.");
                    }
                }

                break;
            case PopulationPartitionMode.Overlapping:
                // Múltipla atribuição é aceita somente porque foi explicitamente configurada pela estratégia.
                break;
            default:
                throw new ArgumentOutOfRangeException(nameof(request), request.Mode, "Modo de particionamento desconhecido.");
        }
    }
}

/// <summary>
/// Adapta uma política pura <c>Func&lt;PopulationPartitionRequest&lt;T&gt;, IReadOnlyList&lt;PopulationPartition&lt;T&gt;&gt;&gt;</c>
/// configurada por código a <see cref="IPopulationPartitioner{T}"/>, validando o resultado antes de devolvê-lo.
/// </summary>
public sealed class DelegatePopulationPartitioner<T> : IPopulationPartitioner<T>
{
    private readonly Func<PopulationPartitionRequest<T>, IReadOnlyList<PopulationPartition<T>>> _policy;

    public DelegatePopulationPartitioner(string id, Func<PopulationPartitionRequest<T>, IReadOnlyList<PopulationPartition<T>>> policy)
    {
        IdRules.Validate(id, nameof(id));
        _policy = policy ?? throw new ArgumentNullException(nameof(policy));
        Id = id;
    }

    public string Id { get; }

    public IReadOnlyList<PopulationPartition<T>> Partition(PopulationPartitionRequest<T> request)
    {
        ArgumentNullException.ThrowIfNull(request);
        var partitions = _policy(request);
        PopulationPartitionValidation.Validate(request, partitions);
        return partitions;
    }
}

/// <summary>
/// Particionador determinístico por blocos contíguos de índice estável: distribui a origem em partes quase iguais,
/// na ordem das saídas declaradas. É a estratégia de referência para o modo padrão <see cref="PopulationPartitionMode.DisjointExhaustive"/>.
/// </summary>
public sealed class ContiguousBlockPartitioner<T> : IPopulationPartitioner<T>
{
    public const string PartitionerId = "evo.partition.contiguous-block/v1";

    public string Id => PartitionerId;

    public IReadOnlyList<PopulationPartition<T>> Partition(PopulationPartitionRequest<T> request)
    {
        ArgumentNullException.ThrowIfNull(request);
        if (request.Mode != PopulationPartitionMode.DisjointExhaustive)
        {
            throw new InvalidOperationException($"{PartitionerId} só suporta DisjointExhaustive.");
        }

        var outputs = request.Outputs;
        var total = request.Source.Count;
        if (outputs.Count > total)
        {
            throw new InvalidOperationException("Mais saídas declaradas do que indivíduos na origem.");
        }

        var baseSize = total / outputs.Count;
        var remainder = total % outputs.Count;
        var partitions = new PopulationPartition<T>[outputs.Count];
        var cursor = 0;
        for (var i = 0; i < outputs.Count; i++)
        {
            var size = baseSize + (i < remainder ? 1 : 0);
            var indices = new int[size];
            for (var j = 0; j < size; j++)
            {
                indices[j] = cursor + j;
            }

            cursor += size;
            partitions[i] = new PopulationPartition<T>(outputs[i], indices);
        }

        IReadOnlyList<PopulationPartition<T>> result = Array.AsReadOnly(partitions);
        PopulationPartitionValidation.Validate(request, result);
        return result;
    }
}

/// <summary>Resultado individual de uma saída de particionamento.</summary>
public enum PopulationPartitionOutcome
{
    Success,
    Failure,
    Cancelled,
}

/// <summary>Resultado agregado de uma saída: sucesso com o resultado do pipeline, falha com mensagem, ou cancelamento.</summary>
public sealed record PopulationPartitionRunResult<T>(
    string OutputId,
    PopulationPartitionOutcome Outcome,
    EvolutionPipelineResult<T>? Result,
    string? Error);

/// <summary>
/// Executa e agrega as saídas de um particionamento. Cada saída é uma execução de pipeline comum e isolada:
/// evaluator, cache, callback, evidence store e diretório próprios, sem estado mutável compartilhado com as
/// demais. A agregação preserva a ordem declarada em <see cref="PopulationPartitionRequest{T}.Outputs"/>,
/// independentemente da ordem de conclusão; falha ou cancelamento de uma saída não contamina as demais.
/// </summary>
public sealed class PopulationPartitionRunner<T>
{
    /// <param name="request">Pedido validado de particionamento.</param>
    /// <param name="partitioner">Estratégia registrada responsável por produzir as partições.</param>
    /// <param name="definitionFactory">
    /// Constrói a definição de pipeline de uma saída a partir de sua descrição e do tamanho de população atribuído;
    /// o chamador é responsável por usar esse tamanho em <see cref="EvolutionOptions.PopulationSize"/>.
    /// </param>
    /// <param name="maxDegreeOfParallelism">1 executa sequencialmente; &gt;1 executa saídas concorrentemente.</param>
    public IReadOnlyList<PopulationPartitionRunResult<T>> Run(
        PopulationPartitionRequest<T> request,
        IPopulationPartitioner<T> partitioner,
        Func<PopulationOutputSpec, int, EvolutionPipelineDefinition<T>> definitionFactory,
        int maxDegreeOfParallelism = 1,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(request);
        ArgumentNullException.ThrowIfNull(partitioner);
        ArgumentNullException.ThrowIfNull(definitionFactory);

        var partitions = partitioner.Partition(request);
        var byOutputId = partitions.ToDictionary(p => p.Output.OutputId, StringComparer.Ordinal);
        var ordered = request.Outputs.Select(spec => byOutputId[spec.OutputId]).ToArray();

        var results = new PopulationPartitionRunResult<T>[ordered.Length];

        void RunOne(int i)
        {
            var partition = ordered[i];
            var spec = partition.Output;
            try
            {
                var definition = definitionFactory(spec, partition.SourceIndices.Count);
                var state = definition.GetState(definition.InitialStateId);
                ValidateCompatibility(request.Source, state.Problem, spec.OutputId);
                if (state.Options.PopulationSize != partition.SourceIndices.Count)
                {
                    throw new InvalidOperationException(
                        $"Saída '{spec.OutputId}': PopulationSize ({state.Options.PopulationSize}) diverge da cardinalidade atribuída ({partition.SourceIndices.Count}).");
                }

                var initial = BuildInitialState(request.Source, partition, state.Problem, cancellationToken);
                var runner = new EvolutionPipelineRunner<T>();
                var result = runner.RunFromInitialPopulation(definition, spec.OutputId, initial, spec.EvidenceDirectory, cancellationToken);
                results[i] = new PopulationPartitionRunResult<T>(spec.OutputId, PopulationPartitionOutcome.Success, result, null);
            }
            catch (OperationCanceledException)
            {
                results[i] = new PopulationPartitionRunResult<T>(spec.OutputId, PopulationPartitionOutcome.Cancelled, null, "cancelado");
            }
            catch (Exception ex)
            {
                results[i] = new PopulationPartitionRunResult<T>(spec.OutputId, PopulationPartitionOutcome.Failure, null, ex.Message);
            }
        }

        if (maxDegreeOfParallelism <= 1)
        {
            for (var i = 0; i < ordered.Length; i++)
            {
                RunOne(i);
            }
        }
        else
        {
            Parallel.For(0, ordered.Length, new ParallelOptions { MaxDegreeOfParallelism = maxDegreeOfParallelism }, RunOne);
        }

        return Array.AsReadOnly(results);
    }

    private static void ValidateCompatibility(PopulationSnapshot<T> source, IGeneticProblem<T> problem, string outputId)
    {
        if (!problem.Shape.Equals(source.Shape) || problem.Values.ValueTypeId != source.ValueTypeId || problem.Functions.Id != source.FunctionSetId)
        {
            throw new InvalidOperationException($"Saída '{outputId}': shape, ValueTypeId ou function set incompatíveis com a origem.");
        }
    }

    /// <summary>
    /// Constrói a população inicial não avaliada da saída: clone profundo próprio de cada genoma atribuído, seguido
    /// de avaliação de fitness pelo caminho normal do problema (mesma rotina usada pelo engine ao rebasear estado
    /// entre identidades de avaliação distintas). Geração lógica e tempo ativo começam em zero.
    /// </summary>
    private static EvolutionState<T> BuildInitialState(
        PopulationSnapshot<T> source,
        PopulationPartition<T> partition,
        IGeneticProblem<T> problem,
        CancellationToken cancellationToken)
    {
        var values = problem.Values;
        var population = new Genome<T>[partition.SourceIndices.Count];
        for (var i = 0; i < population.Length; i++)
        {
            population[i] = source.GenomeAt(partition.SourceIndices[i], values);
        }

        var fitnessEvaluator = problem.CreateFitnessEvaluator();
        var genomeEvaluator = new ReferenceGenomeEvaluator<T>(problem.Functions);
        var fitness = new FitnessResult[population.Length];
        for (var i = 0; i < population.Length; i++)
        {
            fitness[i] = fitnessEvaluator.Evaluate(population[i], genomeEvaluator, cancellationToken);
        }

        return new EvolutionState<T>(0, population, fitness, null, FitnessResult.Invalid(), 0);
    }
}
