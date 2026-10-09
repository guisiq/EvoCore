using System.Text;
using System.Text.Json;

namespace Evo.Core.Persistence;

/// <summary>
/// Checkpoint JSON versionado (schema <c>evo-checkpoint/v1</c>) com escrita atômica e leitura estrita.
/// </summary>
public static class CheckpointFile
{
    /// <summary>Identificador de schema do checkpoint.</summary>
    public const string SchemaVersion = "evo-checkpoint/v1";

    /// <summary>
    /// Salva o estado de forma atômica: o conteúdo é escrito em arquivo temporário no mesmo diretório,
    /// com flush, e substitui o destino por rename. Em falha, o checkpoint anterior permanece intacto.
    /// </summary>
    public static void Save<T>(string path, IGeneticProblem<T> problem, EvolutionOptions options, EvolutionState<T> state)
    {
        ArgumentNullException.ThrowIfNull(path);
        ArgumentNullException.ThrowIfNull(problem);
        ArgumentNullException.ThrowIfNull(options);
        ArgumentNullException.ThrowIfNull(state);
        if (state.Population.Length != options.PopulationSize || state.Fitness.Length != options.PopulationSize)
        {
            throw new ArgumentException("O estado não corresponde a PopulationSize.", nameof(state));
        }

        var population = new IndividualDto[state.Population.Length];
        for (var i = 0; i < population.Length; i++)
        {
            population[i] = new IndividualDto
            {
                Genome = GenomeJson.ToDto(state.Population[i], problem),
                FitnessValue = DoubleBits.Encode(state.Fitness[i].Value),
                FitnessIsValid = state.Fitness[i].IsValid,
            };
        }

        var dto = new CheckpointDto
        {
            SchemaVersion = SchemaVersion,
            ProblemId = problem.ProblemId,
            ValueTypeId = problem.Values.ValueTypeId,
            FunctionSetId = problem.Functions.Id,
            RngVersion = RandomStream.Version,
            Shape = GenomeJson.ToShapeDto(problem.Shape),
            Options = ToOptionsDto(options),
            Generation = state.Generation,
            RootSeed = options.Seed,
            LastImprovement = state.LastImprovement,
            Population = population,
            BestGenome = state.BestGenome is null ? null : GenomeJson.ToDto(state.BestGenome, problem),
            BestFitnessValue = DoubleBits.Encode(state.BestFitness.Value),
            BestFitnessIsValid = state.BestFitness.IsValid,
        };

        var bytes = Encoding.UTF8.GetBytes(JsonSerializer.Serialize(dto, PersistenceJson.Options));
        WriteAtomically(path, bytes);
    }

    /// <summary>
    /// Lê e valida um checkpoint. Todas as verificações (schema, problema, FunctionSet, ValueTypeId, RNG,
    /// shape, opções e genomas) ocorrem antes de qualquer estado ser devolvido; em falha nada é materializado.
    /// </summary>
    public static EvolutionState<T> Load<T>(string path, IGeneticProblem<T> problem, EvolutionOptions options)
    {
        ArgumentNullException.ThrowIfNull(path);
        ArgumentNullException.ThrowIfNull(problem);
        ArgumentNullException.ThrowIfNull(options);

        CheckpointDto? dto;
        try
        {
            dto = JsonSerializer.Deserialize<CheckpointDto>(File.ReadAllText(path, Encoding.UTF8), PersistenceJson.Options);
        }
        catch (JsonException e)
        {
            throw new InvalidDataException("Checkpoint inválido: JSON malformado.", e);
        }

        if (dto is null)
        {
            throw new InvalidDataException("Checkpoint inválido: documento vazio.");
        }

        ValidateHeader(dto, problem, options);

        var n = options.PopulationSize;
        if (dto.Population is null || dto.Population.Length != n)
        {
            throw GenomeJson.Invalid("população com PopulationSize indivíduos");
        }

        if (dto.Generation < 0 || dto.LastImprovement < 0 || dto.LastImprovement > dto.Generation)
        {
            throw GenomeJson.Invalid("geração e último improvement consistentes");
        }

        var population = new Genome<T>[n];
        var fitness = new FitnessResult[n];
        for (var i = 0; i < n; i++)
        {
            var individual = dto.Population[i];
            if (individual.Genome is null)
            {
                throw GenomeJson.Invalid("genoma em cada indivíduo");
            }

            population[i] = GenomeJson.FromDto(individual.Genome, problem, options);
            fitness[i] = new FitnessResult(DoubleBits.Decode(individual.FitnessValue), individual.FitnessIsValid);
        }

        var bestGenome = dto.BestGenome is null ? null : GenomeJson.FromDto(dto.BestGenome, problem, options);
        var bestFitness = new FitnessResult(DoubleBits.Decode(dto.BestFitnessValue), dto.BestFitnessIsValid);

        return new EvolutionState<T>(dto.Generation, population, fitness, bestGenome, bestFitness, dto.LastImprovement);
    }

    private static void ValidateHeader<T>(CheckpointDto dto, IGeneticProblem<T> problem, EvolutionOptions options)
    {
        if (dto.SchemaVersion != SchemaVersion) throw GenomeJson.Invalid("schemaVersion de checkpoint suportado");
        if (dto.ProblemId != problem.ProblemId) throw GenomeJson.Invalid("ProblemId compatível");
        if (dto.ValueTypeId != problem.Values.ValueTypeId) throw GenomeJson.Invalid("ValueTypeId compatível");
        if (dto.FunctionSetId != problem.Functions.Id) throw GenomeJson.Invalid("FunctionSet.Id compatível");
        if (dto.RngVersion != RandomStream.Version) throw GenomeJson.Invalid("versão de RNG compatível");
        if (dto.Shape is null || GenomeJson.ToShape(dto.Shape) != problem.Shape) throw GenomeJson.Invalid("shape compatível");
        if (dto.RootSeed != options.Seed) throw GenomeJson.Invalid("seed raiz compatível");

        var o = dto.Options ?? throw GenomeJson.Invalid("opções de evolução");
        var targetOk = (o.TargetFitness is null && options.TargetFitness is null) ||
                       (o.TargetFitness is not null && options.TargetFitness is { } target &&
                        DoubleBits.SameBits(DoubleBits.Decode(o.TargetFitness), target));
        var compatible =
            o.PopulationSize == options.PopulationSize &&
            o.TournamentSize == options.TournamentSize &&
            o.EliteCount == options.EliteCount &&
            o.Seed == options.Seed &&
            o.StagnationGenerations == options.StagnationGenerations &&
            DoubleBits.SameBits(DoubleBits.Decode(o.CrossoverRate), options.CrossoverRate) &&
            DoubleBits.SameBits(DoubleBits.Decode(o.UniformCrossoverGeneProbability), options.UniformCrossoverGeneProbability) &&
            targetOk;
        if (!compatible)
        {
            throw GenomeJson.Invalid("opções de evolução compatíveis com o checkpoint");
        }
    }

    private static OptionsDto ToOptionsDto(EvolutionOptions options) => new()
    {
        PopulationSize = options.PopulationSize,
        MaxGenerations = options.MaxGenerations,
        TournamentSize = options.TournamentSize,
        EliteCount = options.EliteCount,
        CrossoverRate = DoubleBits.Encode(options.CrossoverRate),
        UniformCrossoverGeneProbability = DoubleBits.Encode(options.UniformCrossoverGeneProbability),
        Seed = options.Seed,
        TargetFitness = options.TargetFitness is { } target ? DoubleBits.Encode(target) : null,
        StagnationGenerations = options.StagnationGenerations,
    };

    /// <summary>
    /// Substitui o destino por rename. No Windows, o destino pode ficar bloqueado por um scanner
    /// transitório; tenta novamente por curto período antes de falhar.
    /// </summary>
    private static void MoveWithRetry(string temp, string fullPath)
    {
        for (var attempt = 1; ; attempt++)
        {
            try
            {
                File.Move(temp, fullPath, overwrite: true);
                return;
            }
            catch (UnauthorizedAccessException) when (attempt < 20)
            {
                Thread.Sleep(25 * attempt);
            }
            catch (IOException) when (attempt < 20)
            {
                Thread.Sleep(25 * attempt);
            }
        }
    }

    private static void WriteAtomically(string path, byte[] bytes)
    {
        var fullPath = Path.GetFullPath(path);
        var directory = Path.GetDirectoryName(fullPath)!;
        Directory.CreateDirectory(directory);
        var temp = Path.Combine(directory, "." + Path.GetFileName(fullPath) + "." + Guid.NewGuid().ToString("N") + ".tmp");
        try
        {
            using (var stream = new FileStream(temp, FileMode.CreateNew, FileAccess.Write, FileShare.None))
            {
                stream.Write(bytes);
                stream.Flush(true);
            }

            MoveWithRetry(temp, fullPath);
        }
        catch
        {
            if (File.Exists(temp))
            {
                File.Delete(temp);
            }

            throw;
        }
    }
}
