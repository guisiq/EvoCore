using System.Text.Json;
using System.Text.Json.Nodes;

namespace Evo.Core.Persistence;

/// <summary>Serialização JSON versionada de um genoma (schema <c>evo-genome/v1</c>).</summary>
public static class GenomeJson
{
    /// <summary>Identificador de schema do genoma.</summary>
    public const string SchemaVersion = "evo-genome/v1";

    /// <summary>Gera o JSON do genoma, incluindo metadados de validação e a fingerprint fenotípica.</summary>
    public static string Serialize<T>(Genome<T> genome, IGeneticProblem<T> problem)
    {
        ArgumentNullException.ThrowIfNull(genome);
        ArgumentNullException.ThrowIfNull(problem);
        return JsonSerializer.Serialize(ToDto(genome, problem), PersistenceJson.Options);
    }

    /// <summary>Lê e valida um genoma; lança <see cref="InvalidDataException"/> antes de materializar estado inválido.</summary>
    public static Genome<T> Deserialize<T>(string json, IGeneticProblem<T> problem, EvolutionOptions options)
    {
        ArgumentNullException.ThrowIfNull(json);
        ArgumentNullException.ThrowIfNull(problem);
        ArgumentNullException.ThrowIfNull(options);
        GenomeDto? dto;
        try
        {
            dto = JsonSerializer.Deserialize<GenomeDto>(json, PersistenceJson.Options);
        }
        catch (JsonException e)
        {
            throw new InvalidDataException("Checkpoint/genoma inválido: JSON malformado.", e);
        }

        if (dto is null)
        {
            throw new InvalidDataException("Checkpoint/genoma inválido: documento vazio.");
        }

        return FromDto(dto, problem, options);
    }

    internal static GenomeDto ToDto<T>(Genome<T> genome, IGeneticProblem<T> problem)
    {
        var codec = ValueCodecs.Resolve(problem.Values);
        var parameters = new JsonArray();
        foreach (var value in genome.ParameterGenes)
        {
            parameters.Add(codec.ToJson(value));
        }

        return new GenomeDto
        {
            SchemaVersion = SchemaVersion,
            ValueTypeId = problem.Values.ValueTypeId,
            FunctionSetId = problem.Functions.Id,
            Shape = ToShapeDto(genome.Shape),
            FunctionGenes = genome.FunctionGenes.ToArray(),
            ConnectionGenes = genome.ConnectionGenes.ToArray(),
            ParameterGenes = parameters,
            OutputGenes = genome.OutputGenes.ToArray(),
            PhenotypeFingerprint = Convert.ToHexString(genome.ComputePhenotypeFingerprint(problem.Functions, problem.Values)),
        };
    }

    internal static Genome<T> FromDto<T>(GenomeDto dto, IGeneticProblem<T> problem, EvolutionOptions options)
    {
        var values = problem.Values;
        var functions = problem.Functions;
        var shape = problem.Shape;

        if (dto.SchemaVersion != SchemaVersion) throw Invalid("schemaVersion de genoma suportado");
        if (dto.ValueTypeId != values.ValueTypeId) throw Invalid("ValueTypeId compatível com o problema");
        if (dto.FunctionSetId != functions.Id) throw Invalid("FunctionSet.Id compatível com o problema");
        if (dto.Shape is null || ToShape(dto.Shape) != shape) throw Invalid("shape compatível com o problema");

        var codec = ValueCodecs.Resolve(values);
        var functionGenes = RequireArray(dto.FunctionGenes, shape.NodeCount, "functionGenes");
        var connectionGenes = RequireArray(dto.ConnectionGenes, shape.NodeCount * shape.MaxArity, "connectionGenes");
        var outputGenes = RequireArray(dto.OutputGenes, shape.OutputCount, "outputGenes");
        if (dto.ParameterGenes is not JsonArray parameterArray || parameterArray.Count != shape.NodeCount)
        {
            throw Invalid("parameterGenes com o número de nós do shape");
        }

        var parameterGenes = new T[shape.NodeCount];
        for (var i = 0; i < parameterGenes.Length; i++)
        {
            parameterGenes[i] = codec.FromJson(parameterArray[i]);
        }

        Genome<T> genome;
        try
        {
            genome = Genome<T>.FromGenes(shape, functionGenes, connectionGenes, parameterGenes, outputGenes, values, functions);
        }
        catch (Exception e) when (e is InvalidGenomeException or ArgumentException)
        {
            throw new InvalidDataException("Checkpoint/genoma inválido: genoma rejeitado pela validação.", e);
        }

        var fingerprint = Convert.ToHexString(genome.ComputePhenotypeFingerprint(functions, values));
        if (!string.Equals(fingerprint, dto.PhenotypeFingerprint, StringComparison.OrdinalIgnoreCase))
        {
            throw Invalid("PhenotypeFingerprint consistente com os genes");
        }

        return genome;
    }

    internal static ShapeDto ToShapeDto(GenomeShape shape) => new()
    {
        InputCount = shape.InputCount,
        NodeCount = shape.NodeCount,
        OutputCount = shape.OutputCount,
        MaxArity = shape.MaxArity,
        LevelsBack = shape.LevelsBack,
    };

    internal static GenomeShape ToShape(ShapeDto dto) =>
        new(dto.InputCount, dto.NodeCount, dto.OutputCount, dto.MaxArity, dto.LevelsBack);

    internal static InvalidDataException Invalid(string expected) =>
        new($"Checkpoint/genoma inválido: esperado {expected}.");

    private static int[] RequireArray(int[]? array, int expectedLength, string name) =>
        array is not null && array.Length == expectedLength ? array : throw Invalid($"{name} com {expectedLength} elementos");
}
