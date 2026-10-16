using System.Globalization;
using System.Text.Json;
using System.Text.Json.Nodes;

namespace Evo.Core.Persistence;

internal static class PersistenceJson
{
    internal static readonly JsonSerializerOptions Options = new()
    {
        WriteIndented = true,
        PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
    };
}

/// <summary>Codificação exata de <see cref="double"/> como bit pattern hexadecimal.</summary>
internal static class DoubleBits
{
    public static string Encode(double value) =>
        "0x" + BitConverter.DoubleToInt64Bits(value).ToString("x16", CultureInfo.InvariantCulture);

    public static double Decode(string? text)
    {
        if (text is null || text.Length != 18 || !text.StartsWith("0x", StringComparison.Ordinal) ||
            !ulong.TryParse(text.AsSpan(2), NumberStyles.AllowHexSpecifier, CultureInfo.InvariantCulture, out var bits))
        {
            throw new InvalidDataException("Checkpoint/genoma inválido: double mal codificado.");
        }

        return BitConverter.Int64BitsToDouble((long)bits);
    }

    public static double DecodeFinite(string? text)
    {
        var value = Decode(text);
        if (!double.IsFinite(value))
        {
            throw new InvalidDataException("Checkpoint/genoma inválido: gene double não finito.");
        }

        return value;
    }

    public static bool SameBits(double left, double right) =>
        BitConverter.DoubleToInt64Bits(left) == BitConverter.DoubleToInt64Bits(right);
}

/// <summary>Codec JSON de um valor de domínio, um por ValueTypeId oficial.</summary>
internal interface IValueCodec<T>
{
    string ValueTypeId { get; }

    JsonNode ToJson(T value);

    T FromJson(JsonNode? node);
}

internal static class ValueCodecs
{
    internal static IValueCodec<T> Resolve<T>(IValueSemantics<T> values)
    {
        object? codec = typeof(T) == typeof(double) ? new DoubleCodec()
            : typeof(T) == typeof(int) ? new Int32Codec()
            : typeof(T) == typeof(double[]) ? new DoubleArrayCodec()
            : typeof(T) == typeof(int[]) ? new Int32ArrayCodec()
            : null;

        if (codec is not IValueCodec<T> typed || typed.ValueTypeId != values.ValueTypeId)
        {
            throw new NotSupportedException($"Sem codec de persistência para o ValueTypeId '{values.ValueTypeId}'.");
        }

        return typed;
    }

    private static InvalidDataException Expected(string what) =>
        new($"Checkpoint/genoma inválido: esperado {what}.");

    private sealed class DoubleCodec : IValueCodec<double>
    {
        public string ValueTypeId => "double";

        public JsonNode ToJson(double value)
        {
            if (!double.IsFinite(value))
            {
                throw new ArgumentException("Parâmetro double não finito não pode ser persistido.");
            }

            return JsonValue.Create(DoubleBits.Encode(value))!;
        }

        public double FromJson(JsonNode? node) =>
            DoubleBits.DecodeFinite(node is JsonValue v && v.TryGetValue(out string? s) ? s : null);
    }

    private sealed class Int32Codec : IValueCodec<int>
    {
        public string ValueTypeId => "int32";

        public JsonNode ToJson(int value) => JsonValue.Create(value)!;

        public int FromJson(JsonNode? node) =>
            node is JsonValue v && v.TryGetValue(out int x) ? x : throw Expected("int32");
    }

    private sealed class DoubleArrayCodec : IValueCodec<double[]>
    {
        public string ValueTypeId => "double[]";

        public JsonNode ToJson(double[] value)
        {
            var codec = new DoubleCodec();
            var array = new JsonArray();
            foreach (var item in value)
            {
                array.Add(codec.ToJson(item));
            }

            return array;
        }

        public double[] FromJson(JsonNode? node)
        {
            if (node is not JsonArray array)
            {
                throw Expected("array de double");
            }

            var codec = new DoubleCodec();
            var result = new double[array.Count];
            for (var i = 0; i < result.Length; i++)
            {
                result[i] = codec.FromJson(array[i]);
            }

            return result;
        }
    }

    private sealed class Int32ArrayCodec : IValueCodec<int[]>
    {
        public string ValueTypeId => "int32[]";

        public JsonNode ToJson(int[] value)
        {
            var array = new JsonArray();
            foreach (var item in value)
            {
                array.Add(JsonValue.Create(item));
            }

            return array;
        }

        public int[] FromJson(JsonNode? node)
        {
            if (node is not JsonArray array)
            {
                throw Expected("array de int32");
            }

            var codec = new Int32Codec();
            var result = new int[array.Count];
            for (var i = 0; i < result.Length; i++)
            {
                result[i] = codec.FromJson(array[i]);
            }

            return result;
        }
    }
}

internal sealed class ShapeDto
{
    public int InputCount { get; set; }
    public int NodeCount { get; set; }
    public int OutputCount { get; set; }
    public int MaxArity { get; set; }
    public int LevelsBack { get; set; }
}

internal sealed class GenomeDto
{
    public string? SchemaVersion { get; set; }
    public string? ValueTypeId { get; set; }
    public string? FunctionSetId { get; set; }
    public ShapeDto? Shape { get; set; }
    public int[]? FunctionGenes { get; set; }
    public int[]? ConnectionGenes { get; set; }
    public JsonNode? ParameterGenes { get; set; }
    public int[]? OutputGenes { get; set; }
    public string? PhenotypeFingerprint { get; set; }
}

internal sealed class OptionsDto
{
    public int PopulationSize { get; set; }
    public int MaxGenerations { get; set; }
    public int TournamentSize { get; set; }
    public int EliteCount { get; set; }
    public string? CrossoverRate { get; set; }
    public string? UniformCrossoverGeneProbability { get; set; }
    public ulong Seed { get; set; }
    public string? TargetFitness { get; set; }
    public int? StagnationGenerations { get; set; }
    public long? TimeLimitTicks { get; set; }
}

internal sealed class IndividualDto
{
    public GenomeDto? Genome { get; set; }
    public string? FitnessValue { get; set; }
    public bool FitnessIsValid { get; set; }
}

internal sealed class CheckpointDto
{
    public string? SchemaVersion { get; set; }
    public string? ProblemId { get; set; }
    public string? ValueTypeId { get; set; }
    public string? FunctionSetId { get; set; }
    public string? RngVersion { get; set; }
    public ShapeDto? Shape { get; set; }
    public OptionsDto? Options { get; set; }
    public int Generation { get; set; }
    public ulong RootSeed { get; set; }
    public int LastImprovement { get; set; }
    public IndividualDto[]? Population { get; set; }
    public GenomeDto? BestGenome { get; set; }
    public string? BestFitnessValue { get; set; }
    public bool BestFitnessIsValid { get; set; }
    public long? ActiveElapsedTicks { get; set; }

    /// <summary>
    /// Campo aditivo (schema v2+): identidade da estratégia de variação injetada no engine.
    /// Ausente em checkpoints legados ou quando o engine usa o caminho legado; nesse caso não é validada.
    /// </summary>
    public string? VariationStrategyId { get; set; }
}
