using System.Globalization;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;

namespace Evo.Examples.Datasets;

/// <summary>Entrada do registro autoritativo <c>datasets.json</c> (origem, licença, shape e SHA-256).</summary>
public sealed record DatasetDescriptor(
    string Id,
    string File,
    string Format,
    int Records,
    int Features,
    string Target,
    string License,
    string Sha256);

/// <summary>Erro explícito de validação de dataset: ausência, checksum ou shape divergentes, ou parse parcial.</summary>
public sealed class DatasetValidationException : Exception
{
    public DatasetValidationException(string message) : base(message)
    {
    }
}

/// <summary>Linhas selecionadas de um dataset já validado (checksum e shape conferidos).</summary>
public sealed record DatasetRows(
    string DatasetId,
    string ObservedSha256,
    int RecordCount,
    int FeatureCount,
    double[][] Features,
    double[] Target);

/// <summary>
/// Loader offline dos datasets versionados em <c>Data/Benchmarks</c>. Nunca baixa, nunca usa fallback
/// sintético: arquivo ausente, checksum divergente ou parse parcial é <see cref="DatasetValidationException"/>.
/// </summary>
public static class DatasetCatalog
{
    private static readonly Lazy<IReadOnlyDictionary<string, DatasetDescriptor>> RegistryLazy = new(LoadRegistry);

    /// <summary>Diretório resolvido onde os arquivos de fixture e <c>datasets.json</c> residem.</summary>
    public static string DataDirectory { get; } = ResolveDataDirectory();

    /// <summary>Registro completo de datasets conhecidos, por ID.</summary>
    public static IReadOnlyDictionary<string, DatasetDescriptor> All => RegistryLazy.Value;

    public static DatasetDescriptor Describe(string datasetId) =>
        All.TryGetValue(datasetId, out var descriptor)
            ? descriptor
            : throw new DatasetValidationException($"dataset não registrado: '{datasetId}'.");

    /// <summary>
    /// Carrega e valida um dataset (checksum e shape) e devolve os primeiros <paramref name="maxRecords"/>
    /// registros, na ordem determinística do arquivo.
    /// </summary>
    public static DatasetRows Load(string datasetId, int maxRecords)
    {
        var descriptor = Describe(datasetId);
        var path = Path.Combine(DataDirectory, descriptor.File);
        if (!File.Exists(path))
        {
            throw new DatasetValidationException(
                $"arquivo de dataset ausente: '{path}' (dataset '{datasetId}'); download e fallback sintético são proibidos em runtime.");
        }

        var bytes = File.ReadAllBytes(path);
        var observedSha = Convert.ToHexString(SHA256.HashData(bytes)).ToLowerInvariant();
        if (!observedSha.Equals(descriptor.Sha256, StringComparison.OrdinalIgnoreCase))
        {
            throw new DatasetValidationException(
                $"checksum divergente para '{datasetId}': esperado={descriptor.Sha256} observado={observedSha}.");
        }

        var (features, target) = descriptor.Format switch
        {
            "arff" => ParseArff(bytes, descriptor),
            "csv-semicolon" => ParseCsvSemicolon(bytes, descriptor),
            _ => throw new DatasetValidationException($"formato de dataset não suportado: '{descriptor.Format}'."),
        };

        if (features.Length != descriptor.Records)
        {
            throw new DatasetValidationException(
                $"shape divergente para '{datasetId}': esperado {descriptor.Records} registros, obtidos {features.Length}.");
        }

        if (maxRecords <= 0 || maxRecords > features.Length)
        {
            throw new DatasetValidationException(
                $"maxRecords inválido para '{datasetId}': {maxRecords} (disponíveis={features.Length}).");
        }

        return new DatasetRows(datasetId, observedSha, maxRecords, descriptor.Features, features[..maxRecords], target[..maxRecords]);
    }

    private static (double[][] Features, double[] Target) ParseArff(byte[] bytes, DatasetDescriptor descriptor)
    {
        var lines = Encoding.UTF8.GetString(bytes).Split('\n');
        var dataStarted = false;
        var featuresList = new List<double[]>();
        var targetList = new List<double>();
        foreach (var raw in lines)
        {
            var line = raw.TrimEnd('\r').Trim();
            if (line.Length == 0 || line.StartsWith('%'))
            {
                continue;
            }

            if (!dataStarted)
            {
                if (line.StartsWith("@data", StringComparison.OrdinalIgnoreCase))
                {
                    dataStarted = true;
                }

                continue;
            }

            var parts = line.Split(',');
            if (parts.Length < descriptor.Features + 1)
            {
                throw new DatasetValidationException($"linha de dados malformada (parse parcial) em '{descriptor.Id}': '{line}'.");
            }

            var row = new double[descriptor.Features];
            for (var i = 0; i < descriptor.Features; i++)
            {
                row[i] = ParseArffField(parts[i], descriptor.Id);
            }

            featuresList.Add(row);
            targetList.Add(ParseArffField(parts[descriptor.Features], descriptor.Id));
        }

        return (featuresList.ToArray(), targetList.ToArray());
    }

    private static double ParseArffField(string rawField, string datasetId)
    {
        var field = rawField.Trim();
        if (double.TryParse(field, NumberStyles.Float, CultureInfo.InvariantCulture, out var value))
        {
            return value;
        }

        if (field.Equals("tested_positive", StringComparison.OrdinalIgnoreCase))
        {
            return 1.0;
        }

        if (field.Equals("tested_negative", StringComparison.OrdinalIgnoreCase))
        {
            return 0.0;
        }

        throw new DatasetValidationException($"campo não numérico não reconhecido em '{datasetId}': '{field}'.");
    }

    private static (double[][] Features, double[] Target) ParseCsvSemicolon(byte[] bytes, DatasetDescriptor descriptor)
    {
        var lines = Encoding.UTF8.GetString(bytes).Split('\n');
        var featuresList = new List<double[]>();
        var targetList = new List<double>();
        var headerSeen = false;
        foreach (var raw in lines)
        {
            var line = raw.TrimEnd('\r');
            if (line.Length == 0)
            {
                continue;
            }

            if (!headerSeen)
            {
                headerSeen = true;
                continue;
            }

            var parts = line.Split(';');
            if (parts.Length < descriptor.Features + 1)
            {
                throw new DatasetValidationException($"linha de dados malformada (parse parcial) em '{descriptor.Id}': '{line}'.");
            }

            var row = new double[descriptor.Features];
            for (var i = 0; i < descriptor.Features; i++)
            {
                row[i] = double.Parse(parts[i], NumberStyles.Float, CultureInfo.InvariantCulture);
            }

            featuresList.Add(row);
            targetList.Add(double.Parse(parts[descriptor.Features], NumberStyles.Float, CultureInfo.InvariantCulture));
        }

        return (featuresList.ToArray(), targetList.ToArray());
    }

    private static IReadOnlyDictionary<string, DatasetDescriptor> LoadRegistry()
    {
        var path = Path.Combine(DataDirectory, "datasets.json");
        if (!File.Exists(path))
        {
            throw new DatasetValidationException($"registro de datasets ausente: '{path}'.");
        }

        using var doc = JsonDocument.Parse(File.ReadAllBytes(path));
        var map = new Dictionary<string, DatasetDescriptor>(StringComparer.Ordinal);
        foreach (var entry in doc.RootElement.GetProperty("datasets").EnumerateArray())
        {
            var descriptor = new DatasetDescriptor(
                entry.GetProperty("id").GetString()!,
                entry.GetProperty("file").GetString()!,
                entry.GetProperty("format").GetString()!,
                entry.GetProperty("records").GetInt32(),
                entry.GetProperty("features").GetInt32(),
                entry.GetProperty("target").GetString()!,
                entry.GetProperty("license").GetString()!,
                entry.GetProperty("sha256").GetString()!);
            map[descriptor.Id] = descriptor;
        }

        return map;
    }

    private static string ResolveDataDirectory()
    {
        var baseDir = AppContext.BaseDirectory;
        var candidate = Path.Combine(baseDir, "Data", "Benchmarks");
        if (Directory.Exists(candidate))
        {
            return candidate;
        }

        var dir = new DirectoryInfo(baseDir);
        while (dir is not null)
        {
            var probe = Path.Combine(dir.FullName, "src", "Evo.Examples", "Data", "Benchmarks");
            if (Directory.Exists(probe))
            {
                return probe;
            }

            dir = dir.Parent;
        }

        throw new DatasetValidationException(
            "diretório de datasets 'Data/Benchmarks' não encontrado; execução offline requer fixtures locais presentes.");
    }
}
