using System.Globalization;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;

namespace Evo.Core.Pipeline;

/// <summary>Manifesto de uma execução; é o último arquivo escrito na criação (ponto de commit).</summary>
public sealed record EvolutionPipelineManifest(
    string Schema,
    string PipelineId,
    string RunId,
    string EffectiveParametersSha256,
    string CreatedUtc);

/// <summary>
/// Armazenamento de evidência em arquivos BCL: manifesto e parâmetros efetivos escritos atomicamente,
/// <c>events.jsonl</c> append-only e checkpoints por estado. Não usa rede, banco ou <c>evidence/spec-status</c>.
/// </summary>
public sealed class FileEvolutionEvidenceStore
{
    public const string ManifestSchema = "evo-pipeline-manifest/v1";

    private static readonly JsonSerializerOptions Json = new()
    {
        PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
    };

    public FileEvolutionEvidenceStore(string directory)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(directory);
        Directory = Path.GetFullPath(directory);
    }

    public string Directory { get; }

    private string ManifestPath => Path.Combine(Directory, "manifest.json");

    private string EffectivePath => Path.Combine(Directory, "effective-parameters.json");

    private string EventsPath => Path.Combine(Directory, "events.jsonl");

    public void CreateRun(EvolutionPipelineManifest manifest, string effectiveParametersJson)
    {
        ArgumentNullException.ThrowIfNull(manifest);
        if (File.Exists(ManifestPath))
        {
            throw new InvalidOperationException("Já existe manifesto nesta evidência; use Resume.");
        }

        System.IO.Directory.CreateDirectory(Directory);
        AtomicWrite(EffectivePath, effectiveParametersJson);
        AtomicWrite(ManifestPath, JsonSerializer.Serialize(manifest, Json));
    }

    public EvolutionPipelineManifest ReadManifest()
    {
        if (!File.Exists(ManifestPath))
        {
            throw new InvalidDataException("Manifesto ausente; nada a retomar.");
        }

        return JsonSerializer.Deserialize<EvolutionPipelineManifest>(File.ReadAllText(ManifestPath, Encoding.UTF8), Json)
            ?? throw new InvalidDataException("Manifesto inválido.");
    }

    public string ReadEffectiveParameters()
    {
        if (!File.Exists(EffectivePath))
        {
            throw new InvalidDataException("Parâmetros efetivos ausentes.");
        }

        return File.ReadAllText(EffectivePath, Encoding.UTF8);
    }

    /// <summary>Lê eventos validando sequência 1..n. Uma linha final truncada (escrita interrompida) é descartada.</summary>
    public IReadOnlyList<EvolutionPipelineEvent> ReadEvents()
    {
        var events = new List<EvolutionPipelineEvent>();
        if (!File.Exists(EventsPath))
        {
            return events;
        }

        var lines = File.ReadAllLines(EventsPath, Encoding.UTF8);
        for (var i = 0; i < lines.Length; i++)
        {
            if (lines[i].Length == 0)
            {
                continue;
            }

            EvolutionPipelineEvent? evt;
            try
            {
                evt = JsonSerializer.Deserialize<EvolutionPipelineEvent>(lines[i], Json);
            }
            catch (JsonException) when (i == lines.Length - 1)
            {
                break;
            }

            if (evt is null || evt.Sequence != events.Count + 1)
            {
                throw new InvalidDataException($"Evento inválido ou fora de sequência na linha {i + 1} de events.jsonl.");
            }

            events.Add(evt);
        }

        return events;
    }

    /// <summary>Anexa um evento e força a escrita em disco antes de retornar.</summary>
    public void AppendEvent(EvolutionPipelineEvent evt)
    {
        System.IO.Directory.CreateDirectory(Directory);
        var bytes = Encoding.UTF8.GetBytes(JsonSerializer.Serialize(evt, Json) + "\n");
        using var stream = new FileStream(EventsPath, FileMode.Append, FileAccess.Write, FileShare.Read);
        stream.Write(bytes);
        stream.Flush(flushToDisk: true);
    }

    public void AppendGeneration(string stateId, int generation, int stateGeneration, FitnessResult best, string digest)
    {
        var path = Path.Combine(StateDirectory(stateId), "generations.csv");
        System.IO.Directory.CreateDirectory(StateDirectory(stateId));
        if (!File.Exists(path))
        {
            File.WriteAllText(path, "generation,stateGeneration,bestValid,bestValue,populationDigest\n", Encoding.UTF8);
        }

        var value = best.IsValid ? best.Value.ToString("R", CultureInfo.InvariantCulture) : string.Empty;
        var row = string.Join(',', generation, stateGeneration, best.IsValid ? "true" : "false", value, digest) + "\n";
        File.AppendAllText(path, row, Encoding.UTF8);
    }

    /// <summary>Grava o checkpoint do estado com <paramref name="write"/> e devolve o SHA-256 do arquivo.</summary>
    public string SaveCheckpoint(string stateId, Action<string> write)
    {
        var path = CheckpointPath(stateId);
        System.IO.Directory.CreateDirectory(Path.GetDirectoryName(path)!);
        write(path);
        return Sha256File(path);
    }

    public string CheckpointPath(string stateId) => Path.Combine(StateDirectory(stateId), "checkpoints", "latest.json");

    public static string Sha256File(string path)
    {
        using var stream = File.OpenRead(path);
        return Convert.ToHexString(SHA256.HashData(stream));
    }

    public static string Sha256Text(string text) =>
        Convert.ToHexString(SHA256.HashData(Encoding.UTF8.GetBytes(text)));

    private string StateDirectory(string stateId)
    {
        IdRules.Validate(stateId, nameof(stateId));
        return Path.Combine(Directory, "states", stateId);
    }

    private static void AtomicWrite(string path, string text)
    {
        var temp = path + "." + Guid.NewGuid().ToString("N") + ".tmp";
        File.WriteAllText(temp, text, new UTF8Encoding(encoderShouldEmitUTF8Identifier: false));
        File.Move(temp, path, overwrite: true);
    }
}
