using System.Diagnostics;
using System.Globalization;
using System.Text;
using System.Text.Json;
using Evo.Core;
using Evo.Core.Persistence;

namespace Evo.Cli;

/// <summary>
/// CLI da demo v1. Comandos: <c>demo symbolic-regression</c>, <c>inspect</c>, <c>resume</c> e
/// <c>benchmark symbolic-regression-30</c> (runner de 30 seeds, fora de <c>dotnet test</c>).
/// </summary>
public static class Program
{
    private const string ConfigFileName = "config.json";
    private const string CheckpointFileName = "checkpoint.json";
    private const string GenomeFileName = "best-genome.json";
    private const string HistoryFileName = "history.csv";
    private const string HistoryHeader = "generation,bestFitness,trainingMse,populationDigest\n";

    private static readonly JsonSerializerOptions JsonOptions = new() { WriteIndented = true };

    /// <summary>Ponto de entrada. Retorna 0 em sucesso, 2 em uso inválido e 1 em erro de execução.</summary>
    public static int Main(string[] args)
    {
        if (args.Length == 0)
        {
            PrintUsage();
            return 0;
        }

        try
        {
            var command = args[0];
            var sub = args.Length > 1 ? args[1] : null;
            return (command, sub) switch
            {
                ("demo", "symbolic-regression") => RunDemo(args[2..]),
                ("inspect", not null) => RunInspect(sub!),
                ("resume", not null) => RunResume(sub!, args[2..]),
                ("benchmark", "symbolic-regression-30") => RunBenchmark(args[2..]),
                _ => Usage(),
            };
        }
        catch (Exception ex) when (ex is FormatException or OverflowException)
        {
            Console.Error.WriteLine($"argumento numérico inválido: {ex.Message}");
            PrintUsage();
            return 2;
        }
        catch (Exception ex) when (ex is IOException or InvalidDataException or InvalidOperationException or ArgumentException or InvalidGenomeException)
        {
            Console.Error.WriteLine($"erro: {ex.Message}");
            return 1;
        }
    }

    private static int Usage()
    {
        PrintUsage();
        return 2;
    }

    private static void PrintUsage()
    {
        Console.WriteLine("Evo.Cli — demo v1 (regressão simbólica, dataset fixo de 201 pontos).");
        Console.WriteLine("Uso:");
        Console.WriteLine("  demo symbolic-regression [--seed N] [--generations N] [--population N] [--workers N] [--out DIR]");
        Console.WriteLine("  inspect <genome.json>");
        Console.WriteLine("  resume <checkpoint.json> [--out DIR]");
        Console.WriteLine("  benchmark symbolic-regression-30 [--generations N] [--population N] [--workers N] [--out DIR]");
        Console.WriteLine($"Contrato RNG: {RandomStream.Version}");
    }

    private static int RunDemo(string[] args)
    {
        var options = ParseOptions(args, new EvolutionOptions(), out var outDir);
        outDir ??= Path.Combine("runs", $"demo-seed-{options.Seed}");
        Directory.CreateDirectory(outDir);

        var problem = new SymbolicRegressionProblem();
        var result = Execute(problem, options, outDir, resume: null);
        Console.WriteLine($"artefatos: {Path.GetFullPath(outDir)}");
        Console.WriteLine($"termination={result.Termination} generations={result.GenerationsRun}");
        return 0;
    }

    private static int RunResume(string checkpointPath, string[] args)
    {
        var problem = new SymbolicRegressionProblem();
        var dir = Path.GetDirectoryName(Path.GetFullPath(checkpointPath))!;
        var baseOptions = ReadConfig(Path.Combine(dir, ConfigFileName));
        var options = ParseOptions(args, baseOptions, out var outDir);
        var state = CheckpointFile.Load(checkpointPath, problem, options);
        if (!state.ElapsedKnown)
        {
            Console.Error.WriteLine(CheckpointFile.LegacyElapsedDiagnostic);
        }

        outDir ??= dir;
        Directory.CreateDirectory(outDir);

        var result = Execute(problem, options, outDir, state, Path.Combine(dir, HistoryFileName));
        Console.WriteLine($"artefatos: {Path.GetFullPath(outDir)}");
        Console.WriteLine($"termination={result.Termination} generations={result.GenerationsRun}");
        return 0;
    }

    private static int RunInspect(string genomePath)
    {
        var problem = new SymbolicRegressionProblem();
        var genome = GenomeJson.Deserialize(File.ReadAllText(genomePath), problem, new EvolutionOptions());
        var functions = problem.Functions;
        var evaluator = new ReferenceGenomeEvaluator<double>(functions);
        var active = genome.Analyze(functions);
        Console.WriteLine($"nós ativos: {active.ActiveNodeCount}");
        Console.WriteLine($"profundidade ativa: {active.ActiveDepth}");
        Console.WriteLine($"fingerprint: {Convert.ToHexString(genome.ComputePhenotypeFingerprint(functions, problem.Values))}");
        Console.WriteLine($"training MSE: {FormatMse(SymbolicRegressionProblem.Mse(genome, evaluator, SymbolicRegressionDataset.TrainingIndices))}");
        Console.WriteLine($"validation MSE: {FormatMse(SymbolicRegressionProblem.Mse(genome, evaluator, SymbolicRegressionDataset.ValidationIndices))}");
        return 0;
    }

    private static int RunBenchmark(string[] args)
    {
        var baseOptions = ParseOptions(args, new EvolutionOptions(), out var outDir);
        outDir ??= Path.Combine("runs", "quality-30");
        Directory.CreateDirectory(outDir);

        var problem = new SymbolicRegressionProblem();
        var evaluator = new ReferenceGenomeEvaluator<double>(problem.Functions);
        var rows = new StringBuilder("seed,trainingMse,validationMse,generation,terminationReason,elapsedSeconds\n");
        var passing = 0;
        for (ulong seed = 0; seed < 30; seed++)
        {
            var options = baseOptions with { Seed = seed };
            var sw = Stopwatch.StartNew();
            var engine = new EvolutionEngine<double>();
            var result = engine.Run(problem, options);
            sw.Stop();
            var validation = SymbolicRegressionProblem.Mse(result.BestGenome, evaluator, SymbolicRegressionDataset.ValidationIndices);
            var training = -result.BestFitness.Value;
            if (validation < 1e-3)
            {
                passing++;
            }

            rows.Append(CultureInfo.InvariantCulture, $"{seed},{training:R},{validation:R},{result.GenerationsRun},{result.Termination},{sw.Elapsed.TotalSeconds:F3}\n");
            Console.WriteLine($"seed={seed} validationMSE={validation:E3} generation={result.GenerationsRun} termination={result.Termination}");
        }

        File.WriteAllText(Path.Combine(outDir, "seed-results.csv"), rows.ToString());
        var status = passing >= 24 ? "PASS" : "FAIL";
        var summary = $"Meta de qualidade 24/30 (validation MSE < 1e-3): {passing}/30 — {status}\n" +
                      "Quality gate reportado: não é gate técnico e não altera defaults.\n";
        File.WriteAllText(Path.Combine(outDir, "quality-summary.txt"), summary);
        Console.Write(summary);
        return 0;
    }

    private static EvolutionResult<double> Execute(
        SymbolicRegressionProblem problem,
        EvolutionOptions options,
        string outDir,
        EvolutionState<double>? resume,
        string? priorHistoryPath = null)
    {
        File.WriteAllText(Path.Combine(outDir, ConfigFileName), JsonSerializer.Serialize(options, JsonOptions));

        var history = new StringBuilder(HistoryHeader);
        if (resume is not null && priorHistoryPath is not null && File.Exists(priorHistoryPath))
        {
            AppendHistoryUpTo(history, priorHistoryPath, resume.Generation);
        }

        var checkpointPath = Path.Combine(outDir, CheckpointFileName);

        var engine = new EvolutionEngine<double>(
            progress: p =>
            {
                history.Append(CultureInfo.InvariantCulture, $"{p.Generation},{p.BestFitness.Value:R},{-p.BestFitness.Value:R},{p.PopulationDigest}\n");
                Console.WriteLine($"gen={p.Generation} trainingMSE={-p.BestFitness.Value:E4}");
            },
            checkpoint: state => CheckpointFile.Save(checkpointPath, problem, options, state));

        var result = resume is null
            ? engine.Run(problem, options)
            : engine.Resume(problem, options, resume);

        File.WriteAllText(Path.Combine(outDir, HistoryFileName), history.ToString());
        WriteBest(outDir, problem, result);
        return result;
    }

    /// <summary>Copia as linhas do histórico anterior estritamente anteriores à geração do checkpoint, que é reavaliada na retomada.</summary>
    private static void AppendHistoryUpTo(StringBuilder history, string priorHistoryPath, int generation)
    {
        foreach (var line in File.ReadLines(priorHistoryPath).Skip(1))
        {
            if (line.Length == 0)
            {
                continue;
            }

            var comma = line.IndexOf(',');
            if (int.Parse(line.AsSpan(0, comma), CultureInfo.InvariantCulture) < generation)
            {
                history.Append(line).Append('\n');
            }
        }
    }

    private static void WriteBest(string outDir, SymbolicRegressionProblem problem, EvolutionResult<double> result)
    {
        var genome = result.BestGenome;
        File.WriteAllText(Path.Combine(outDir, GenomeFileName), GenomeJson.Serialize(genome, problem));

        var functions = problem.Functions;
        var evaluator = new ReferenceGenomeEvaluator<double>(functions);
        var active = genome.Analyze(functions);
        var program = new StringBuilder();
        program.AppendLine($"fingerprint: {Convert.ToHexString(genome.ComputePhenotypeFingerprint(functions, problem.Values))}");
        program.AppendLine($"nós ativos: {active.ActiveNodeCount}");
        program.AppendLine("nós ativos (índice: função, conexões):");
        foreach (var node in active.ActiveNodes)
        {
            var connections = new List<string>();
            for (var slot = 0; slot < problem.Shape.MaxArity; slot++)
            {
                connections.Add(genome.GetConnection(node, slot).ToString(CultureInfo.InvariantCulture));
            }

            program.AppendLine($"{node}: f{genome.FunctionGenes[node]}, [{string.Join(", ", connections)}]");
        }

        program.AppendLine($"training MSE: {FormatMse(SymbolicRegressionProblem.Mse(genome, evaluator, SymbolicRegressionDataset.TrainingIndices))}");
        program.AppendLine($"validation MSE: {FormatMse(SymbolicRegressionProblem.Mse(genome, evaluator, SymbolicRegressionDataset.ValidationIndices))}");
        File.WriteAllText(Path.Combine(outDir, "best-program.txt"), program.ToString());
    }

    private static string FormatMse(double value) => value.ToString("E6", CultureInfo.InvariantCulture);

    private static EvolutionOptions ReadConfig(string path)
    {
        if (!File.Exists(path))
        {
            throw new InvalidOperationException($"config.json não encontrado ao lado do checkpoint: {path}");
        }

        return JsonSerializer.Deserialize<EvolutionOptions>(File.ReadAllText(path))
               ?? throw new InvalidDataException("config.json vazio.");
    }

    private static EvolutionOptions ParseOptions(string[] args, EvolutionOptions current, out string? outDir)
    {
        outDir = null;
        var options = current;
        for (var i = 0; i < args.Length; i++)
        {
            var key = args[i];
            if (i + 1 >= args.Length)
            {
                throw new ArgumentException($"opção sem valor: {key}");
            }

            var value = args[++i];
            switch (key)
            {
                case "--seed": options = options with { Seed = ulong.Parse(value, CultureInfo.InvariantCulture) }; break;
                case "--generations": options = options with { MaxGenerations = int.Parse(value, CultureInfo.InvariantCulture) }; break;
                case "--population": options = options with { PopulationSize = int.Parse(value, CultureInfo.InvariantCulture) }; break;
                case "--workers": options = options with { WorkerCount = int.Parse(value, CultureInfo.InvariantCulture) }; break;
                case "--out": outDir = value; break;
                default: throw new ArgumentException($"opção desconhecida: {key}");
            }
        }

        return options;
    }
}
