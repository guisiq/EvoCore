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
                ("benchmark", "symbolic-regression-suite") => RunRegressionSuiteBenchmark(args[2..]),
                ("benchmark", "regression-grid") => RunRegressionGrid(args[2..]),
                ("example", "list") => RunExampleList(),
                ("example", "run") => RunExampleRun(args[2..]),
                ("benchmark", "list") => RunExampleList(),
                ("benchmark", "run") => RunExampleRun(args[2..]),
                ("pipeline", "list") => PipelineCommands.List(),
                ("pipeline", "run") => PipelineCommands.Run(args[2..]),
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
        Console.WriteLine("  benchmark symbolic-regression-suite [--generations N] [--population N] [--workers N] [--out DIR]");
        Console.WriteLine("  benchmark regression-grid [--seeds N] [--stride N] [--scale N] [--out DIR]");
        Console.WriteLine("  example list");
        Console.WriteLine("  example run <id> [--out DIR]");
        Console.WriteLine("  benchmark list");
        Console.WriteLine("  benchmark run <id> [--out DIR]");
        Console.WriteLine("  pipeline list");
        Console.WriteLine("  pipeline run <definicao.json> [--run-id ID] [--out DIR]");
        Console.WriteLine($"Contrato RNG: {RandomStream.Version}");
    }

    private static int RunExampleList()
    {
        foreach (var scenario in Evo.Examples.Scenarios.BenchmarkScenarioRegistry.All)
        {
            Console.WriteLine($"{scenario.Id}\t{scenario.Description}");
        }

        return 0;
    }

    private static int RunExampleRun(string[] args)
    {
        if (args.Length == 0)
        {
            Console.Error.WriteLine("uso: example run <id> [--out DIR] | benchmark run <id> [--out DIR]");
            return 2;
        }

        var id = args[0];
        string? outDir = null;
        for (var i = 1; i < args.Length; i++)
        {
            if (args[i] == "--out" && i + 1 < args.Length)
            {
                outDir = args[++i];
            }
        }

        Evo.Examples.Scenarios.IBenchmarkScenario scenario;
        try
        {
            scenario = Evo.Examples.Scenarios.BenchmarkScenarioRegistry.Resolve(id);
        }
        catch (KeyNotFoundException ex)
        {
            Console.Error.WriteLine($"erro: {ex.Message}");
            return 1;
        }

        var result = scenario.Run(CancellationToken.None);
        outDir ??= Path.Combine("runs", $"scenario-{id}");
        Directory.CreateDirectory(outDir);
        var manifest = new
        {
            schema = "evo-benchmark-scenario-result/v1",
            scenarioId = result.ScenarioId,
            gatesPassed = result.GatesPassed,
            gates = result.Gates,
            measurements = result.Measurements,
            rawData = result.RawData,
            parameters = result.Parameters,
            summary = result.Summary,
            environment = new
            {
                runtime = System.Runtime.InteropServices.RuntimeInformation.FrameworkDescription,
                architecture = System.Runtime.InteropServices.RuntimeInformation.ProcessArchitecture.ToString(),
                os = System.Runtime.InteropServices.RuntimeInformation.OSDescription,
            },
        };
        File.WriteAllText(Path.Combine(outDir, "scenario-result.json"), JsonSerializer.Serialize(manifest, JsonOptions));
        Console.WriteLine($"cenário={id} gatesPassed={result.GatesPassed} resumo={result.Summary}");
        Console.WriteLine($"artefatos: {Path.GetFullPath(outDir)}");
        return result.GatesPassed ? 0 : 1;
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
        var rows = new StringBuilder("seed,trainingMse,validationMse,generation,terminationReason,elapsedSeconds,workers\n");
        var effectiveWorkers = baseOptions.WorkerCount > 0 ? baseOptions.WorkerCount : Environment.ProcessorCount;
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

            rows.Append(CultureInfo.InvariantCulture, $"{seed},{training:R},{validation:R},{result.GenerationsRun},{result.Termination},{sw.Elapsed.TotalSeconds:F3},{effectiveWorkers}\n");
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

    private static int RunRegressionGrid(string[] args)
    {
        var seeds = 8;
        var stride = 5;
        var scale = 1;
        string? outDir = null;
        for (var i = 0; i < args.Length - 1; i += 2)
        {
            switch (args[i])
            {
                case "--seeds":
                    seeds = int.Parse(args[i + 1], CultureInfo.InvariantCulture);
                    break;
                case "--stride":
                    stride = int.Parse(args[i + 1], CultureInfo.InvariantCulture);
                    break;
                case "--scale":
                    scale = int.Parse(args[i + 1], CultureInfo.InvariantCulture);
                    break;
                case "--out":
                    outDir = args[i + 1];
                    break;
                default:
                    throw new ArgumentException($"opção desconhecida: {args[i]}");
            }
        }

        outDir ??= Path.Combine("runs", "regression-grid");
        var stopwatch = Stopwatch.StartNew();
        var total = RegressionGridConfigurations.All.Count * RegressionGridCatalog.All.Count * seeds;
        var done = 0;
        var report = RegressionGridRunner.Run(
            seeds,
            stride,
            scale,
            new SynchronousProgress<string>(line =>
            {
                done++;
                var eta = TimeSpan.FromSeconds(stopwatch.Elapsed.TotalSeconds / done * (total - done));
                Console.WriteLine($"{line} eta={eta:hh\\:mm\\:ss}");
                Console.Out.Flush();
            }));
        stopwatch.Stop();
        report.Write(outDir);

        var summary = new StringBuilder();
        summary.Append(CultureInfo.InvariantCulture,
            $"Grade: {RegressionGridConfigurations.All.Count} configurações x {RegressionGridCatalog.All.Count} problemas x {seeds} seeds = {report.Runs.Count} execuções\n");
        summary.Append(CultureInfo.InvariantCulture, $"Tempo total: {stopwatch.Elapsed.TotalMinutes:F1} min\n");
        summary.Append("Qualidade contínua: accuracyDigits = -log10(validationNmse); nenhum limiar é aplicado.\n\n");
        summary.Append("experimento,execuções,digitosMedianos,digitosMedios,reuseEdgeRatioMedioDelta\n");
        foreach (var group in report.Runs.GroupBy(run => run.Experiment).OrderBy(g => g.Key, StringComparer.Ordinal))
        {
            var digits = group.Select(run => RegressionGridReport.AccuracyDigits(run.ValidationNmse)).OrderBy(d => d).ToArray();
            var median = digits.Length % 2 == 1
                ? digits[digits.Length / 2]
                : (digits[(digits.Length / 2) - 1] + digits[digits.Length / 2]) / 2.0;
            summary.Append(CultureInfo.InvariantCulture,
                $"{group.Key},{group.Count()},{median:F3},{digits.Average():F3}," +
                $"{group.Average(run => run.FinalPopulationMean.ReuseEdgeRatio - run.InitialPopulationMean.ReuseEdgeRatio):F4}\n");
        }

        File.WriteAllText(Path.Combine(outDir, "grid-summary.txt"), summary.ToString());
        Console.Write(summary.ToString());
        Console.WriteLine($"artefatos: {Path.GetFullPath(outDir)}");
        return 0;
    }

    private static int RunRegressionSuiteBenchmark(string[] args)    {
        var baseOptions = ParseOptions(args, new EvolutionOptions(), out var outDir);
        outDir ??= Path.Combine("runs", "symbolic-regression-suite");
        Directory.CreateDirectory(outDir);

        var effectiveWorkers = baseOptions.WorkerCount > 0 ? baseOptions.WorkerCount : Environment.ProcessorCount;
        var observations = new List<RegressionGenerationObservation>();
        var results = new List<RegressionRunResult>();

        foreach (var definition in RegressionSuiteCatalog.All)
        {
            var problem = definition.CreateGeneticProblem();
            var evaluator = new ReferenceGenomeEvaluator<double>(problem.Functions);
            for (ulong seed = 0; seed < 30; seed++)
            {
                var seedOptions = baseOptions with { Seed = seed };
                DagReuseAverages? initialPopulationMean = null;
                DagReuseAverages finalPopulationMean = default;
                var stopwatch = Stopwatch.StartNew();
                var engine = new EvolutionEngine<double>(
                    checkpoint: state =>
                    {
                        var bestMetrics = DagReuseMetrics.Measure(state.BestGenome!, problem.Functions);
                        var populationMetrics = state.Population
                            .Select(genome => DagReuseMetrics.Measure(genome, problem.Functions))
                            .ToArray();
                        var populationMean = DagReuseAverages.From(populationMetrics);
                        initialPopulationMean ??= populationMean;
                        finalPopulationMean = populationMean;
                        var validation = MeanSquaredError(
                            state.BestGenome!,
                            evaluator,
                            definition.Dataset.ValidationInputs,
                            definition.Target);
                        observations.Add(new RegressionGenerationObservation(
                            definition.Id,
                            seed,
                            state.Generation,
                            -state.BestFitness.Value,
                            validation,
                            NormalizedMeanSquaredError(
                                validation,
                                definition.Dataset.ValidationInputs,
                                definition.Target),
                            bestMetrics,
                            populationMean));
                    });
                var result = engine.Run(problem, seedOptions);
                stopwatch.Stop();

                var bestMetricsFinal = DagReuseMetrics.Measure(result.BestGenome, problem.Functions);
                var validationFinal = MeanSquaredError(
                    result.BestGenome,
                    evaluator,
                    definition.Dataset.ValidationInputs,
                    definition.Target);
                var validationNmseFinal = NormalizedMeanSquaredError(
                    validationFinal,
                    definition.Dataset.ValidationInputs,
                    definition.Target);
                results.Add(new RegressionRunResult(
                    definition.Id,
                    seed,
                    -result.BestFitness.Value,
                    validationFinal,
                    validationNmseFinal,
                    result.GenerationsRun,
                    result.Termination,
                    stopwatch.Elapsed.TotalSeconds,
                    effectiveWorkers,
                    validationNmseFinal < 1e-3,
                    bestMetricsFinal,
                    initialPopulationMean ?? default,
                    finalPopulationMean));
                Console.WriteLine(
                    $"{definition.Id} seed={seed} validationNMSE={validationNmseFinal:E3} " +
                    $"reuse={bestMetricsFinal.ReuseEdgeRatio:P1}");
            }
        }

        WriteRegressionHistory(Path.Combine(outDir, "evolution-history.csv"), observations);
        WriteRegressionGenerationSummary(Path.Combine(outDir, "generation-summary.csv"), observations);
        WriteRegressionResults(Path.Combine(outDir, "suite-results.csv"), results);
        WriteRegressionProblemSummary(Path.Combine(outDir, "problem-summary.csv"), results);
        var totalPassing = results.Count(result => result.Success);
        File.WriteAllText(Path.Combine(outDir, "suite-summary.txt"),
            $"Suíte: {RegressionSuiteCatalog.All.Count} problemas x 30 seeds = {results.Count} execuções\n" +
            $"Sucesso (validation NMSE < 1e-3): {totalPassing}/{results.Count}\n" +
            "Reutilização: nó ativo com fan-out semântico > 1; reuse edges contam referências além da primeira.\n");
        Console.WriteLine($"Suíte concluída: {totalPassing}/{results.Count} sucessos.");
        return 0;
    }

    private static double MeanSquaredError(
        Genome<double> genome,
        IGenomeEvaluator<double> evaluator,
        double[] inputs,
        Func<double, double> target)
    {
        var input = new double[1];
        var output = new double[1];
        var sum = 0.0;
        foreach (var value in inputs)
        {
            input[0] = value;
            evaluator.Evaluate(genome, input, output);
            var difference = output[0] - target(value);
            sum += difference * difference;
        }

        return sum / inputs.Length;
    }

    private static double NormalizedMeanSquaredError(
        double mse,
        double[] inputs,
        Func<double, double> target)
    {
        var mean = inputs.Average(target);
        var variance = inputs.Average(value =>
        {
            var difference = target(value) - mean;
            return difference * difference;
        });
        return variance == 0.0 ? (mse == 0.0 ? 0.0 : double.PositiveInfinity) : mse / variance;
    }

    private static void WriteRegressionHistory(
        string path,
        IReadOnlyList<RegressionGenerationObservation> observations)
    {
        var csv = new StringBuilder(
            "problem,seed,generation,bestTrainingMse,bestValidationMse,bestValidationNmse," +
            "bestActiveNodes,bestReusedActiveNodes,bestReuseEdges,bestSemanticReferences," +
            "bestReusedActiveNodeRatio,bestReuseEdgeRatio," +
            "populationMeanActiveNodes,populationMeanReusedActiveNodes,populationMeanReuseEdges," +
            "populationMeanSemanticReferences,populationMeanReusedActiveNodeRatio,populationMeanReuseEdgeRatio\n");
        foreach (var observation in observations)
        {
            csv.Append(CultureInfo.InvariantCulture,
                $"{observation.Problem},{observation.Seed},{observation.Generation}," +
                $"{observation.TrainingMse:R},{observation.ValidationMse:R},{observation.ValidationNmse:R}," +
                $"{observation.Best.ActiveNodes},{observation.Best.ReusedActiveNodes}," +
                $"{observation.Best.ReuseEdges},{observation.Best.SemanticReferences}," +
                $"{observation.Best.ReusedActiveNodeRatio:R},{observation.Best.ReuseEdgeRatio:R}," +
                $"{observation.PopulationMean.ActiveNodes:R},{observation.PopulationMean.ReusedActiveNodes:R}," +
                $"{observation.PopulationMean.ReuseEdges:R},{observation.PopulationMean.SemanticReferences:R}," +
                $"{observation.PopulationMean.ReusedActiveNodeRatio:R},{observation.PopulationMean.ReuseEdgeRatio:R}\n");
        }

        File.WriteAllText(path, csv.ToString());
    }

    private static void WriteRegressionGenerationSummary(
        string path,
        IReadOnlyList<RegressionGenerationObservation> observations)
    {
        var csv = new StringBuilder(
            "problem,generation,seeds,meanBestValidationNmse,meanBestActiveNodes," +
            "meanBestReusedActiveNodeRatio,meanBestReuseEdgeRatio," +
            "meanPopulationActiveNodes,meanPopulationReusedActiveNodeRatio,meanPopulationReuseEdgeRatio\n");
        foreach (var group in observations.GroupBy(item => (item.Problem, item.Generation)))
        {
            csv.Append(CultureInfo.InvariantCulture,
                $"{group.Key.Problem},{group.Key.Generation},{group.Count()}," +
                $"{group.Average(item => item.ValidationNmse):R}," +
                $"{group.Average(item => item.Best.ActiveNodes):R}," +
                $"{group.Average(item => item.Best.ReusedActiveNodeRatio):R}," +
                $"{group.Average(item => item.Best.ReuseEdgeRatio):R}," +
                $"{group.Average(item => item.PopulationMean.ActiveNodes):R}," +
                $"{group.Average(item => item.PopulationMean.ReusedActiveNodeRatio):R}," +
                $"{group.Average(item => item.PopulationMean.ReuseEdgeRatio):R}\n");
        }

        File.WriteAllText(path, csv.ToString());
    }

    private static void WriteRegressionResults(
        string path,
        IReadOnlyList<RegressionRunResult> results)
    {
        var csv = new StringBuilder(
            "problem,seed,trainingMse,validationMse,validationNmse,generation,terminationReason," +
            "elapsedSeconds,workers,success,bestActiveNodes,bestReusedActiveNodes,bestReuseEdges," +
            "bestSemanticReferences,bestReusedActiveNodeRatio,bestReuseEdgeRatio," +
            "initialPopulationMeanReusedActiveNodeRatio,finalPopulationMeanReusedActiveNodeRatio," +
            "populationReusedActiveNodeRatioDelta,initialPopulationMeanReuseEdgeRatio," +
            "finalPopulationMeanReuseEdgeRatio,populationReuseEdgeRatioDelta\n");
        foreach (var result in results)
        {
            csv.Append(CultureInfo.InvariantCulture,
                $"{result.Problem},{result.Seed},{result.TrainingMse:R},{result.ValidationMse:R}," +
                $"{result.ValidationNmse:R},{result.Generation},{result.Termination}," +
                $"{result.ElapsedSeconds:F3},{result.Workers},{result.Success}," +
                $"{result.Best.ActiveNodes},{result.Best.ReusedActiveNodes},{result.Best.ReuseEdges}," +
                $"{result.Best.SemanticReferences},{result.Best.ReusedActiveNodeRatio:R},{result.Best.ReuseEdgeRatio:R}," +
                $"{result.InitialPopulationMean.ReusedActiveNodeRatio:R},{result.FinalPopulationMean.ReusedActiveNodeRatio:R}," +
                $"{result.FinalPopulationMean.ReusedActiveNodeRatio - result.InitialPopulationMean.ReusedActiveNodeRatio:R}," +
                $"{result.InitialPopulationMean.ReuseEdgeRatio:R},{result.FinalPopulationMean.ReuseEdgeRatio:R}," +
                $"{result.FinalPopulationMean.ReuseEdgeRatio - result.InitialPopulationMean.ReuseEdgeRatio:R}\n");
        }

        File.WriteAllText(path, csv.ToString());
    }

    private static void WriteRegressionProblemSummary(
        string path,
        IReadOnlyList<RegressionRunResult> results)
    {
        var csv = new StringBuilder(
            "problem,runs,successes,successRate,meanValidationMse,meanValidationNmse," +
            "meanBestActiveNodes,meanBestReusedActiveNodeRatio,meanBestReuseEdgeRatio," +
            "meanInitialPopulationReusedActiveNodeRatio,meanFinalPopulationReusedActiveNodeRatio," +
            "meanPopulationReusedActiveNodeRatioDelta,meanInitialPopulationReuseEdgeRatio," +
            "meanFinalPopulationReuseEdgeRatio,meanPopulationReuseEdgeRatioDelta,meanElapsedSeconds\n");
        foreach (var group in results.GroupBy(result => result.Problem))
        {
            csv.Append(CultureInfo.InvariantCulture,
                $"{group.Key},{group.Count()},{group.Count(item => item.Success)}," +
                $"{group.Average(item => item.Success ? 1.0 : 0.0):R}," +
                $"{group.Average(item => item.ValidationMse):R},{group.Average(item => item.ValidationNmse):R}," +
                $"{group.Average(item => item.Best.ActiveNodes):R}," +
                $"{group.Average(item => item.Best.ReusedActiveNodeRatio):R}," +
                $"{group.Average(item => item.Best.ReuseEdgeRatio):R}," +
                $"{group.Average(item => item.InitialPopulationMean.ReusedActiveNodeRatio):R}," +
                $"{group.Average(item => item.FinalPopulationMean.ReusedActiveNodeRatio):R}," +
                $"{group.Average(item => item.FinalPopulationMean.ReusedActiveNodeRatio - item.InitialPopulationMean.ReusedActiveNodeRatio):R}," +
                $"{group.Average(item => item.InitialPopulationMean.ReuseEdgeRatio):R}," +
                $"{group.Average(item => item.FinalPopulationMean.ReuseEdgeRatio):R}," +
                $"{group.Average(item => item.FinalPopulationMean.ReuseEdgeRatio - item.InitialPopulationMean.ReuseEdgeRatio):R}," +
                $"{group.Average(item => item.ElapsedSeconds):R}\n");
        }

        File.WriteAllText(path, csv.ToString());
    }

    private sealed record RegressionGenerationObservation(
        string Problem,
        ulong Seed,
        int Generation,
        double TrainingMse,
        double ValidationMse,
        double ValidationNmse,
        DagReuseMetrics Best,
        DagReuseAverages PopulationMean);

    private sealed record RegressionRunResult(
        string Problem,
        ulong Seed,
        double TrainingMse,
        double ValidationMse,
        double ValidationNmse,
        int Generation,
        TerminationReason Termination,
        double ElapsedSeconds,
        int Workers,
        bool Success,
        DagReuseMetrics Best,
        DagReuseAverages InitialPopulationMean,
        DagReuseAverages FinalPopulationMean);

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
