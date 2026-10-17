using System.Globalization;
using System.Text.Json;
using Evo.Core;
using Evo.Core.Pipeline;

namespace Evo.Cli;

/// <summary>
/// Comando `pipeline` da CLI: carrega um pipeline dinâmico (schema <c>evo-pipeline/v1</c>) a partir de um
/// arquivo JSON e o executa sobre o engine real, usando um registro de problemas conhecidos pela CLI.
/// Nenhum pipeline é hardcoded: a definição completa (estados, opções, condições) vem do arquivo do usuário.
/// </summary>
internal static class PipelineCommands
{
    private static readonly JsonSerializerOptions JsonOptions = new() { WriteIndented = true };

    private static EvolutionPipelineRegistry<double> BuildRegistry()
    {
        var registry = new EvolutionPipelineRegistry<double>();
        registry.RegisterProblem("symbolic-regression", static () => new SymbolicRegressionProblem());
        return registry;
    }

    public static int List()
    {
        Console.WriteLine("problemas registrados para `pipeline run`:");
        Console.WriteLine("  symbolic-regression");
        Console.WriteLine($"schema de definição de pipeline: {EvolutionPipelineDocument.Schema}");
        return 0;
    }

    public static int Run(string[] args)
    {
        if (args.Length == 0)
        {
            Console.Error.WriteLine("uso: pipeline run <definicao.json> [--run-id ID] [--out DIR]");
            return 2;
        }

        var definitionPath = args[0];
        string? runId = null;
        string? outDir = null;
        for (var i = 1; i < args.Length; i++)
        {
            switch (args[i])
            {
                case "--run-id" when i + 1 < args.Length:
                    runId = args[++i];
                    break;
                case "--out" when i + 1 < args.Length:
                    outDir = args[++i];
                    break;
            }
        }

        if (!File.Exists(definitionPath))
        {
            Console.Error.WriteLine($"erro: definição de pipeline não encontrada: '{definitionPath}'.");
            return 1;
        }

        var json = File.ReadAllText(definitionPath);
        var registry = BuildRegistry();
        var definition = EvolutionPipelineDocument.Load(json, registry);
        runId ??= "run-" + DateTime.UtcNow.ToString("yyyyMMddHHmmss", CultureInfo.InvariantCulture);
        outDir ??= Path.Combine("runs", $"pipeline-{definition.PipelineId}-{runId}");
        Directory.CreateDirectory(outDir);

        var runner = new EvolutionPipelineRunner<double>();
        var result = runner.Run(definition, runId, outDir);

        var summary = new
        {
            schema = "evo-pipeline-cli-result/v1",
            pipelineId = definition.PipelineId,
            runId,
            finalStateId = result.FinalStateId,
            transitions = result.Transitions,
            generationsRun = result.GenerationsRun,
            termination = result.Termination.ToString(),
            nativeTermination = result.NativeTermination?.ToString(),
            reason = result.Reason,
            bestFitness = result.BestFitness.Value,
            evidenceDirectory = Path.GetFullPath(outDir),
        };
        File.WriteAllText(Path.Combine(outDir, "pipeline-result.json"), JsonSerializer.Serialize(summary, JsonOptions));
        Console.WriteLine($"pipeline={definition.PipelineId} runId={runId} estadoFinal={result.FinalStateId} transicoes={result.Transitions} terminacao={result.Termination} motivo={result.Reason}");
        Console.WriteLine($"artefatos: {Path.GetFullPath(outDir)}");
        return 0;
    }
}
