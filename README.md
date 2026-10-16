# EvoCore product repository

This repository contains the EvoCore .NET solution, its source projects, tests,
benchmarks, and package/build configuration.

From this directory:

```powershell
dotnet restore EvoCore.sln --locked-mode
dotnet build EvoCore.sln -c Release --no-restore
dotnet test EvoCore.sln -c Release --no-build --blame-hang --blame-hang-timeout 10m
```

When included under the Spec Kit repository, this repository is located at
`product/EvoCore`. Its remote and parent-repository submodule registration are
pending repository creation.

## Primeiro uso reproduzível (v1)

```powershell
dotnet restore product/EvoCore/EvoCore.sln --locked-mode
dotnet build product/EvoCore/EvoCore.sln -c Release --no-restore
dotnet test product/EvoCore/tests/Evo.Cli.Tests -c Release --no-restore
dotnet run --project product/EvoCore/src/Evo.Cli -c Release --no-restore -- demo symbolic-regression --seed 42
dotnet run --project product/EvoCore/src/Evo.Cli -c Release --no-restore -- inspect <run-dir>/best-genome.json
dotnet run --project product/EvoCore/src/Evo.Cli -c Release --no-restore -- resume <run-dir>/checkpoint.json
dotnet run --project product/EvoCore/src/Evo.Cli -c Release --no-restore -- benchmark symbolic-regression-30 --out <dir>
dotnet run --project product/EvoCore/src/Evo.Cli -c Release --no-restore -- benchmark symbolic-regression-suite --out <dir>
dotnet run --project product/EvoCore/src/Evo.Cli -c Release --no-restore -- benchmark regression-grid --out <dir>
```

A demo não usa rede, banco ou GPU. A meta 24/30 do benchmark de 30 seeds é quality gate reportado, não gate técnico.

## Suíte exploratória de regressão simbólica

O comando `benchmark symbolic-regression-suite` é aditivo e não altera o benchmark
fixo `symbolic-regression-30`. Ele executa em 30 seeds os seis problemas
univariados da suíte Nguyen que são representáveis pelo conjunto de funções do
Evo Core: `nguyen-1`, `nguyen-2`, `nguyen-3`, `nguyen-4`, `nguyen-7` e
`nguyen-8`. As expressões e os domínios seguem a suíte publicada em
[_GP Needs Better Benchmarks_](https://doi.org/10.1145/2330163.2330273), conforme
a implementação de referência
[`reg_bench.symbolic_regression.nguyen`](https://regression-benchmarks.readthedocs.io/en/latest/api/reg_bench.symbolic_regression.nguyen.html).
A adaptação local usa 201 pontos uniformemente espaçados e determinísticos por
domínio, alternando índices entre treino e validação; a seed afeta somente a
evolução.

Cada execução grava:

- `suite-results.csv`: uma linha por problema e seed, com MSE, NMSE, tempo,
  métricas finais e deltas entre a população inicial e final;
- `evolution-history.csv`: uma linha por problema, seed e geração, incluindo o
  melhor genoma e as médias da população;
- `generation-summary.csv`: médias das 30 seeds por problema e geração;
- `problem-summary.csv`: sucesso, qualidade e mudança média de reutilização por
  problema;
- `suite-summary.txt`: contagem agregada de execuções e sucessos.

`reusedActiveNodes` conta nós ativos com fan-out semântico maior que um.
`reuseEdges` conta as referências além da primeira para cada nó.
`reusedActiveNodeRatio` divide os nós reutilizados pelos nós ativos;
`reuseEdgeRatio` divide `reuseEdges` pelas referências semânticas a nós. As
colunas `populationMean*` preservam a média real da população, sem arredondamento.
Referências a slots acima da aridade da função são ignoradas. O sucesso da suíte
usa `validation NMSE < 1e-3`, evitando comparar MSEs absolutos de problemas com
escalas de saída diferentes.

## Grade de regressão com baselines e qualidade contínua

O comando `benchmark regression-grid` é exploratório e também aditivo: não toca
`symbolic-regression-30` nem `symbolic-regression-suite`, nem altera contratos,
thresholds ou algoritmos do núcleo.

```powershell
dotnet run --project product/EvoCore/src/Evo.Cli -c Release --no-restore -- `
  benchmark regression-grid --seeds 8 --stride 5 --out runs/regression-grid
```

Opções: `--seeds N` (seeds por célula, padrão 8), `--stride N` (intervalo de
gerações registradas, padrão 5), `--scale N` (divisor de orçamento usado apenas
por smoke tests, padrão 1) e `--out DIR`.

**Problemas.** Cinco famílias com três níveis de complexidade crescente sobre o
mesmo domínio, todas representáveis pelo conjunto padrão de funções: `poly`
(grau polinomial, `[-1, 1]`), `log` (composição logarítmica, `[0, 2]`),
`rational` (divisão protegida, `[-1, 1]`), `root` (radiciação aninhada,
`[0, 4]`) e `nonsmooth` (não suavidade, `[-2, 2]`). A amostragem reaproveita o
dataset determinístico de 201 pontos da suíte Nguyen.

**Configurações.** Duas funções de baseline e três variações do engine:

| Experimento | Papel | Descrição |
| --- | --- | --- |
| `baseline-random-search` | baseline | Genomas aleatórios independentes com o mesmo orçamento de avaliações; fluxo RNG 101, separado dos fluxos 1 e 2 do engine. |
| `baseline-drift` | baseline | `TournamentSize = 1` e `CrossoverRate = 0`: deriva com elitismo e mutação. |
| `evo-default` | variação | Defaults normativos (população 500, 200 gerações). |
| `evo-pop1000` | variação | População 1000, 200 gerações. |
| `evo-gen400` | variação | População 500, 400 gerações. |

As duas últimas gastam o dobro de avaliações; a coluna `evaluations` permite a
comparação com orçamento equalizado. Cada execução roda com `WorkerCount = 1`,
preservando a equivalência bit a bit entre contagens de workers, e o paralelismo
fica no nível das execuções. Como os jobs são concorrentes, `elapsedSeconds` mede
tempo de parede sob contenção e não deve ser interpretado como microbenchmark
isolado; para eficiência, compare também avaliações e o throughput da grade.

Artefatos gerados:

- `grid-generations.csv`: tidy, uma linha por experimento, problema, seed e
  geração observada;
- `grid-runs.csv`: uma linha por execução, com deltas de reutilização;
- `grid-configurations.csv` e `grid-problems.csv`: metadados da grade;
- `grid-summary.txt`: resumo por experimento em escala contínua.

**Qualidade contínua.** A coluna `accuracyDigits = -log10(validationNmse)`
substitui o limiar discreto por uma escala linear em ordens de grandeza; o
critério antigo é exatamente `accuracyDigits >= 3`. Nenhum limiar é aplicado nos
relatórios da grade.

## Notebook de análise

O repositório pai (Spec Kit) hospeda o notebook poliglota C#
`notebooks/evo-regression-analysis.dib`, que lê os CSVs desta grade e produz um
gráfico por métrica com uma cor por experimento, linhas individuais em baixa
opacidade, média em destaque e baselines pontilhados, além da avaliação de
qualidade contínua. O notebook não é consumido pela build nem pelos testes deste
repositório.

## Seletor e variação próprios de um consumidor externo

Um projeto externo pode implementar o problema genético e estender a evolução
com seu próprio seletor de pais (`IParentSelector<T>`) e operador de variação
(`IVariationOperator<T>`), compondo-os com `CompositeVariationOperator<T>` sem
depender de nenhum tipo interno de `Evo.Core`. O exemplo abaixo (ver também
`tests/Evo.ExternalConsumer`) compila com uma referência de projeto a
`Evo.Core`:

```csharp
using Evo.Core;

// Seletor próprio: torneio binário com empate para o menor índice.
public sealed class ConsumerParentSelector<T> : IParentSelector<T>
{
    public int SelectParentIndex(ParentSelectionContext<T> context, ref RandomStream random)
    {
        var first = random.NextInt(0, context.Count);
        var second = random.NextInt(0, context.Count);
        var cmp = EvolutionOperators<T>.CompareFitness(context.Fitness[second], context.Fitness[first]);
        return cmp > 0 || (cmp == 0 && second < first) ? second : first;
    }
}

// Operador de variação próprio: usa o seletor acima com crossover/mutação padrão.
public sealed class ConsumerVariationOperator<T> : IVariationOperator<T>
{
    private readonly EvolutionOperators<T> _operators;
    private readonly IParentSelector<T> _selector;

    public ConsumerVariationOperator(IGeneticProblem<T> problem, IParentSelector<T> selector)
    {
        _operators = new EvolutionOperators<T>(problem);
        _selector = selector;
    }

    public Genome<T> CreateOffspring(VariationContext<T> context, ref RandomStream random) =>
        _operators.BreedWith(_selector, context.Population, context.Fitness, context.Options, ref random);
}

// Composição: metade das posições usa o operador do consumidor, metade usa o
// operador padrão do engine (Sequential encadearia os dois no mesmo descendente).
var problem = new MyGeneticProblem();
var composite = new CompositeVariationOperator<double>(
    CompositionMode.Partitioned,
    new IVariationOperator<double>[]
    {
        new ConsumerVariationOperator<double>(problem, new ConsumerParentSelector<double>()),
        new StandardVariationOperator<double>(problem),
    },
    weights: new[] { 1.0, 1.0 });

var engine = new EvolutionEngine<double>(composite);
var result = engine.Run(problem, new EvolutionOptions { PopulationSize = 64, MaxGenerations = 50, Seed = 1 });
```

O construtor `EvolutionEngine<T>(IVariationOperator<T> variation, ...)` preserva
o overload legado (sem variação injetada, torneio padrão); checkpoints gravados
com uma estratégia customizada carregam a identidade canônica da composição
(`CompositeVariationOperator<T>.StrategyId`, incluindo modo, pesos e estágios
aninhados) e são rejeitados com `InvalidDataException` antes de qualquer estado
ser materializado se retomados com uma estratégia diferente.
