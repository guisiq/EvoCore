# Datasets de benchmark

Estes arquivos são fixtures versionadas dos cenários de benchmark. A execução
do `Evo.Cli` deve ser offline: não há download, atualização automática nem
fallback sintético quando um arquivo está ausente ou inválido.

O arquivo `datasets.json` registra origem, licença, shape e SHA-256. O loader
deve validar o checksum antes de interpretar os dados e falhar explicitamente
em caso de divergência.

## Proveniência e atribuição

- `boston-housing`: Harrison, D. e Rubinfeld, D. L., _Hedonic prices and the
  demand for clean air_, Journal of Environmental Economics and Management 5
  (1978), 81–102. Fonte original StatLib; cópia obtida do OpenML dataset 531,
  registrado como `Public`.
- `pima-diabetes`: proprietários originais National Institute of Diabetes and
  Digestive and Kidney Diseases; doação de Vincent Sigillito. Cópia obtida do
  OpenML dataset 37, registrado como `Public`.
- `wine-quality-red`: Cortez, P.; Cerdeira, A.; Almeida, F.; Matos, T.; Reis,
  J. (2009), _Wine Quality_, UCI Machine Learning Repository,
  <https://doi.org/10.24432/C56S3T>. Licença
  [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/legalcode).

## Uso responsável

- `boston-housing` inclui a variável histórica `B`, construída a partir de uma
  suposição racial problemática. O cenário existe somente para reprodução
  técnica de uma matriz de regressão; não deve ser usado para decisão social,
  inferência causal ou demonstração de equidade.
- `pima-diabetes` representa um grupo demográfico específico e não autoriza
  uso clínico ou generalização populacional.
- `wine-quality-red` requer atribuição aos autores e ao UCI Machine Learning
  Repository conforme CC BY 4.0.

Os cenários devem registrar no resultado o ID do dataset, o checksum observado,
o número de registros realmente usados e a regra determinística de seleção.
