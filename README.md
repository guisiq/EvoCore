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

`powershell
dotnet restore product/EvoCore/EvoCore.sln --locked-mode
dotnet build product/EvoCore/EvoCore.sln -c Release --no-restore
dotnet test product/EvoCore/tests/Evo.Cli.Tests -c Release --no-restore
dotnet run --project product/EvoCore/src/Evo.Cli -c Release --no-restore -- demo symbolic-regression --seed 42
dotnet run --project product/EvoCore/src/Evo.Cli -c Release --no-restore -- inspect <run-dir>/best-genome.json
dotnet run --project product/EvoCore/src/Evo.Cli -c Release --no-restore -- resume <run-dir>/checkpoint.json
dotnet run --project product/EvoCore/src/Evo.Cli -c Release --no-restore -- benchmark symbolic-regression-30 --out <dir>
`

A demo não usa rede, banco ou GPU. A meta 24/30 do benchmark de 30 seeds é quality gate reportado, não gate técnico.
