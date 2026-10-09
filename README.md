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
