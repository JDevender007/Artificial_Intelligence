#Requires -Version 5.1
<#
.SYNOPSIS
    Stage and push project notebooks to Kaggle (updates the kernel in kernels.json).

.DESCRIPTION
    For each selected project, copies notebooks/kaggle_run.ipynb into
    tools/kaggle/_stage/<project>/ and writes a kernel-metadata.json that reuses the
    existing kernel id, then optionally runs `kaggle kernels push`.

.PARAMETER Only
    Folder name to push, e.g. 04-intent-routing. Default: every mapped project.

.PARAMETER DryRun
    Write the staging folders and print what would run, without calling Kaggle.

.EXAMPLE
    .\tools\kaggle\push_kernels.ps1 -DryRun
    .\tools\kaggle\push_kernels.ps1 -Only 04-intent-routing
#>
[CmdletBinding()]
param(
    [string]$Only = "",
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$root = Split-Path -Parent (Split-Path -Parent $here)
$mapFile = Join-Path $here "kernels.json"
$stageRoot = Join-Path $here "_stage"

if (-not (Test-Path $mapFile)) { throw "Missing $mapFile" }
$map = Get-Content $mapFile -Raw | ConvertFrom-Json

$selected = @()
foreach ($prop in $map.PSObject.Properties) {
    if ($prop.Name -notmatch '^\d{2}-') { continue }
    if ($Only -and $prop.Name -ne $Only) { continue }
    $gpu = if ($prop.Value.gpu) { "true" } else { "false" }
    $selected += [pscustomobject]@{ Project = $prop.Name; Id = $prop.Value.kernel_id; Gpu = $gpu }
}
if ($selected.Count -eq 0) { throw "No project matched '$Only'. Known: $((($map.PSObject.Properties | Where-Object Name -match '^\d{2}-').Name) -join ', ')" }

foreach ($k in $selected) {
    $projDir = Join-Path $root $k.Project
    $nb = Join-Path $projDir "notebooks\kaggle_run.ipynb"
    if (-not (Test-Path $nb)) { Write-Warning "skip $($k.Project): no notebooks/kaggle_run.ipynb"; continue }

    $stage = Join-Path $stageRoot $k.Project
    New-Item -ItemType Directory -Force -Path $stage | Out-Null
    Copy-Item $nb (Join-Path $stage "kaggle_run.ipynb") -Force

    $meta = [ordered]@{
        id               = $k.Id
        title            = $k.Project
        code_file        = "kaggle_run.ipynb"
        language         = "python"
        kernel_type      = "notebook"
        is_private       = "true"
        enable_gpu       = $k.Gpu
        enable_internet  = "true"
        enable_tpu       = "false"
        dataset_sources  = @()
        competition_sources = @()
        kernel_sources   = @()
    }
    $metaPath = Join-Path $stage "kernel-metadata.json"
    # Kaggle's CLI rejects a UTF-8 BOM, and Set-Content -Encoding UTF8 writes one on Windows PowerShell 5.1.
    $json = $meta | ConvertTo-Json -Depth 4
    [System.IO.File]::WriteAllText($metaPath, $json, (New-Object System.Text.UTF8Encoding($false)))

    Write-Host ("[{0}] gpu={1} id={2}" -f $k.Project, $k.Gpu, $k.Id)
    if ($DryRun) {
        Write-Host ("    staged: {0}" -f $stage)
        continue
    }
    & kaggle kernels push -p $stage
    if ($LASTEXITCODE -ne 0) { Write-Warning "push failed for $($k.Project) (exit $LASTEXITCODE)" }
    Write-Host ("    status: kaggle kernels status {0}" -f $k.Id)
}

Write-Host ""
Write-Host "Done. Watch with: kaggle kernels list --user devenderj"
if ($DryRun) { Write-Host "Dry run only - nothing was pushed." }
