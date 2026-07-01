param(
  [string]$Image = "f2m-bodex:local",
  [string]$OutputDir = "f2m_outputs",
  [string[]]$BodexArgs = @()
)

$ErrorActionPreference = "Stop"
$Workspace = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path

New-Item -ItemType Directory -Force -Path (Join-Path $Workspace "external\BODex\$OutputDir") | Out-Null

docker run --rm --gpus all `
  -v "${Workspace}:/workspace" `
  -w /workspace/external/BODex `
  $Image `
  @BodexArgs
