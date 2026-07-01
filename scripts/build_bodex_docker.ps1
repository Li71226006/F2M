param(
  [string]$Image = "f2m-bodex:local",
  [string]$Dockerfile = "external\BODex\docker\x86.dockerfile"
)

$ErrorActionPreference = "Stop"
$Workspace = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path

docker build `
  -f (Join-Path $Workspace $Dockerfile) `
  -t $Image `
  (Join-Path $Workspace "external\BODex")
