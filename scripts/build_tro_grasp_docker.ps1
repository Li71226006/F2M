param(
  [string]$Image = "f2m-tro-grasp:local"
)

$ErrorActionPreference = "Stop"
$Workspace = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path

docker build `
  -f (Join-Path $Workspace "F2M\related_methods\tro_grasp\Dockerfile") `
  -t $Image `
  $Workspace
