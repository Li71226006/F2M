param(
  [string]$Image = "f2m-tro-grasp:local",
  [string]$ObjectName = "ycb+baseball",
  [string]$Mode = "4f_no_little",
  [string]$OutputDir = "/workspace/F2M/results/related_methods/tro_grasp",
  [string]$Source = "dataset"
)

$ErrorActionPreference = "Stop"
$Workspace = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path

docker run --rm --gpus all `
  -v "${Workspace}:/workspace" `
  -w /workspace/external/TRO-Grasp `
  $Image `
  python3 render_mujoco_sparse_grasps.py `
    --object $ObjectName `
    --modes $Mode `
    --output-dir $OutputDir `
    --source $Source
