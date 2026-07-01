param(
  [string]$ObjectName = "ycb+baseball",
  [string]$Mode = "4f_no_little",
  [string]$OutputDir = "E:\UM\sparse-finger-grasp-project\F2M\results\related_methods\tro_grasp_local",
  [string]$Source = "dataset",
  [string]$PythonExe = "D:\anaconda3\envs\lerobot2\python.exe"
)

$ErrorActionPreference = "Stop"
$Workspace = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$TroRoot = Join-Path $Workspace "external\TRO-Grasp"

Push-Location $TroRoot
try {
  & $PythonExe .\render_mujoco_sparse_grasps.py `
    --object $ObjectName `
    --modes $Mode `
    --output-dir $OutputDir `
    --source $Source
}
finally {
  Pop-Location
}
