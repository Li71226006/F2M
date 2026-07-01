param(
  [Parameter(Mandatory = $true)][string]$Urdf,
  [Parameter(Mandatory = $true)][string]$PointCloud,
  [string]$InitialQ = "",
  [string]$RobotLinkPc = "",
  [string]$RobotName = "shadowhand",
  [string]$ObjectName = "object",
  [string]$Mode = "",
  [string[]]$Modes = @("5f_full", "4f_no_little", "3f_thumb_index_middle", "2f_thumb_index"),
  [string]$OutputDir = "results/default",
  [switch]$Render,
  [string]$PythonExe = "D:\anaconda3\envs\lerobot2\python.exe"
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$RenderFlag = if ($Render) { "--render" } else { "--no-render" }
$CommandArgs = @(
  ".\main.py", "sequential_qp",
  "--config", "configs/cfet_default.json",
  "--robot-name", $RobotName,
  "--urdf", $Urdf,
  "--point-cloud", $PointCloud,
  "--object", $ObjectName,
  "--output-dir", $OutputDir,
  $RenderFlag
)
if ($Mode -ne "") {
  $CommandArgs += @("--mode", $Mode)
}
elseif ($Modes.Count -gt 0) {
  $CommandArgs += @("--modes")
  $CommandArgs += $Modes
}
if ($InitialQ -ne "") { $CommandArgs += @("--initial-q", $InitialQ) }
if ($RobotLinkPc -ne "") { $CommandArgs += @("--robot-link-pc", $RobotLinkPc) }

Push-Location $Root
try {
  & $PythonExe @CommandArgs
}
finally {
  Pop-Location
}
