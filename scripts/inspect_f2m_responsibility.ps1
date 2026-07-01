param(
  [Parameter(Mandatory = $true)][string]$Urdf,
  [Parameter(Mandatory = $true)][string]$PointCloud,
  [string]$InitialQ = "",
  [string]$RobotLinkPc = "",
  [string]$RobotName = "shadowhand",
  [string]$ObjectName = "object",
  [string]$Mode = "4f_no_little",
  [string]$OutputDir = "results/responsibility",
  [string]$PythonExe = "D:\anaconda3\envs\lerobot2\python.exe"
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$CommandArgs = @(
  ".\main.py", "responsibility",
  "--config", "configs/cfet_default.json",
  "--robot-name", $RobotName,
  "--urdf", $Urdf,
  "--point-cloud", $PointCloud,
  "--object", $ObjectName,
  "--mode", $Mode,
  "--output-dir", $OutputDir
)
if ($InitialQ -ne "") { $CommandArgs += @("--initial-q", $InitialQ) }
if ($RobotLinkPc -ne "") { $CommandArgs += @("--robot-link-pc", $RobotLinkPc) }

Push-Location $Root
try {
  & $PythonExe @CommandArgs
}
finally {
  Pop-Location
}
