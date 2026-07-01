param(
  [Parameter(Mandatory = $true)][string]$BodexNpy,
  [Parameter(Mandatory = $true)][string]$Urdf,
  [string]$RobotLinkPc = "",
  [string]$SourceXml = "external/DexGraspBench/assets/hand/shadow/right_hand.xml",
  [ValidateSet("pregrasp_qpos", "grasp_qpos", "squeeze_qpos")]
  [string]$QposKey = "grasp_qpos",
  [int]$ObjectPoints = 2048,
  [double[]]$RootXyzOffset = @(0.0, 0.0, 0.0),
  [switch]$AlignCenters,
  [switch]$CalibrateHandFrame,
  [string]$OutputDir = "results/related_methods/bodex/converted",
  [string]$PythonExe = "D:\anaconda3\envs\lerobot2\python.exe"
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$CommandArgs = @(
  ".\related_methods\bodex\bodex_bridge.py",
  "--bodex-npy", $BodexNpy,
  "--urdf", $Urdf,
  "--source-xml", $SourceXml,
  "--qpos-key", $QposKey,
  "--object-points", "$ObjectPoints",
  "--root-xyz-offset", "$($RootXyzOffset[0])", "$($RootXyzOffset[1])", "$($RootXyzOffset[2])",
  "--output-dir", $OutputDir
)
if ($RobotLinkPc -ne "") { $CommandArgs += @("--robot-link-pc", $RobotLinkPc) }
if ($AlignCenters) { $CommandArgs += @("--align-centers") }
if ($CalibrateHandFrame) { $CommandArgs += @("--calibrate-hand-frame") }

Push-Location $Root
try {
  & $PythonExe @CommandArgs
}
finally {
  Pop-Location
}
