param(
  [Parameter(Mandatory = $true)][string]$BodexNpy,
  [Parameter(Mandatory = $true)][string]$Urdf,
  [string]$RobotLinkPc = "",
  [string]$SourceXml = "external/DexGraspBench/assets/hand/shadow/right_hand.xml",
  [ValidateSet("pregrasp_qpos", "grasp_qpos", "squeeze_qpos")]
  [string]$QposKey = "grasp_qpos",
  [string]$RobotName = "shadowhand",
  [string]$ObjectName = "bodex_object",
  [string]$Mode = "",
  [string[]]$Modes = @("5f_full", "4f_no_little", "3f_thumb_index_middle", "2f_thumb_index"),
  [string]$OutputDir = "results/bodex_f2m_run",
  [int]$ObjectPoints = 2048,
  [double[]]$RootXyzOffset = @(0.0, 0.0, 0.0),
  [switch]$AlignCenters,
  [switch]$CalibrateHandFrame,
  [string]$PythonExe = "D:\anaconda3\envs\lerobot2\python.exe"
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$ConvertDir = Join-Path $OutputDir "_converted"

$ConvertArgs = @{
  BodexNpy = $BodexNpy
  Urdf = $Urdf
  SourceXml = $SourceXml
  QposKey = $QposKey
  ObjectPoints = $ObjectPoints
  RootXyzOffset = $RootXyzOffset
  OutputDir = $ConvertDir
  PythonExe = $PythonExe
}
if ($RobotLinkPc -ne "") { $ConvertArgs.RobotLinkPc = $RobotLinkPc }
if ($AlignCenters) { $ConvertArgs.AlignCenters = $true }
if ($CalibrateHandFrame) { $ConvertArgs.CalibrateHandFrame = $true }

& (Join-Path $Root "scripts\convert_bodex_output.ps1") @ConvertArgs

$QPath = Join-Path $Root (Join-Path $ConvertDir "q_from_bodex.pt")
$PcPath = Join-Path $Root (Join-Path $ConvertDir "object_pc_normals.pt")

$RunArgs = @{
  Urdf = $Urdf
  PointCloud = $PcPath
  InitialQ = $QPath
  RobotName = $RobotName
  ObjectName = $ObjectName
  OutputDir = $OutputDir
  Render = $true
  PythonExe = $PythonExe
}
if ($Mode -ne "") { $RunArgs.Mode = $Mode }
elseif ($Modes.Count -gt 0) { $RunArgs.Modes = $Modes }
if ($RobotLinkPc -ne "") { $RunArgs.RobotLinkPc = $RobotLinkPc }

& (Join-Path $Root "scripts\run_f2m.ps1") @RunArgs
