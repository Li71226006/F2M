param(
  [string]$ObjectName = "ycb+baseball",
  [string]$Mode = "4f_no_little",
  [string]$OutputDir = "graph_exp/f2m_responsibility",
  [string]$PythonExe = "D:\anaconda3\envs\lerobot2\python.exe"
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)

Push-Location $Root
try {
  & $PythonExe .\main.py responsibility `
    --config configs/cfet_default.json `
    --object $ObjectName `
    --mode $Mode `
    --output-dir $OutputDir
}
finally {
  Pop-Location
}
