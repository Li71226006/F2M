param(
  [string]$ResultRoot = "graph_exp/f2m_tro_grasp",
  [string]$PythonExe = "D:\anaconda3\envs\lerobot2\python.exe"
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)

Push-Location $Root
try {
  & $PythonExe -c "from visualizer.mujoco_renderer import MujocoRenderer; MujocoRenderer(r'$ResultRoot').make_index()"
}
finally {
  Pop-Location
}
