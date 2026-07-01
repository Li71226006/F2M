# F2M 使用说明

F2M 是我们自己的方法工程。外部方法，例如 BODex、TRO-Grasp，只放在 `related_methods/` 下面，通过 Docker 或脚本生成输入；F2M 自己的方法只负责读取机器人模型、物体点云和初始抓取姿态，然后做责任划分和顺序 QP 优化。

## 先看什么

1. `docs/ZH_F2M_CODE_WALKTHROUGH.md`
   - F2M 的公式主线；
   - Part A 责任分配和 Part B 顺序 QP 的代码对应关系；
   - 推荐阅读顺序和论文 Method 章节骨架。

2. `main.py`
   - 命令行入口，理解 `responsibility` 和 `sequential_qp` 两种运行模式。

3. `method/contact_targets.py`、`method/sequential_qp.py`、`method/qp_terms.py`
   - 当前方法最核心的三个文件。

4. `configs/cfet_gap_v2.json`
   - 可选 v2 配置；
   - active finger 阶段直接优化责任缺口，而不是强制追踪离散 patch 点。

## 目录结构

- `main.py`：F2M 主入口，只跑我们自己的方法。
- `configs/`：少量公开配置，不放外部方法的大量参数。
- `method/`：我们自己的算法代码。
- `method/pose_adapter.py`：姿态格式转换，例如 BODex 的 `xyz + quat + joints` 转成 F2M 的 `xyz + rpy + joints`。
- `method/robot_processor.py`：读取 URDF、建立 hand model、读取初始 q。
- `method/object_processor.py`：读取物体点云，生成 object patch。
- `method/responsibility.py`：责任权重、disabled responsibility、residual、target 选择。
- `method/sequential_qp.py`：顺序 QP 主流程。
- `method/geometry.py`、`graph_utils.py`、`kinematics.py`、`hand_model.py`：本地基础工具，不再 import TRO 的 Python 文件。
- `related_methods/`：外部方法的 Docker、配置和 bridge，例如 `related_methods/bodex/bodex_bridge.py`。
- `scripts/`：PowerShell 调用脚本。
- `visualizer/`：HTML/PLY 可视化。
- `results/`：所有输出结果。

## q / qpos 的统一规则

BODex 输出的 `grasp_qpos` 通常是 29 维：

```text
29 = xyz(3) + quat_wxyz(4) + ShadowHand joints(22)
```

F2M 内部使用 30 维：

```text
30 = xyz(3) + roll/pitch/yaw(3) + WRJ2/WRJ1(2) + ShadowHand joints(22)
```

所以 bridge 会做三件事：

1. 四元数 `qw qx qy qz` 转成欧拉角 `roll pitch yaw`。
2. 按关节名重排 22 个手指关节。
3. BODex 没有的 `WRJ2/WRJ1` 默认填 0。

XML/MJCF 现在用于读取外部方法的关节顺序，例如 BODex 的 ShadowHand XML。F2M 当前 QP/FK 后端仍然使用 URDF；如果以后要直接用 MJCF 做 FK/Jacobian，可以在 `robot_processor.py` 下面再加 `MjcfRobotModel`。

## 从 BODex 输出跑 F2M

一键转换并优化：

```powershell
cd E:\UM\sparse-finger-grasp-project\F2M
powershell -ExecutionPolicy Bypass -File .\scripts\run_f2m_from_bodex.ps1 `
  -BodexNpy "E:\UM\sparse-finger-grasp-project\external\DexGraspBench\output\bodex_sparse_official_shadow\graspdata\battery\full_static\0.npy" `
  -Urdf "E:\UM\sparse-finger-grasp-project\external\TRO-Grasp\data\data_urdf\robot\shadowhand\shadow_hand_right_extended.urdf" `
  -RobotLinkPc "E:\UM\sparse-finger-grasp-project\external\TRO-Grasp\data\PointCloud\robot\shadowhand.pt" `
  -ObjectName bodex_battery `
  -Mode 4f_no_little `
  -OutputDir results/bodex_battery_f2m
```

推荐加 `-CalibrateHandFrame`，它会用 BODex MJCF 手模型在同一个 qpos 下的 mesh 中心，校准 F2M URDF 手模型的 root 平移：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_f2m_from_bodex.ps1 `
  -BodexNpy "E:\UM\sparse-finger-grasp-project\external\DexGraspBench\output\bodex_sparse_official_shadow\graspdata\battery\full_static\0.npy" `
  -Urdf "E:\UM\sparse-finger-grasp-project\external\TRO-Grasp\data\data_urdf\robot\shadowhand\shadow_hand_right_extended.urdf" `
  -RobotLinkPc "E:\UM\sparse-finger-grasp-project\external\TRO-Grasp\data\PointCloud\robot\shadowhand.pt" `
  -ObjectName bodex_battery_aligned `
  -Mode 4f_no_little `
  -OutputDir results/bodex_battery_f2m_aligned `
  -CalibrateHandFrame
```

`-AlignCenters` 仍然保留，但只用于调试坐标系：它会平移手的根节点，让手部点云中心靠近物体点云中心。正式实验不要把它当真实结果。`-CalibrateHandFrame` 更合理，因为它对齐的是 BODex 手模型和 F2M 手模型，而不是把手强行对到物体中心。

## 单独转换 BODex 输出

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\convert_bodex_output.ps1 `
  -BodexNpy "path\to\bodex.npy" `
  -Urdf "path\to\shadow_hand_right_extended.urdf" `
  -RobotLinkPc "path\to\shadowhand.pt" `
  -OutputDir results/related_methods/bodex/converted
```

转换后会生成：

- `q_from_bodex.pt`：F2M 可读取的 30 维 q。
- `object_pc_normals.pt`：从 BODex object mesh 采样得到的 `[x,y,z,nx,ny,nz]` 点云。
- `bridge_meta.json`：记录源文件、关节名、维度、是否做了对齐。

## 跑 F2M 自己的方法

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_f2m.ps1 `
  -Urdf "path\to\hand.urdf" `
  -PointCloud "path\to\object_pc_normals.pt" `
  -InitialQ "path\to\q.pt" `
  -ObjectName object_name `
  -Mode 4f_no_little `
  -OutputDir results/my_run `
  -Render
```

责任划分单独看：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\inspect_f2m_responsibility.ps1 `
  -Urdf "path\to\hand.urdf" `
  -PointCloud "path\to\object_pc_normals.pt" `
  -InitialQ "path\to\q.pt" `
  -OutputDir results/my_responsibility
```

## 输出文件

每个 mode 会写到：

```text
F2M/results/<run_name>/<mode>/
```

主要文件：

- `preview.html`：优化前后 HTML 对比图。
- `interactive_viewer.html`：BODex 风格的交互式 viewer，使用 `model-viewer` 加载 GLB。
- `<mode>/<mode>.glb`、`<mode>/<mode>.png`、`<mode>/<mode>.gif`：每个 mode 的 3D/预览资产。
- `before.ply`、`after.ply`：优化前后 3D 点云/mesh 文件。
- `stats.json`：优化前后指标和每一步 QP 记录。
- `responsibility.csv`：C0/Cstar 责任矩阵。
- `q_result.pt`：`q_start` 和 `q_refined`。

结果目录根部还有 `index.html`，可以直接打开它进入对应 mode 的预览。

## 外部方法 Docker

BODex Docker：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\build_bodex_docker.ps1
powershell -ExecutionPolicy Bypass -File .\scripts\run_bodex_docker.ps1 -BodexArgs <BODex自己的命令参数>
```

TRO-Grasp Docker：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\build_tro_grasp_docker.ps1
powershell -ExecutionPolicy Bypass -File .\scripts\run_tro_grasp_docker.ps1 -ObjectName ycb+baseball -Mode 4f_no_little
```

外部方法生成的结果再通过 bridge 转成 F2M 输入。F2M 算法代码不 import 外部方法的内部 Python 文件。
