# BODex 相关方法

这个目录只放 BODex 作为外部方法的 Docker/配置/bridge。F2M 的责任划分和顺序 QP 不写在这里。

## 输出格式

BODex/DexGraspBench 的 `.npy` 抓取文件通常是一个 dict，常见字段：

- `pregrasp_qpos`
- `grasp_qpos`
- `squeeze_qpos`
- `obj_path`
- `obj_scale`
- `obj_pose`
- `scene_path`

ShadowHand 的 qpos 是：

```text
xyz(3) + quat_wxyz(4) + hand_joints(22) = 29
```

F2M 内部需要：

```text
xyz(3) + rpy(3) + WRJ2/WRJ1(2) + hand_joints(22) = 30
```

转换逻辑在 `bodex_bridge.py`。

## 转成 F2M 输入

```powershell
cd E:\UM\sparse-finger-grasp-project\F2M
powershell -ExecutionPolicy Bypass -File .\scripts\convert_bodex_output.ps1 `
  -BodexNpy "path\to\bodex.npy" `
  -Urdf "path\to\shadow_hand_right_extended.urdf" `
  -RobotLinkPc "path\to\shadowhand.pt" `
  -OutputDir results/related_methods/bodex/converted
```

输出：

- `q_from_bodex.pt`
- `object_pc_normals.pt`
- `bridge_meta.json`

## 注意坐标系

BODex 的 MJCF hand root 和 F2M 当前 URDF hand root 不一定是同一个 frame。bridge 默认保持 BODex 原始坐标。

推荐优先使用：

```powershell
-CalibrateHandFrame
```

它会用 BODex MJCF 在同一个 qpos 下算出的手 mesh 中心，去校准 F2M URDF 手模型的 root 平移。`-AlignCenters` 也保留，但它只是调试开关，会把手部点云中心直接对到物体点云中心，不适合作为正式实验结果。
