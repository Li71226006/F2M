# TRO-Grasp

这里放 TRO-Grasp 这个外部相关方法的 Docker 和调用配置。它不承载 F2M 自己的算法逻辑。

默认脚本：

```powershell
powershell -ExecutionPolicy Bypass -File F2M\scripts\build_tro_grasp_docker.ps1
powershell -ExecutionPolicy Bypass -File F2M\scripts\run_tro_grasp_docker.ps1 -ObjectName ycb+baseball -Mode 4f_no_little
```

默认输出：

```text
F2M/results/related_methods/tro_grasp
```
