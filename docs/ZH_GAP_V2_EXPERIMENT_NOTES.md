# Responsibility-Gap v2 初步实验记录

更新时间：2026-07-01

这份记录保存当前 v2 方法的第一轮快速结果。结论先说清楚：代码已经能在 `lerobot2` 环境下跑通，新的 contact/penetration-saturated affinity 也已接入；但 responsibility-gap v2 目前还没有稳定超过 point-tracking baseline，暂时不能作为最终论文主结果。

## 运行环境

- Conda env: `lerobot2`
- Python: 3.12.13
- Torch: 2.11.0+cu126
- CVXPY: 1.9.2

## 输入

BODex official full-static seed:

- `external/DexGraspBench/output/bodex_sparse_official_shadow/graspdata/battery/full_static/0.npy`
- `external/DexGraspBench/output/bodex_sparse_official_shadow/graspdata/mouse/full_static/0.npy`
- `external/DexGraspBench/output/bodex_sparse_official_shadow/graspdata/rubiks/full_static/0.npy`
- `external/DexGraspBench/output/bodex_sparse_official_shadow/graspdata/sodacan/full_static/0.npy`

转换输出：

- `results/gap_v2_inputs/<object>/_converted/q_from_bodex.pt`
- `results/gap_v2_inputs/<object>/_converted/object_pc_normals.pt`
- `results/gap_v2_inputs/<object>/_converted/bridge_meta.json`

## 方法配置

v2:

- `configs/cfet_gap_v2.json`
- `active_objective_mode = responsibility_gap`
- `cycles = 1`
- `qp_iters = 2`
- `palm_qp_iters = 1`

baseline:

- `configs/cfet_default.json`
- `active_objective_mode = point_tracking`

## 新增评估脚本

新增脚本：

- `scripts/evaluate_gap_v2.py`

作用：对已经保存的 `q_result.pt` 重新计算统一指标，而不是只看旧的 disabled residual。

核心指标：

- `legacy_gain`: 旧指标，`start_residual_mass - final_residual_mass`。
- `disabled_kernel_gain`: 只看 disabled responsibility，经 patch compensation kernel 后的缺口下降。
- `target_kernel_gain`: v2 论文目标，active 原始责任保持 + disabled 责任补偿，经 kernel 后的缺口下降。
- `self_retention_after`: active fingers 自己原有责任的保持比例。

运行：

```powershell
conda run -n lerobot2 python scripts\evaluate_gap_v2.py `
  --result-roots results\gap_v2_baseline results\gap_v2_batch `
  --output results\gap_v2_metric_compare.csv `
  --device cpu
```

输出：

- `results/gap_v2_metric_compare.csv`
- `results/gap_v2_metric_compare_pivot.csv`

## 统一指标对比

表中 `target_*` 是当前 v2 最应该看的论文指标，越大越好。

| Object | Mode | legacy baseline | legacy v2 | target baseline | target v2 | self baseline | self v2 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| battery | 4f_no_little | 1.036 | 1.047 | 0.201 | 0.200 | 1.000 | 1.000 |
| battery | 3f_thumb_index_middle | 1.610 | 1.605 | 1.300 | 1.204 | 1.000 | 1.000 |
| battery | 2f_thumb_index | 0.760 | 0.000 | 1.679 | 0.000 | 1.000 | 1.000 |
| mouse | 4f_no_little | 3.750 | 3.714 | 2.072 | 2.072 | 1.000 | 1.000 |
| mouse | 3f_thumb_index_middle | 2.686 | 1.605 | 4.000 | 3.463 | 1.000 | 1.000 |
| mouse | 2f_thumb_index | 0.864 | 0.821 | 3.602 | 3.500 | 1.000 | 1.000 |
| rubiks | 4f_no_little | 1.170 | 0.616 | 1.180 | 1.056 | 1.000 | 1.000 |
| rubiks | 3f_thumb_index_middle | 1.320 | 1.360 | 1.525 | 1.672 | 0.999 | 1.000 |
| rubiks | 2f_thumb_index | 1.020 | 1.019 | 1.620 | 1.618 | 1.000 | 1.000 |
| sodacan | 4f_no_little | 3.780 | 3.768 | 2.831 | 2.623 | 1.000 | 1.000 |
| sodacan | 3f_thumb_index_middle | 1.527 | 1.527 | 1.444 | 1.663 | 1.000 | 1.000 |
| sodacan | 2f_thumb_index | 1.457 | 1.504 | 2.880 | 2.887 | 1.000 | 1.000 |

按 `target_kernel_gain` 统计，v2 只赢了 3/12：

- `rubiks / 3f_thumb_index_middle`
- `sodacan / 3f_thumb_index_middle`
- `sodacan / 2f_thumb_index`

其余 case 是 baseline 更好或基本持平。

## 可视化

代表 case 已渲染：

- 好 case: `results/gap_v2_render/mouse_4f_v2/index.html`
- 失败 case: `results/gap_v2_render/battery_2f_v2/index.html`

对应单模式预览：

- `results/gap_v2_render/mouse_4f_v2/4f_no_little/preview.html`
- `results/gap_v2_render/battery_2f_v2/2f_thumb_index/preview.html`

## 当前判断

1. 代码路径是通的，`lerobot2` 可以跑。
2. affinity 的接触/穿深饱和已经接入，穿深和近接触不会被距离项反向惩罚成低 affinity。
3. 当前 v2 理论表达更符合“少指责任分布逼近五指责任分布”，但数值效果还不稳定。
4. `self_retention_after` 几乎总是 1，说明当前 self-retention 指标太宽松；由于 patch kernel 扩散后很容易覆盖自身责任，它没有真正约束住“别丢掉自己的责任”。
5. `battery / 2f_thumb_index` 是最明显失败样本：v2 的 target kernel gain 为 0，需要重点 debug。

## 下一步

建议先不要扩大到更多物体，而是先调方法本身：

1. 把 acceptance guard 统一改成 `target_kernel_gap`，不要混用旧 residual。
2. 重新设计 self-retention：不要只看 kernel-diffused coverage，可以增加 per-finger raw responsibility retention 或局部 patch retention。
3. 给 responsibility-gap QP 增加一个弱几何方向项，避免 finite-difference responsibility Jacobian 太弱时完全不动。
4. 在 `battery / 2f_thumb_index` 和 `mouse / 3f_thumb_index_middle` 上做小网格调参，再扩展更多小物体。
