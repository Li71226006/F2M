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
4. 初版 `self_retention_after` 几乎总是 1，说明 kernel-diffused self-retention 太宽松；后续 v2b 已改成 raw patch-level retention。
5. 初版 `battery / 2f_thumb_index` 是最明显失败样本：v2 的 target kernel gain 为 0；v2b 通过弱几何方向项把它修到非零改进，但仍低于 baseline。

## v2b 更新

v2b 改动：

- `self_retention` 改成 raw patch-level，不再用 compensation kernel 扩散后的 coverage。
- responsibility-gap QP 增加弱几何方向项：从当前 positive gap 中选少量 reachable patch，作为 soft objective 给手指一个移动方向，但不作为硬约束。
- 新增参数：
  - `responsibility_direction_weight = 30.0`
  - `responsibility_direction_targets = 2`

v2b 批量输出：

- `results/gap_v2b_batch/<object>/<mode>/stats.json`
- `results/gap_v2b_metric_compare.csv`
- `results/gap_v2b_metric_compare_pivot.csv`

按 `target_kernel_gain`，v2b 相比 baseline 赢了 5/12：

| Object | Mode | target baseline | target v2b | self baseline | self v2b | Winner |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| battery | 4f_no_little | 0.177 | 0.161 | 0.977 | 0.938 | baseline |
| battery | 3f_thumb_index_middle | 1.583 | 1.546 | 0.900 | 0.893 | baseline |
| battery | 2f_thumb_index | 1.432 | 1.294 | 0.970 | 0.970 | baseline |
| mouse | 4f_no_little | 1.914 | 0.891 | 0.960 | 0.963 | baseline |
| mouse | 3f_thumb_index_middle | 4.200 | 4.319 | 0.930 | 0.925 | v2b |
| mouse | 2f_thumb_index | 3.414 | 3.383 | 0.887 | 0.887 | baseline |
| rubiks | 4f_no_little | 1.284 | 1.373 | 0.985 | 0.917 | v2b |
| rubiks | 3f_thumb_index_middle | 1.584 | 1.610 | 0.902 | 0.902 | v2b |
| rubiks | 2f_thumb_index | 1.567 | 1.567 | 0.963 | 0.963 | tie |
| sodacan | 4f_no_little | 2.943 | 1.027 | 0.968 | 0.982 | baseline |
| sodacan | 3f_thumb_index_middle | 1.369 | 1.566 | 0.991 | 0.996 | v2b |
| sodacan | 2f_thumb_index | 3.412 | 3.832 | 0.993 | 0.993 | v2b |

## 下一步

已完成：

1. acceptance guard 已以 `target_kernel_gap` 为 active-finger v2 接受条件。
2. self-retention 已改成 raw responsibility retention。
3. responsibility-gap QP 已增加弱几何方向项。

仍建议继续：

1. 对 `responsibility_direction_weight` 和 `responsibility_self_weight` 做小网格调参。
2. 对 `mouse / 4f_no_little`、`sodacan / 4f_no_little` 这类 v2b 退化 case 单独渲染检查。
3. 如果要写论文主结果，建议把 baseline、v2、v2b 同时放进 ablation，而不是只报 v2b。
