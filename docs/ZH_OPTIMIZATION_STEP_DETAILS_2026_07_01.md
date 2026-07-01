# F2M 逐步优化明细：当前推荐主方法

更新时间：2026-07-01

这份文档把当前最适合作为论文主线的 F2M 方法写清楚：**五指抓取生成少指抓取的责任缺口补偿优化**。它对应当前默认配置 `configs/cfet_default.json`：

```json
"active_objective_mode": "point_tracking"
```

结论先说：从已经跑过的 BODex 小物体实验看，这条 **residual target selection + 多假设单指 QP + acceptance guard** 路线目前最稳。`responsibility_gap` 和 `candidate_pool` 更像实验分支，思想更漂亮，但当前数值效果还没有稳定超过这条主线。

---

## 0. 总体循环

### 输入

- 五指 seed：\(q^5\)
- 当前少指姿态：\(q_t\)
- disabled fingers：\(D\)
- active finger mask
- object point cloud / normals
- object patches：\(\{(x_k,n_k)\}_{k=1}^K\)
- hand URDF / link surface point cloud

### 输出

- 优化后的少指姿态：\(q^{r*}\)
- 每一步日志：`sequence.history`
- 每个 mode 的 `q_result.pt`、`stats.json`、GLB viewer

### 循环结构

```text
for cycle in cycles:
    for active non-thumb finger f:
        1. 重算当前少指责任和 residual responsibility
        2. 为 finger f 选择 residual targets
        3. 对 distal/middle/proximal 分别解单指 QP
        4. 用 hypothesis score 选择候选
        5. 用 acceptance guard 接受或拒绝

    6. thumb 单独 QP
    7. palm/wrist 单独 QP
```

代码位置：

- `method/sequential_qp.py::sequential_qp_refine`
- `method/sequential_qp.py::solve_active_finger_step`

---

## Step 1：责任建模

### 1.1 物体数据处理

物体点云降采样为 object patches：

\[
O_k=(x_k,n_k)
\]

代码：

- `method/object_processor.py::ObjectProcessor`
- `method/point_cloud.py::make_object_patches`

### 1.2 灵巧手数据处理

手部由 URDF、link surface point cloud 和 FK 共同定义：

\[
\mathcal U=(\text{URDF},\{P_i^L\},\text{joint order},\text{FK})
\]

当前姿态下 link surface 点：

\[
y=T_i(q)p,\quad p\in P_i^L
\]

代码：

- `method/robot_processor.py::RobotProcessor`
- `method/hand_model.py::HandModel`
- `method/mesh_utils.py::load_link_geometries`

### 1.3 link-patch affinity

对 link surface 点 \(y\) 和 patch \(k=(x_k,n_k)\)，先定义 signed side：

\[
s=n_k^\top(y-x_k)
\]

当前 affinity 使用接触/穿深饱和：

\[
a(y,k)=
\begin{cases}
1, & s\le d_{\text{contact}}\ \text{and}\ \|y-x_k\|\le r_{\text{contact}}\\
\exp(-\|y-x_k\|^2/\sigma^2), & \text{otherwise}
\end{cases}
\]

这样处理的含义是：

- 近接触和轻微穿深都算满覆盖责任；
- 深穿不会继续加分，只会在后面的 collision penalty / acceptance guard 里受罚；
- \(r_{\text{contact}}\) 防止远处同切平面点被误判成该 patch 的覆盖。

代码：

- `method/graph_utils.py::build_link_patch_graph`

关键参数：

- `affinity_contact_distance = 0.0015`
- `affinity_contact_radius = 0.020`
- `penetration_saturates_affinity = True`

### 1.4 finger responsibility

同一根手指的多个 link 聚合：

\[
A_{f,k}(q)=\sum_{i\in links(f)}A_{i,k}(q)
\]

五指 seed 下：

\[
A^5_{f,k}=A_{f,k}(q^5)
\]

disabled responsibility：

\[
R_D(k)=\sum_{f\in D}A^5_{f,k}
\]

当前少指 active non-thumb coverage：

\[
R^r(k,q_t)=\sum_{f\in active,\ f\neq thumb}A_{f,k}(q_t)
\]

责任缺口：

\[
R_{res}(k,q_t)=\max(0,R_D(k)-R^r(k,q_t))
\]

代码：

- `method/contact_targets.py::compute_finger_responsibility`
- `method/contact_targets.py::link_patch_to_finger_coverage`
- `method/responsibility.py::disabled_responsibility`
- `method/responsibility.py::residual`
- `method/sequential_qp.py::reduced_residual_mass`

---

## Step 2：active finger target selection

这一步不是 QP，而是把 object-side 的责任缺口变成当前手指可执行的少数几何目标。

对当前 active finger \(f\)，先计算每个 patch 的分数：

\[
score_{f,k}
=
R_{res}(k,q_t)
\cdot S_{reach}(f,k)
\cdot S_{part}(part)
\]

其中：

- \(R_{res}(k,q_t)\)：这个 patch 还缺多少 disabled responsibility；
- \(S_{reach}\)：当前手指 link surface 到 patch 的可达性，距离越近越大；
- \(S_{part}\)：distal / middle / proximal 的先验权重。

然后做 Top-NMS：

```text
repeat up to targets_per_finger:
    取 score 最大的 patch
    如果低于 min_target_weight 就停止
    记录该 patch
    抑制 nms_radius 内的邻近 patch
```

目标点放在物体法向外侧：

\[
z_k=x_k+d_{\text{gap}}n_k
\]

每个 target 记为：

\[
m=(f,k,w_m,link,part,candidate,z_m)
\]

这回答了“为什么从责任缺口 patch 里选目标”：因为 \(R_{res}\) 是物体表面上仍未被少指补回来的责任分布。target selection 的作用，是从这个分布里挑出当前手指够得着、值得补、并且空间上分散的少数 patch，交给后续 QP。

代码：

- `method/contact_targets.py::select_targets_for_finger`
- `method/contact_targets.py::target_scoring_distance`
- `method/contact_targets.py::choose_semantic_target_link`

关键参数：

- `targets_per_finger = 4`
- `target_gap = 0.003`
- `target_reach_sigma = 0.070`
- `min_target_weight = 0.02`
- `nms_radius = 0.018`
- `part_assignment_mode = "multi_hypothesis"`
- `contact_part_candidates = ["distal", "middle", "proximal"]`

---

## Step 3：active finger 多假设 QP

### 3.1 输入

- 当前姿态 \(q_t\)
- 五指 reference \(q^5\)
- 当前手指 \(f\)
- targets \(C_f^\star\)
- object point cloud / normals

### 3.2 优化变量

代码使用全维变量：

\[
\Delta q\in\mathbb R^n
\]

但冻结非当前手指：

\[
\Delta q_{\neg f}=0
\]

所以有效变量是：

\[
\Delta q_f
\]

### 3.3 target tracking

对 target \(m\)，由 link/candidate 选出手部 surface 点：

\[
y_m(q_t)=T^{L_i}(q_t)p^\star
\]

一阶线性化：

\[
y_m(q_t+\Delta q)
\approx
y_m(q_t)+J_m\Delta q
\]

目标项：

\[
E_{target}
=
\sum_{m\in C_f^\star}
\lambda_c\max(w_m,w_{\min})
\left\|
y_m(q_t)+J_m\Delta q-z_m
\right\|^2
\]

代码：

- `method/sequential_qp.py::solve_single_finger_qp`
- `method/qp_terms.py::closest_surface_point_linearization`

### 3.4 joint anchor

让当前手指不要偏离五指 seed 太远：

\[
E_{anchor}
=
\lambda_a
\left\|
q_f+\Delta q_f-q_f^5
\right\|^2
\]

### 3.5 step regularization

控制局部线性化步长：

\[
E_{step}=\lambda_s\|\Delta q\|^2
\]

### 3.6 关节限位和步长约束

\[
q_{min}\le q_t+\Delta q\le q_{max}
\]

\[
-\Delta q_{max}\le \Delta q_f\le \Delta q_{max}
\]

代码中写成当前姿态附近的上下界：

\[
\Delta q_f
\le
\min(\Delta q_{max},q_{max}-q_t)
\]

\[
\Delta q_f
\ge
\max(-\Delta q_{max},q_{min}-q_t)
\]

### 3.7 hand-object collision

使用点云 signed-distance proxy：

\[
\phi(y)>0\text{ outside},\quad \phi(y)<0\text{ penetrating}
\]

线性化：

\[
\phi(y+J\Delta q)
\approx
\phi(y)+\nabla\phi(y)^\top J\Delta q
\]

软约束：

\[
\phi(y)+\nabla\phi(y)^\top J\Delta q+s
\ge
-d_{allow},\quad s\ge0
\]

惩罚：

\[
E_{collision}=\lambda_{col}\|s\|^2
\]

当前策略是：优化谁，就主要检查谁的 hand-object penetration；全局穿深仍记录到日志里。

代码：

- `method/qp_terms.py::add_surface_collision_constraints`
- `method/metrics.py::reduced_penetration_stats`

关键参数：

- `surface_collision_points = 48`
- `surface_collision_k = 12`
- `surface_collision_margin = 0.012`
- `allowed_penetration = 0.003`
- `collision_slack_weight = 80000.0`

### 3.8 near-surface band

代码参数叫 `contact_band`，论文里建议写成 **near-surface band**。它不是物理接触力约束，只是防止候选接触点离物体太远：

\[
\phi(y)\le d_{band}+s_{far}
\]

如果启用 penetration side，则同时有：

\[
\phi(y)+s_{pen}\ge-d_{allow}
\]

也就是近似把点约束在：

\[
-d_{allow}\le \phi(y)\le d_{band}
\]

代码：

- `method/qp_terms.py::add_contact_band_constraints`

当前默认配置里 `contact_band = false`，因为它会让部分 BODex seed 变硬、变难解；作为后续实验项保留。

### 3.9 self-collision

当前移动手指点 \(y_i\) 到其他障碍手部点 \(y_j\) 的距离：

\[
d_{ij}=\|y_i-y_j\|
\]

线性化：

\[
d_{ij}+n_{ij}^{\top}J_i\Delta q+s
\ge
d_{self}
\]

惩罚：

\[
E_{self}=\lambda_{self}\|s\|^2
\]

当前范围按用户确认过的版本：

- 优化 active non-thumb 时，主要看它和 thumb、palm 的自碰撞；
- 优化 thumb 时，主要看 thumb 和 active non-thumb、palm 的自碰撞；
- disabled fingers 不参与自碰撞 guard。

代码：

- `method/qp_terms.py::add_self_collision_constraints`

### 3.10 多假设选择

对同一根手指，分别尝试：

\[
part\in\{\text{distal},\text{middle},\text{proximal}\}
\]

每个 part 单独选 targets、解 QP、评估候选，然后用评分选择：

\[
score =
\lambda_R\Delta R
+\lambda_N\Delta near
+\lambda_d\Delta d_f
+\lambda_{fn}\Delta near_f
+\lambda_W\Delta W
-\lambda_P\Delta penetration
-\lambda_q\|\Delta q_f\|
\]

代码：

- `method/sequential_qp.py::hypothesis_score`
- `method/sequential_qp.py::solve_active_finger_step`

---

## Step 4：acceptance guard

QP 解出来不一定接受。acceptance guard 的作用是防止“目标追到了，但整体责任或安全性变差”。

active finger 接受条件主要看：

1. residual responsibility 是否下降或至少没有明显变坏；
2. moving finger 自己是否更接近物体或 near ratio 提升；
3. moving finger 的 penetration 是否没有明显恶化；
4. near ratio 是否没有明显下降。

代码：

- `method/sequential_qp.py::accept_candidate_step`
- `method/sequential_qp.py::evaluate_candidate`

日志字段：

- `accepted`
- `residual_before` / `residual_after`
- `penetration_scope`
- `max_penetration_before_mm` / `max_penetration_after_mm`
- `global_max_penetration_before_mm` / `global_max_penetration_after_mm`
- `moving_finger_gap_before` / `moving_finger_gap_after`
- `wrench_before` / `wrench_after`

---

## Step 5：thumb 单独优化

thumb 不直接承担 active non-thumb 的 residual transfer，但它影响抓取支撑、对向接触和 wrench proxy。

thumb target score：

\[
s_{thumb,k}
=
\lambda_{opp}s_{opp,k}
+\lambda_ns_{normal,k}
+\lambda_ws_{wrench,k}
\]

thumb QP 复用单指 QP 框架：

- 变量：thumb joints；
- 目标：thumb targets tracking；
- 约束：joint limit、step bound、hand-object collision、near-surface band、self-collision。

接受条件：

- thumb penetration 不明显恶化；
- thumb 距离物体变近，或 near ratio 提升，或 opposition/wrench proxy 改善。

代码：

- `method/contact_targets.py::select_thumb_targets`
- `method/sequential_qp.py::accept_thumb_step`
- `method/sequential_qp.py::solve_single_finger_qp`

---

## Step 6：palm/wrist 单独优化

palm/wrist 只做 reach prealignment，不承担 disabled responsibility。

变量：

\[
\Delta z_{palm}
=
[\Delta x,\Delta y,\Delta z,\Delta roll,\Delta pitch,\Delta yaw,\Delta q_{WRJ}]
\]

当前默认不打开 wrist：

```text
palm_phase_include_wrist = False
```

palm reach 目标：

\[
E_{palm-reach}
=
\sum_j
\lambda_{reach}
\left\|
y_j(q)+J_j\Delta z_{palm}-z_j
\right\|^2
\]

其中：

\[
z_j=x_j+d_{target}n_j
\]

代码：

- `method/sequential_qp.py::solve_palm_qp`
- `method/sequential_qp.py::accept_palm_step`

---

## Step 7：wrench proxy 和 bench 指标

当前 F2M 内部没有真实 force closure 求解；wrench 是一个低成本代理：

\[
\omega_k=
\begin{bmatrix}
-n_k\\
(x_k-x_{com})\times(-n_k)
\end{bmatrix}
\]

\[
\bar{\omega}(q)
=
\frac{
\sum_k R^r(k,q)\omega_k
}{
\sum_k R^r(k,q)+\epsilon
}
\]

希望 \(\|\bar{\omega}\|\) 下降，但它只是候选评分和 acceptance 的辅助项。

真正“能不能抓起来”目前主要用 DexGraspBench 复评：

- `sim_success`
- `sim_delta_pos_m`
- `sim_delta_angle_deg`
- `ho_pene_mm`
- `self_pene_mm`
- `contact_dist_mm`

代码：

- `method/contact_targets.py::wrench_balance_stats`
- `scripts/export_to_dexgraspbench.py`
- `scripts/run_dexgraspbench_eval_serial.py`
- `scripts/summarize_dexgraspbench_eval.py`

注意：DexGraspBench 里 `delta_pos=100`、`delta_angle=100` 通常是失败哨兵值，不是真实移动 100 米或 100 度。

---

## 当前主线和实验分支

推荐主线：

- 配置：`configs/cfet_default.json`
- active objective：`point_tracking`
- target assignment：`legacy_nms`
- 方法表述：`residual target selection + multi-hypothesis single-finger QP`

实验分支：

- `configs/cfet_gap_v2.json`
  - 直接优化 responsibility distribution gap；
  - 理论上更贴近“五指责任和少指责任相似”，但当前数值结果不稳定。
- `configs/cfet_candidate_pool_v5.json`
  - residual region + hand candidate pool matching；
  - 目前只作为 ablation / future work，不作为主结果。

当前实验判断：

- strict_v4/默认 point-tracking 路线在四个 BODex 小物体、4F/3F/2F sweep 中整体最稳；
- candidate-pool v5 在 rubiks 个别 case 有改善，但在 battery、sodacan 明显变差；
- hard-contact 版本曾尝试过，但容易让 QP 变硬、不可行或穿深转移，所以没有作为主线保留。

---

## 论文里建议怎么写

Method 主线建议写成四段：

1. **Responsibility Graph Construction**
   - object patches；
   - hand link surface；
   - contact/penetration-saturated affinity；
   - finger responsibility。

2. **Disabled Responsibility Gap**
   - 五指 seed 的 \(A^5\)；
   - disabled responsibility \(R_D\)；
   - 当前少指 coverage \(R^r(q)\)；
   - residual \(R_{res}\)。

3. **Residual Target Assignment**
   - score \(=R_{res}\cdot S_{reach}\cdot S_{part}\)；
   - Top-NMS；
   - \(z_k=x_k+d_{gap}n_k\)；
   - distal/middle/proximal 多假设。

4. **Sequential Constrained QP**
   - 单指 target tracking；
   - joint anchor / step regularization；
   - hand-object penetration；
   - self-collision；
   - thumb support；
   - palm/wrist prealignment；
   - acceptance guard。

这样写的好处是和当前最稳代码完全一致，也能解释你关心的核心问题：少指不是“硬追某个点”，而是通过少数 residual targets 引导当前手指，让整体 disabled responsibility gap 尽量下降，同时不牺牲太多安全性和已有责任。
