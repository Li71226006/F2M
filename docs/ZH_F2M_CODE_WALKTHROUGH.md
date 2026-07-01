# F2M 代码导读与公式主线

更新时间：2026-07-01

这份文档是给接手 F2M 写论文用的。建议先读它，再回到代码里看注释。

## 0. 一句话

F2M 当前实现的是一个“五指抓取生成少指抓取”的责任缺口补偿优化方法：

\[
(\mathcal U,\mathcal O,q^5,D)
\rightarrow (A^5,A^r)
\rightarrow \Delta R(q)
\rightarrow C_f^\star
\rightarrow q^{r*}
\]

其中：

- \(\mathcal U\)：ShadowHand URDF 和 link surface point cloud；
- \(\mathcal O=\{(p_j,n_j)\}\)：物体点云和法向；
- \(q^5\)：外部方法给的五指 seed grasp；
- \(D\)：禁用手指集合；
- \(A^5\)：五指抓取下的 finger-patch responsibility；
- \(A^r\)：少指抓取当前姿态下的 finger-patch responsibility；
- \(\Delta R(q)\)：五指责任和少指责任之间的缺口；
- \(C_f^\star\)：某根 active finger 要追的 patch targets；
- \(q^{r*}\)：最终少指优化姿态。

注意：代码里没有真实接触力求解，所以部分量是几何/接触代理；但方法定位是从五指抓取迁移/优化到少指抓取。

## 1. 代码架构

入口：

- `F2M/main.py`：命令行入口，只负责解析参数并调用 `SequentialQP`。

数据准备：

- `method/config.py`：公开配置，只放少量实验级参数。
- `method/func.py`：把配置扩展成内部 solver args。
- `method/robot_processor.py`：读取 URDF、link 点云和初始 q。
- `method/object_processor.py`、`method/point_cloud.py`：读取物体点云并采样 object patches。
- `method/pose_adapter.py`：外部 BODex 姿态到 F2M q 格式的转换。

Part A：责任分配：

- `method/graph_utils.py`：构建 link-patch soft graph。
- `method/contact_targets.py`：finger responsibility、residual targets、thumb targets。
- `method/responsibility.py`：对外门面，提供 `disabled_responsibility`、`residual`、`select_targets`。

Part B：顺序 QP：

- `method/sequential_qp.py`：主优化流程，active finger / thumb / palm 三阶段。
- `method/qp_terms.py`：QP 里的 target linearization、collision、near-surface band、self-collision。
- `method/metrics.py`：接受/拒绝候选时用的 penetration、near ratio、wrench、opposition 指标。

输出和外部桥接：

- `visualizer/mujoco_renderer.py`：输出 HTML、PLY、GLB 预览。
- `related_methods/bodex/bodex_bridge.py`：把 BODex 输出转换为 F2M 输入。

## 2. Part A 公式：责任从哪里来

### 2.1 物体数据处理：Object patches

物体点云先被降采样成 \(K\) 个 patch：

\[
O_k=(x_k,n_k)
\]

代码：

- `method/object_processor.py::ObjectProcessor`
- `method/point_cloud.py::make_object_patches`

### 2.2 灵巧手数据处理：URDF、link surface、FK

灵巧手一侧先从 URDF 和 link surface point cloud 建立可计算的 hand model：

\[
\mathcal U=(\text{URDF},\{P_i^L\},\text{joint order},\text{FK})
\]

其中：

- \(P_i^L\)：link \(L_i\) 在局部坐标系下的 surface point cloud；
- \(T_i(q)\)：由 FK 得到的 link pose；
- \(T_i(q)P_i^L\)：当前姿态下的 link world surface points。

代码：

- `method/robot_processor.py::RobotProcessor`
- `method/hand_model.py::HandModel`
- `method/mesh_utils.py::load_link_geometries`

这一部分必须放在 link-patch affinity 前面，因为 \(A_{i,k}(q)\) 需要同时知道物体 patch 和手部 link surface。

### 2.3 Link-patch affinity

对每个 hand link \(L_i\) 和 object patch \(O_k\)，用 link surface 点到 patch center 的最近距离：

\[
d_{i,k}(q)=\min_{p\in P_i^L}\|T_i(q)p-x_k\|
\]

方向项：

\[
\alpha_{i,k}=\max(0,n_k^\top u_{k,i})
\]

soft affinity：

\[
A_{i,k}(q)=\exp(-d_{i,k}^2/\sigma^2)\alpha_{i,k}
\]

代码：

- `method/graph_utils.py::build_link_patch_graph`

注意：这里不是直接求真实接触力，也不是二值 contact，而是用几何 proximity + normal consistency 建一个责任代理图。它服务于“五指到少指”的责任差距优化。

### 2.4 Finger responsibility

同一根手指的多个 link 聚合：

\[
A_{f,k}(q)=\sum_{i\in links(f)} A_{i,k}(q)
\]

代码：

- `method/contact_targets.py::link_patch_to_finger_coverage`
- `method/contact_targets.py::compute_finger_responsibility`

### 2.5 五指责任、少指责任和责任缺口

五指 seed 中，每根手指对 object patch 的责任是：

\[
A^5_{f,k}=A_{f,k}(q^5)
\]

少指当前姿态下，active non-thumb 手指的责任是：

\[
A^r_k(q)=\sum_{f\in active,\ f\neq thumb}A_{f,k}(q)
\]

如果只看“被禁用手指原来承担的部分”，可以写成：

\[
R_D(k)=\sum_{f\in D}A_{f,k}(q^5)
\]

当前优化实际追的是这个责任缺口的 residual：

\[
\Delta R(k,q)=R_{res}(k,q)=\max(0,R_D(k)-A^r_k(q))
\]

代码：

- `method/responsibility.py::disabled_responsibility`
- `method/responsibility.py::residual`
- `method/sequential_qp.py::reduced_residual_mass`
- `method/sequential_qp.py::sequential_qp_refine`

重点：`sequential_qp_refine` 在每根 active finger 优化前都会重新算 \(\Delta R(q)\)，所以 target 不是一开始固定死的。顺序优化的目的就是让少指责任 \(A^r(q)\) 尽量补上五指抓取被移除后留下的责任缺口。

### 2.6 Target selection

这一步的作用是把“责任缺口分布”变成“某根手指可以执行的具体几何目标”。

责任缺口 \(R_{res}(k,q)\) 是定义在 object patch 上的标量。它表示：

> 五指抓取中被禁用手指原本覆盖了 patch \(k\)，但当前少指抓取还没有把这个区域补回来。

所以“从责任缺口 patch 中挑目标”的意思不是随便挑物体点，而是优先挑这些 patch：

1. 五指抓取里确实因为禁用手指产生了缺口，即 \(R_{res}(k,q)\) 大；
2. 当前 active finger \(f\) 的某个 link surface 够得着，即距离不太远；
3. 这个 link/part 适合承担这个目标，例如 distal/middle/proximal 先验合理。

对某根 active finger \(f\)，patch \(k\) 的抽象得分是：

\[
score_{f,k}=R_{res}(k,q)\cdot S_{reach}(f,k)\cdot S_{part}(part)
\]

其中：

- \(R_{res}(k,q)\)：这个 patch 还缺多少责任；
- \(S_{reach}(f,k)\)：这根手指到这个 patch 的可达性，代码里主要由 link surface 到 patch 的距离衰减得到；
- \(S_{part}(part)\)：使用 distal/middle/proximal 的先验权重。

被选中的目标不是只保存 patch id，而是一个 `FingerTarget`：

\[
m=(f,k,w_m,L_i,c,z_m)
\]

其中：

- \(f\)：当前要优化的 active finger；
- \(k\)：被选中的 object patch；
- \(w_m\)：这个目标的权重；
- \(L_i\)：具体由这根手指的哪个 link 去追；
- \(c\)：这个 link 上的哪个候选 surface region，例如 distal center；
- \(z_m\)：QP 里要追踪的 3D 目标点。

目标点 \(z_m\) 放在 patch 法向外侧，而不是放在物体内部：

\[
z_m=x_k+d_{gap}n_k
\]

这样做是为了让手指表面靠近物体表面附近，而不是直接追到物体内部导致穿模。

代码：

- `method/contact_targets.py::select_targets_for_finger`

### 2.7 当前 target selection 的性质和局限

当前实现不是 cluster-level contact region optimization，也不是完整的 optimal transport。它更准确的名字应该是：

> greedy residual-to-hand correspondence assignment

也就是：

1. 对当前 finger \(f\)，先计算每个 object patch 的责任缺口 \(R_{res}(k,q)\)；
2. 用 reachability 把这根手指当前不容易够到的 patch 降权；
3. 每次选择 score 最大的 patch；
4. 用 NMS 抑制附近 patch，避免多个目标挤在同一小片区域；
5. 给每个选中的 patch 分配一个 hand-side candidate，即某个 link 和某个 surface region；
6. 后续 QP 让这个 hand-side surface point 去追对应 object-side target。

所以当前代码实际做的是：

\[
C_f^\star
=
\operatorname{TopNMS}_k
\left(
R_{res}(k,q)\cdot Reach_f(k)
\right)
\]

然后再把每个 \(k\) 配成：

\[
m=(f,L_i,c,k,w_m,z_k)
\]

这解释了它为什么有效：它把“少指还缺哪些物体区域”转成了少数局部几何约束，避免把所有 residual patch 都塞进 QP。

但它也有明确局限：

1. 它是点级 patch selection，不是真正的连续接触区域；
2. NMS 只是空间去重，不理解 patch cluster 的形状；
3. 当前 hand part/link assignment 主要由距离、候选优先级和 part prior 决定，不是全局最优分配；
4. 它没有显式建模“一个 residual region 应该由哪个 hand part 贴上去”的区域级对应关系。

如果后续要把方法做得更强，可以把这一节升级成三种版本之一：

**版本 A：保守论文版。**
保留当前代码，把它写成 greedy residual-to-correspondence assignment。优点是和代码完全一致，容易解释实验；缺点是理论创新感较弱。

**版本 B：区域版。**
先对 \(R_{res}(k,q)\) 高的 patches 做 region grouping，例如基于 patch 邻域或半径聚类，得到 residual regions \(\mathcal G_j\)。然后每个 region 选一个代表 target 或多个 anchors。这样论文里可以说补偿的是 residual contact region，而不是离散点。

**版本 C：匹配/运输版。**
把 residual object patches 和 active hand candidates 看成二分图：

\[
object\ patches\ k
\leftrightarrow
hand\ candidates\ h=(f,L_i,c)
\]

用代价

\[
Cost(h,k)
=
\lambda_d d(h,k)
-\lambda_R R_{res}(k,q)
+\lambda_n C_{normal}(h,k)
+\lambda_p C_{part}(h)
\]

做 top-k matching 或 entropic optimal transport。这样就从启发式 Top-NMS 变成更理论化的 responsibility transport / assignment。

当前 F2M 最适合先写版本 A，并在方法或讨论里说明它近似了版本 C 的思想。这样不会过度声称代码没有实现的东西。

## 3. Part B 公式：QP 怎么动关节

### 3.0 v2：直接优化责任缺口，而不是强制追点

当前保留了旧的 point-tracking QP，同时新增了一个可选 v2：

```json
"active_objective_mode": "responsibility_gap"
```

配置文件：

- `configs/cfet_gap_v2.json`

v2 的核心变化是：

> active finger 不再预先指定哪个 link 必须追某个 patch 点，而是移动到一个新姿态，使少指产生的责任分布尽量补偿目标责任分布。

首先定义 patch 补偿核：

\[
K_{a,b}\in[0,1]
\]

表示 source patch \(a\) 上产生的责任，可以多大程度补偿 target patch \(b\) 的责任。当前代码用位置接近、法向相似和 wrench proxy 相似构建：

\[
K_{a,b}
=
\exp
\left(
-\frac{\|x_a-x_b\|^2}{\sigma_x^2}
-\frac{1-n_a^\top n_b}{\sigma_n}
-\lambda_w\frac{\|\omega_a-\omega_b\|^2}{\sigma_w^2}
\right)
\]

代码：

- `method/responsibility_gap.py::build_patch_compensation_kernel`

少指当前责任 \(R^r(q)\) 经过补偿核扩散后：

\[
\hat R^r(q)=K^\top R^r(q)
\]

目标责任分布包含两部分：

\[
T(k)
=
\sum_{f\in active} A_f^5(k)
+
\lambda_D
\sum_{d\in D}A_d^5(k)
\]

也就是：

1. active fingers 自己在五指抓里原本承担的责任；
2. disabled fingers 留下、需要被补偿的责任。

整体责任缺口：

\[
E_{gap}(q)
=
\left\|
\max(0,T-\hat R^r(q))
\right\|^2
\]

对当前 finger \(f\)，代码用 finite difference 估计责任 Jacobian：

\[
G_f(k)=
\frac{\partial A_f(k,q)}{\partial q_f}
\]

然后局部线性化：

\[
\hat R^r(q+\Delta q_f)
\approx
\hat R^r(q)+K^\top G_f\Delta q_f
\]

QP 主目标：

\[
\min_{\Delta q_f,e}
\|e\|^2
\]

\[
e\ge T-\hat R^r(q)-K^\top G_f\Delta q_f,\quad e\ge0
\]

同时加 self-retention，防止当前手指为了补别人而丢掉自己的原始责任：

\[
E_{self}
=
\left\|
\max(0,A_f^5-K^\top A_f(q+\Delta q_f))
\right\|^2
\]

所以 v2 active finger QP 是：

\[
\min_{\Delta q_f}
E_{gap}
+
\lambda_{self}E_{self}
+
E_{anchor}
+
E_{step}
+
E_{safe}
\]

其中 \(E_{safe}\) 仍然包含 hand-object penetration、near-surface band、self-collision、joint limits。

代码：

- `method/responsibility_gap.py`
- `method/sequential_qp.py::solve_single_finger_responsibility_gap_qp`
- `method/sequential_qp.py::sequential_qp_refine`

这版更符合“五指生成少指”的核心目标：不是追踪离散 target 点，而是让少指的新姿态产生与目标责任相似的责任分布。

### 3.1 单根手指 QP

以下是旧的 point-tracking QP，仍保留作为 baseline / fallback。

上一节选出来的 `FingerTarget` 还不是运动结果。真正去追踪目标点 \(z_m\) 的，是 finger \(f\) 上被选中的 link \(L_i\) 的一个 surface point。

代码会在 link \(L_i\) 的候选 surface region \(c\) 里找一个代表点：

\[
y_m(q)=T_i(q)p^\star,\quad p^\star\in P_i^L(c)
\]

直观地说：

> object patch 给出“应该补哪里”，FingerTarget 指定“用哪根手指、哪个 link、哪个候选表面去补”，QP 则让这个 link surface point \(y_m\) 去追踪目标点 \(z_m\)。

变量是全维 \(\Delta q\)，但非当前手指关节被约束为 0：

\[
\Delta q_{\neg f}=0
\]

对 target \(m\)，被选中的 link surface 点一阶线性化：

\[
y_m(q+\Delta q)\approx y_m(q)+J_m\Delta q
\]

target objective：

\[
E_{target}
=
\sum_m \lambda_c w_m
\|y_m(q)+J_m\Delta q-z_m\|^2
\]

再加：

\[
E_{anchor}=\lambda_a\|q_f+\Delta q_f-q^5_f\|^2
\]

\[
E_{step}=\lambda_s\|\Delta q\|^2
\]

代码：

- `method/sequential_qp.py::solve_single_finger_qp`
- `method/qp_terms.py::closest_surface_point_linearization`

### 3.2 Hand-object collision

点云 signed distance proxy：

\[
\phi(y)>0 \text{ outside},\quad \phi(y)<0 \text{ penetrating}
\]

允许轻微穿深：

\[
\phi(y)+\nabla\phi(y)^\top J_y\Delta q+s\ge -d_{allow}
\]

代码：

- `method/qp_terms.py::add_surface_collision_constraints`

### 3.3 Near-surface band

代码参数仍叫 `contact_band`，但论文里建议写成 near-surface band：

\[
\phi(y)\le d_{band}
\]

若同时启用 penetration side：

\[
-d_{allow}\le \phi(y)\le d_{band}
\]

它不是接触力约束，只是防止手指离物体太远。

代码：

- `method/qp_terms.py::add_contact_band_constraints`

### 3.4 Self-collision

当前 moving finger 点 \(y_a\) 到其他 active finger 点 \(y_b\)：

\[
\|y_a-y_b\|\ge d_{self}
\]

线性化后进入 QP，并带 slack 惩罚。

代码：

- `method/qp_terms.py::add_self_collision_constraints`

### 3.5 多假设 active finger

对每根 active finger，分别尝试：

\[
part\in\{\text{distal},\text{middle},\text{proximal}\}
\]

每个 part 解一个 QP rollout，然后按综合分数选：

\[
score =
\lambda_R\Delta R+
\lambda_N\Delta near+
\lambda_d\Delta d_f+
\lambda_{fn}\Delta near_f+
\lambda_W\Delta W-
\lambda_P\Delta penetration-
\lambda_q\|\Delta q_f\|
\]

代码：

- `method/sequential_qp.py::solve_active_finger_step`
- `method/sequential_qp.py::hypothesis_score`

### 3.6 Thumb phase

thumb 不背 residual。它单独做 opposition/support：

\[
s_{thumb,k}
=
\lambda_{opp}s_{opp,k}
+\lambda_ns_{normal,k}
+\lambda_ws_{wrench,k}
\]

代码：

- `method/contact_targets.py::select_thumb_targets`
- `method/sequential_qp.py::accept_thumb_step`

### 3.7 Palm/wrist phase

palm/wrist 不背责任，只做 reach prealignment：

\[
E_{palm}
=
\sum_j
\|y_j(q)+J_j\Delta z_{palm}-
(x_j+d_{target}n_j)\|^2
\]

代码：

- `method/sequential_qp.py::solve_palm_qp`
- `method/sequential_qp.py::accept_palm_step`

## 4. 推荐阅读顺序

1. `F2M/README.md`
   - 先搞清楚输入输出、q 维度、怎么跑。

2. `F2M/docs/ZH_F2M_CODE_WALKTHROUGH.md`
   - 也就是本文，先建立公式和代码地图。

3. `F2M/main.py`
   - 看命令行怎么进入 `SequentialQP`。

4. `method/config.py` 和 `method/func.py::build_solver_args`
   - 看哪些参数是公开配置，哪些是内部 solver knob。

5. `method/object_processor.py`、`method/point_cloud.py`
   - 先看物体点云如何变成 object patches。

6. `method/robot_processor.py`、`method/hand_model.py`、`method/mesh_utils.py`
   - 再看灵巧手 URDF、link mesh/point cloud、FK 和 joint order 怎么准备。

7. `method/graph_utils.py`
   - 看 link-patch graph \(A_{i,k}\) 怎么算。

8. `method/contact_targets.py`
   - 看 \(A_{f,k}\)、target selection、thumb target。

9. `method/responsibility.py`
   - 看 Part A 对外接口。

10. `method/sequential_qp.py`
   - 看完整优化主循环。

11. `method/qp_terms.py`
    - 回头细看每个 QP 约束项。

12. `method/metrics.py`
    - 最后看接受/拒绝和实验指标。

## 5. 当前代码质量判断

整体结构比 6/30 的单文件快照更适合写论文和继续开发：入口、数据处理、责任分配、QP 项、可视化基本拆开了。

目前要注意三点：

1. 这仍然是 point-cloud signed-distance proxy，不是 watertight mesh SDF。
2. `contact_band` 这个名字容易误导，论文里应统一称 near-surface band。
3. wrench 是基于 patch 法向和责任权重的平衡代理，不是严格 force closure。论文写法应强调它是启发式平衡项。

## 6. 写论文时可用的章节骨架

Method 可以写成：

1. Problem Definition
   - 输入 \((\mathcal U,\mathcal O,q^5,D)\)，输出 \(q^{r*}\)。

2. Object and Dexterous-Hand Preprocessing
   - object patch；
   - hand URDF / link surface / FK；

3. Responsibility Graph and Responsibility Gap
   - link-patch affinity；
   - finger responsibility；
   - disabled responsibility \(R_D\)；
   - reduced-hand responsibility \(A^r(q)\)；
   - responsibility gap \(\Delta R(q)\)。

4. Residual Responsibility Targeting
   - \(R_{res}\) / \(\Delta R\)；
   - target selection \(C_f^\star\)；
   - distal/middle/proximal multi-hypothesis。

5. Sequential Local QP Refinement
   - active finger QP；
   - collision / near-surface band / self-collision；
   - thumb opposition；
   - palm/wrist prealignment。

6. Acceptance and Diagnostics
   - residual、near ratio、penetration、wrench proxy；
   - accept/reject guard。
