# -*- coding: utf-8 -*-
"""field.py —— 势场建模的正式实现（2026-10-28 从零重建）。

本文件替代被删除的旧势场实现。全项目的势场只有这一个定义点。

一、为什么要重建
----------------
旧实现对路，位置错。三条实测事实钉死了错在哪：

1. **势函数定义在邻域图上**（250 m 邻接，W_ij 逐边权重）。
   闭式核对：完全图等权时 Φ = ½Σ_{i<j}(û_i−û_j)² = (n/2)Σ(û_i−ū)²，
   与图无关（2026-10-29 复算：相对误差 0，逐位相等）。所以 Φ 本身不需要图。
   把 Φ 定义在邻域图上，梯度就变成「我比**邻居**多过载多少」。

2. **邻域均值 ≠ 全局均值。**
   拉向邻居均值的行归一迭代，稳态是**度加权均值**。真实机队度分布
   不规则（实测 n=12 为 3/5/8），实测残差 n=12 压死在 1.95e-1，
   n=2 单边图周期性振荡永不收敛。全局均值一律由基站回传（见 §三），
   不做任何邻居迭代。

3. **朴素梯度的各向异性 60 倍。**
   ∂Φ/∂b_i = −n(û_i−ū)·m'(u_i)·u_i/b_i，其中 u_i/b_i 在欠载机上是
   0.011、过载机上是 0.65。闭环实测朴素梯度只有 1~2 倍改善（n=12
   六机同亏 5.35→2.61），而换成闭式定点迭代是 730415 倍（5.33→7.3e-6）。

二、正式定义
------------
    观测映射（三档，默认 C）：
        A  û = clip(u,0,3)       u>3 段导数恒 0 → 过载方向全灭（缺陷根源）
        B  û = clip(u,0,30)      同一缺陷，阈值抬高
        C  û = u (u≤3)；3+3ln(1+(u−3)/3) (u>3)
                                   健康带严格恒等、尾部严格单调无封顶

    势函数：
        Φ = (n/2) Σ_i (û_i − ū)²         ū = (1/n)Σ_j û_j

    约束下的全局最小点（闭式，c_i = u_i·b_i 与 b 无关）：
        b*_i = B_total · c_i / Σ_j c_j

    作用律（三档，默认 closed）：
        closed   b ← b + relax·(b* − b)          定点迭代，实测最优
        natgrad  Δb ∝ (û−ū)·b/(m'(u)·u)          自然梯度（除 Hessian 对角）
        grad     Δb ∝ −∂Φ/∂b                      朴素梯度（保留作对照）

三、全局量：基站回传
--------------------
势场里的 ū 不是无人机之间协商出来的，是地面宿主基站 TBS 统计后经回传
链路下发的。TBS 在接入握手里已经看到每架机承载多少用户、多少需求，
O(n) 算完一次广播；回传 100 MHz，一条几十字节的控制消息代价可忽略。

本模型不存在无人机之间的一致性协议，也不存在任何分布式求均值的迭代。
"""
from __future__ import annotations

import numpy as np

from uav_env import (ACT_HIGH, ACT_LOW, B_TOTAL, DB_MAX, DP_MAX, DPI_MAX,
                     PF_RELAX, PF_U_MAP)
import pfmath as P


# ============================ 观测映射 ============================
def u_hat(u: np.ndarray, mode: str | None = None):
    """观测映射，返回 (û, dû/du)。默认取 config.PF_U_MAP。"""
    return P.MAPS[mode or PF_U_MAP](np.asarray(u, dtype=np.float64))


# ============================ 势场 ============================
class GlobalField:
    """全局均衡势场：Φ = (n/2)Σ(û−ū)²，作用于带宽维。

    参数
    ----
    mode    : closed | natgrad | grad（见模块 docstring）
    map_name: A_clip3 | B_clip30 | C_logc（默认 uav_env.PF_U_MAP）
    relax   : 松弛系数（closed/natgrad 用）
    """

    def __init__(self, mode: str = "closed", map_name: str | None = None,
                 relax: float = PF_RELAX, u_star: float | None = None):
        """u_star: 锚定目标负载（None=纯均衡，等分到 Σc/B）。
        给定时按锚定势的闭式解 b*=c/u*，总量不够则按比例缩（退化为等分）。"""
        assert mode in ("closed", "natgrad", "grad"), mode
        self.mode = mode
        self.map_name = map_name or PF_U_MAP
        self.relax = float(relax)
        self.u_star = u_star

    @property
    def target_u(self) -> float:
        """锚定目标（唯一来源）：None 时取 U_STAR_DEFAULT，否则取 u_star。"""
        from uav_env import U_STAR_DEFAULT
        return U_STAR_DEFAULT if self.u_star is None else float(self.u_star)

    def reset(self) -> None:
        pass

    # ---------------- 量 ----------------
    def phi(self, env) -> float:
        """真 Φ（集中式，只用于度量，不是控制器输入）。"""
        ue, _ = u_hat(env.u, self.map_name)
        return float(0.5 * len(ue) * np.sum((ue - ue.mean()) ** 2))

    def mean_phi(self, env) -> np.ndarray:
        """基站回传的 ū（(N,) 全同）。"""
        ue, _ = u_hat(env.u, self.map_name)
        return np.full(len(ue), float(ue.mean()))

    def b_target(self, env) -> np.ndarray:
        """锚定势的闭式均衡：b* = c/u*。

        - u_star=None：纯均衡 b* = B·c/Σc（把 u 全部拉平到 Σc/B），
          基站回传 Σc 后逐机可算。
        - u_star=0.8：锚定 b* = c/u*，逐机可算；总量超 B 时由
          env._project 统一缩到 B，缩完恰好等于 B·c/Σc。
          n12 实测新旧公式逐位差 1.7e-12。
        """
        c = env.u * env.b
        if self.u_star is None:
            # 纯势场消融：基站回传 Σc，b* = (B/n)·c/mean(c)
            cbar = float(c.mean())
            return env.b_total * c / max(len(c) * cbar, 1e-12)
        # 锚定：逐机可算，总量由 env._project（基站协调）归一化
        return c / max(self.u_star, 1e-9)

    def _b_step(self, env) -> np.ndarray:
        """带宽维的场目标，按 mode 分支（Φ 的三种下降口径）。

        closed  : 闭式定点 b*_i = c_i/u*，约束下的全局最小点（零学习参数）。
        natgrad : Δb = (û−ū)·b/(m'(u)·u) —— 朴素梯度除以 Hessian 对角
                  （∂²Φ/∂b² ≈ n(m'u/b)²），即自然梯度。
        grad    : Δb = −∂Φ/∂b / n = (û−ū)·m'(u)·(u/b) —— 朴素梯度。
                  除以 n 是因为 Φ=(n/2)Σ(û−ū)² 的梯度本身带 n 因子；不除
                  则 n200 的步长是 n12 的 16.7 倍、全部顶到 ±5 裁剪界，
                  与 n12 不可比。

        三支都逐机可算：ū 由基站回传，其余全是本机量。
        """
        if self.mode == "closed":
            return self.b_target(env)
        ue, dm = u_hat(env.u, self.map_name)
        ubar = float(ue.mean())
        if self.mode == "natgrad":
            step = (ue - ubar) * env.b / np.maximum(dm * env.u, 1e-12)
        else:                                   # grad
            step = -self.grad_b(env) / max(len(ue), 1)
        return env.b + step

    def grad_b(self, env) -> np.ndarray:
        """∂Φ/∂b（ū 由基站回传）。"""
        ue, dm = u_hat(env.u, self.map_name)
        ubar = self.mean_phi(env)
        return -len(ue) * (ue - ubar) * dm * (env.u / env.b)

    # ---------------- 动作 ----------------
    def target_state(self, env) -> np.ndarray:
        """场的均衡点 x_field (N,5)——执行器框架的唯一动作来源。

        逐维：
            Δb  : b*_i = c_i/u*（u*=None 时等分到 Σc/B；总量不够自动退化）
            Δπ  : 负载约束的对偶价格（π = −μ），按对偶上升走一步：
                  π*_i = clip(π_i + δ·(u* − u_i), −3, 3)
                  来源 Ye et al. arXiv:1205.2833 Eq.(15) 的对偶次梯度
                  μ_j ← μ_j − δ(K_j − Σx_ij)，取负号记法 π = −μ 后符号翻正。
                  过载机 π 下降（让出用户），空闲机 π 上升（招揽用户）。
            Δp  : 维持 u_i ≤ u* 所需的最小功率；无锚定时不动（保持当前功率）
            Δxy : 自己负责用户的加权质心（Lloyd 覆盖势闭式解）

        这是「势场即算法本体」的落点：x_field 完全由场决定、逐机可算、
        零学习参数、与 N 无关。执行器（pf_rl）只能决定**走多快**，
        不能改变**走向哪里**。
        """
        from uav_env import (DXY_MAX, P_MIN, PI_DUAL_STEP, PI_MIN, PI_MAX,
                            U_STAR_DEFAULT)
        n = env.n_uav
        x = np.zeros((n, 5))
        x[:, 0] = self._b_step(env)                     # 带宽维：按 mode 分支
        _u_ref = self.target_u
        # π 是关联的相对偏置：softmax 对常数平移不变，因此每步去掉公共分量，
        # 只保留逐机差异。否则对偶上升会把全员一起推到钳位边界（实测 n12
        # 全员 −2.92），差异被压没。
        # 负载读数走观测映射 û：u>3 之后 A_clip3 导数恒 0（过载方向全灭）、
        # C_logc 严格单调——这是 --map 消融臂唯一能分出胜负的位置。
        ue, _ = u_hat(env.u, self.map_name)
        ue_ref = float(u_hat(np.array([_u_ref]), self.map_name)[0][0])
        _pi_t = env.pi + PI_DUAL_STEP * (ue_ref - ue)
        _pi_t = _pi_t - float(_pi_t.mean())
        x[:, 1] = np.clip(_pi_t, PI_MIN, PI_MAX)        # π 对偶上升一步
        x[:, 2] = env.p if self.u_star is None else 1.0  # 功率（锚定时回到 1.0）
        # 位置：Lloyd 质心（按设定点基，PHYS_HOLD=1 后每步可更新）
        dom, w, upos, base = _lloyd_inputs(env)
        _sw, tgt, ok = _lloyd_centroids(dom, w, upos, n)
        x[:, 3:5] = np.where(ok[:, None],
                             base + np.clip(tgt - base, -DXY_MAX, DXY_MAX),
                             base)
        return x

    def act(self, env, with_pos: bool = True) -> np.ndarray:
        """(N,5) 五维动作：场均衡点 × 固定松弛（纯势场臂，η≡1）。

        这是执行器框架在 η≡1 处的消融：零学习参数。
        """
        x = self.target_state(env)
        cur = np.zeros((env.n_uav, 5))
        cur[:, 0] = env.b
        cur[:, 1] = env.pi
        cur[:, 2] = env.p
        pos_base = (env._phys._setpoint if getattr(env, "_phys", None) is not None
                    else env.pos)
        cur[:, 3:5] = pos_base
        d = x - cur
        # 带宽维与 pfrl.build_action 同口径：先按物理单步上限裁，
        # 再乘松弛系数。反序（先乘 0.2 再裁 ±5）会把 b 维单步上限
        # 从 1.0 放大到 5.0，纯势场臂就拿到了主方法表达不出的幅度。
        # 2026-11-02 rizvi3 尺度修复：上限改 env.b_scale（按 b_total
        # 归一，其余场景 = DB_MAX 逐位不变）；锚定 b*=c/u* 无总量上限，
        # 严重过载场景 b* 可冲到 MHz 量级，全局 ±5 裁剪在 kHz 级
        # 场景里单步 22 倍总带宽 → _project 归一 → bang-bang（docs/03 §11）。
        bs = getattr(env, "b_scale", DB_MAX)
        d[:, 0] = np.clip(d[:, 0], -bs, bs) * self.relax
        a = np.clip(d, ACT_LOW[None, :], ACT_HIGH[None, :])
        if not with_pos:
            a[:, 3:5] = 0.0
        return a

    def act_db(self, env) -> np.ndarray:
        return self.act(env, with_pos=False)[:, 0]


def _lloyd_inputs(env):
    """Lloyd 质心的四个输入：归属、需求权重、用户位置、位置基准。

    **用户位置不是精确全局量**（2026-10-30 用户裁定）：走 `env.lloyd_user_pos()`，
    即延迟 0.5 秒 + 10 米测量误差的版本。历史实测该区间内近似无损、超出才劣化，
    固化为一档默认口径。

    位置基准必须是物理层设定点（PHYS_HOLD 期间实时位置落后设定点，拿实时
    位置做反馈会让设定点越走越远）；无物理层时退化为位置本身。
    """
    dom = env.assoc.argmax(axis=0)
    w = env.dem
    upos = (env.lloyd_user_pos() if hasattr(env, "lloyd_user_pos")
            else env.user_pos)
    base = (env._phys._setpoint if getattr(env, "_phys", None) is not None
            else env.pos)
    return dom, w, upos, base


def _lloyd_centroids(dom, w, user_pos, n_uav):
    """逐机加权质心（向量化）：返回 (权重和, 质心(N,2), 有权重掩码(N,)).

    与原逐机 `dom == i` 循环数学等价；bincount 的求和顺序按用户下标递增，
    与布尔掩码求和同序，机器精度内一致。800 架下把 O(N·M) 的 Python 循环
    换成三次 bincount。
    """
    sw = np.bincount(dom, weights=w, minlength=n_uav)
    safe = np.maximum(sw, 1e-12)
    tx = np.bincount(dom, weights=w * user_pos[:, 0], minlength=n_uav) / safe
    ty = np.bincount(dom, weights=w * user_pos[:, 1], minlength=n_uav) / safe
    return sw, np.stack([tx, ty], axis=1), sw > 1e-12


def b_star_central(env) -> np.ndarray:
    """集中式闭式最优（神谕上界，控制器不可用，只作对照）。"""
    c = env.u * env.b
    return env.b_total * c / c.sum()
