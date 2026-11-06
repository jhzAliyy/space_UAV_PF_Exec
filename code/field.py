# -*- coding: utf-8 -*-
"""field.py —— 方向决定器（2026-11-03 重建，全项目场的唯一定义点）。

基本设计：**场只输出方向，不输出速度。**
- 场看到的：滞后 0.5 s 的需求快照（谁在哪、多闲多挤）。
- 场算出的：快照下的闭式最优点位置（锚定带宽、对偶价格方向、
  可行功率、Lloyd 质心）——全部保留自势场建模。
- 场输出的：纯方向。标量维（b/π/p）是符号 ±1；位置维是平面单位矢量。
  零距离、零力度、零速度、零参数。

每步走多少（步长）完全归执行器：
    满速基线  full_act  = 物理单步上限 ⊙ 方向          （速度恒最大，裸叠）
    调优基线  tuned_act = 历史常数执行器的显式化        （对照遗迹，四常数）
    学习执行器  a = clip(Λη⊙(X*−X), ±U)，η∈[0.50,2.01]^5 逐维正增益
                            （executors/net_executor.py，唯一合成点）

本文件不含任何"从 ckpt 续跑"路径；训练侧同样只有从零模式。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from uav_env import DB_MAX, PI_MAX, PI_MIN, STEP_LOW, STEP_HIGH, U_STAR
from config import TUNED_RELAX, TUNED_PI_STEP
import pfmath as P


def u_hat(u: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """观测映射 û（唯一口径 C_logc：健康带严格恒等、尾部严格单调无封顶）。"""
    return P.MAPS["C_logc"](np.asarray(u, dtype=np.float64))


@dataclass
class FieldReading:
    """场的输出包：调用时生成，直接喂给执行器模块。

    s      (N,5) 纯方向：标量维 ±1，位置维平面单位矢量
    r_norm (N,5) 归一化残距 |x*−x|/U（观测用；执行器可读，动作不含）
    target (N,5) 闭式目标 x*（旧版执行器需要目标差，保留在包内）
    cur    (N,5) 当前状态（同上）
    step   (5,)  物理单步上限（b 维=env.b_scale 口径）
    """
    s: np.ndarray
    r_norm: np.ndarray
    target: np.ndarray
    cur: np.ndarray
    step: np.ndarray


class Field:
    """方向决定器。无构造参数——场没有可调的东西。"""

    # ---------------- 闭式目标（建模本体，唯一分支） ----------------
    def b_target(self, env) -> np.ndarray:
        """锚定势闭式均衡 b* = c/u*，c = u·b 与 b 无关；总量由基站归一。"""
        c = env.u * env.b
        bstar = c / max(U_STAR, 1e-9)
        s = bstar.sum()
        if s > env.b_total:                    # 总量超 B：按比例缩（=B·c/Σc）
            bstar = bstar * (env.b_total / s)
        return bstar

    def target_state(self, env) -> np.ndarray:
        """快照闭式目标 x*（N,5）：b*/π 次梯度/可行功率/Lloyd 质心。

        π 无快照内最优解（对偶均衡需迭代），其"目标"只是次梯度挪一步的
        落点——方向才是它的严格产出（见 direction）。
        """
        n = env.n_uav
        x = np.zeros((n, 5))
        x[:, 0] = self.b_target(env)
        ue, _ = u_hat(env.u)
        ue_ref = float(u_hat(np.array([U_STAR]))[0][0])
        pi_t = env.pi + (ue_ref - ue)          # 次梯度方向（幅度归执行器）
        pi_t = pi_t - float(pi_t.mean())       # softmax 平移不变，去公共分量
        x[:, 1] = np.clip(pi_t, PI_MIN, PI_MAX)
        x[:, 2] = 1.0                          # 功率回标称（可行目标）
        dom, w, upos, base = _lloyd_inputs(env)
        _, tgt, ok = _lloyd_centroids(dom, w, upos, n)
        x[:, 3:5] = np.where(ok[:, None], tgt, base)
        return x

    # ---------------- 输出包（执行器的唯一输入） ----------------
    def reading(self, env) -> FieldReading:
        """调用时取势场输出：执行器模块拿到包后自行处理再输出动作。"""
        x = self.target_state(env)
        cur = self._current(env)
        step = self._step(env)
        s = np.zeros_like(x)
        s[:, 0] = np.sign(x[:, 0] - cur[:, 0])
        s[:, 1] = np.sign(x[:, 1] - cur[:, 1])
        s[:, 2] = np.sign(x[:, 2] - cur[:, 2])
        dxy = x[:, 3:5] - cur[:, 3:5]
        norm = np.maximum(np.linalg.norm(dxy, axis=1, keepdims=True), 1e-12)
        s[:, 3:5] = dxy / norm
        r = np.abs(x - cur) / step[None, :]
        return FieldReading(s=s, r_norm=r, target=x, cur=cur, step=step)

    # ---------------- 度量 ----------------
    def phi(self, env) -> float:
        """真 Φ（集中式度量，非控制器输入）。"""
        ue, _ = u_hat(env.u)
        return float(0.5 * len(ue) * np.sum((ue - ue.mean()) ** 2))

    # ---------------- 内部 ----------------
    def _current(self, env) -> np.ndarray:
        cur = np.zeros((env.n_uav, 5))
        cur[:, 0] = env.b
        cur[:, 1] = env.pi
        cur[:, 2] = env.p
        cur[:, 3:5] = (env._phys._setpoint
                       if getattr(env, "_phys", None) is not None else env.pos)
        return cur

    def _step(self, env) -> np.ndarray:
        """物理单步上限（b 维 = env.b_scale，2026-11-02 尺度修复口径）。"""
        u = STEP_HIGH.copy().astype(np.float64)
        u[0] = env.b_scale if env.b_scale < DB_MAX else STEP_HIGH[0]
        return u


# ============================ Lloyd 辅助 ============================
def _lloyd_inputs(env):
    """归属、需求权重、用户位置（滞后 0.5s+10m 口径）、位置基准（设定点）。"""
    dom = env.assoc.argmax(axis=0)
    w = env.dem
    upos = (env.lloyd_user_pos() if hasattr(env, "lloyd_user_pos")
            else env.user_pos)
    base = (env._phys._setpoint if getattr(env, "_phys", None) is not None
            else env.pos)
    return dom, w, upos, base


def _lloyd_centroids(dom, w, user_pos, n_uav):
    """逐机加权质心（向量化，与逐机循环数学等价）。"""
    sw = np.bincount(dom, weights=w, minlength=n_uav)
    safe = np.maximum(sw, 1e-12)
    tx = np.bincount(dom, weights=w * user_pos[:, 0], minlength=n_uav) / safe
    ty = np.bincount(dom, weights=w * user_pos[:, 1], minlength=n_uav) / safe
    return sw, np.stack([tx, ty], axis=1), sw > 1e-12
