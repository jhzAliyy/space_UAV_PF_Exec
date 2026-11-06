# -*- coding: utf-8 -*-
"""旧版（调优遗迹）执行器：2026-10 调参律的逐位显式化，作对照保留。

历史四常数：b ×0.2（TUNED_RELAX）、π 对偶次梯度步 0.05
（TUNED_PI_STEP）、p/xy 顶格。此律生在方向决定器之前——它就是
"场内嵌执行器"时代的产物本身，因此 b/π 两维的目标直接取旧口径
（b* = c/u* 不做总量预缩、π 目标含 0.05 步长），从 env 原量自算，
不经过新场的可行化目标；p/xy 与场目标一致（p*=1.0、Lloyd 质心）。
"""
from __future__ import annotations

import numpy as np

from uav_env import DB_MAX, PI_MAX, PI_MIN, STEP_HIGH, STEP_LOW, U_STAR
from config import TUNED_PI_STEP, TUNED_RELAX
from executors.base import Executor, FieldReading
from field import u_hat


class TunedExecutor(Executor):
    name = "tuned"

    def act(self, env, reading: FieldReading) -> np.ndarray:
        # b：旧锚定目标 c/u*（不预缩），先裁单步上限再乘 0.2
        # （顺序照旧：先裁后乘，反序会放大单步上限）
        bs = getattr(env, "b_scale", DB_MAX)
        b_star = env.u * env.b / max(U_STAR, 1e-9)
        d = reading.target - reading.cur
        d[:, 0] = np.clip(b_star - env.b, -bs, bs) * TUNED_RELAX
        # π：旧对偶次梯度一步（步长 0.05，softmax 平移不变去公共分量）
        ue, _ = u_hat(env.u)
        ue_ref = float(u_hat(np.array([U_STAR]))[0][0])
        pi_t = env.pi + TUNED_PI_STEP * (ue_ref - ue)
        pi_t = pi_t - float(pi_t.mean())
        d[:, 1] = np.clip(pi_t, PI_MIN, PI_MAX) - env.pi
        # p（目标 1.0）与 xy（质心差逐维裁方形）与旧 act 逐位一致
        return np.clip(d, STEP_LOW[None, :], STEP_HIGH[None, :])
