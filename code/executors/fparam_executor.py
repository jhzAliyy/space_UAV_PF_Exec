# -*- coding: utf-8 -*-
"""A2a 场参数化执行器（2026-11-03，猜想二第一级，裁决 P4）。

07_猜想.md A2a：动作空间从逐机 5 维 η 降到 5 个全局场参数——网络
逐机输出 logit（结构与观测照旧），跨机共识（均值）成全局参数，
参数调制执行律本身：

    d_b  = (λ_b·b* − b) · relax         （λ_b：锚定目标缩放）
    d_π  = (π* − π) · relax · g_π       （relax：b/π 松弛，恒等=PF_RELAX）
    d_p  = (1 − p) · g_p                （g_p：功率回标称增益）
    d_xy = (质心 − xy) · g_xy           （g_xy：Lloyd 追踪增益）

参数域（param = mid·exp(amp·x)，x = log(η)/ETA_AMP ∈ [−1,1]，保界；
Net 零初始化输出层 ⇒ η≡1 ⇒ 恒等参数）：
    λ_b   ∈ [0.8, 1.25]   恒等 1
    relax ∈ [0.1, 0.4]    恒等 PF_RELAX(0.2)
    g_π   ∈ [0.5, 2.0]    恒等 1
    g_p   ∈ [0.75, 1.33]  恒等 1
    g_xy  ∈ [0.75, 1.33]  恒等 1

恒等参数下与综合版 net 执行器 η≡1 逐位一致（热启动=纯势场+PF_RELAX）。
P4 判据复用训练日志：η 贴界占比 = 参数贴界占比（映射单调）。

接入（默认路径逐位不变）：
    PF_EXECUTOR=fparam 切换 trainer 的 execute 与 ckpt 名段（learned→a2a）；
    评估 FIELD_ALIAS "a2a"→本执行器。
"""
from __future__ import annotations

import numpy as np
import torch

from config import PF_RELAX
from executors.net_executor import (NetExecutor, make_obs, logits_to_eta,
                                    ETA_AMP)

# 参数中点与半幅（amp = 0.5·ln(hi/lo)，几何对称）
P_MID = np.array([1.0, PF_RELAX, 1.0, 1.0, 1.0])
P_AMP = np.array([0.5 * np.log(1.25 / 0.8),    # λ_b
                  0.5 * np.log(0.4 / 0.1),     # relax
                  0.5 * np.log(2.0 / 0.5),     # g_π
                  0.5 * np.log(1.33 / 0.75),   # g_p
                  0.5 * np.log(1.33 / 0.75)])  # g_xy


def params_from_eta(eta: np.ndarray) -> np.ndarray:
    """η (N,5) → 全局参数 (5,)（跨机均值共识；η∈[exp(−0.7),exp(0.7)]）。"""
    x = np.log(np.maximum(eta, 1e-12)) / ETA_AMP
    return P_MID * np.exp(P_AMP * x.mean(axis=0))


def execute(eta: np.ndarray, reading) -> np.ndarray:
    """参数化执行：全局参数调制目标缩放/松弛/各维增益；clip 物理单步界。

    恒等参数（η≡1）下与 net_executor.execute(η≡1, reading) 逐位一致。"""
    lam, relax, g_pi, g_p, g_xy = params_from_eta(eta)
    t, c = reading.target, reading.cur
    d = t - c
    d[:, 0] = (lam * t[:, 0] - c[:, 0]) * relax
    d[:, 1] *= relax * g_pi
    d[:, 2] *= g_p
    d[:, 3:5] *= g_xy
    return np.clip(d, -reading.step[None, :], reading.step[None, :])


class FparamExecutor(NetExecutor):
    """A2a 执行器：网络/观测照旧（22→5 log-η），执行换参数化律。"""
    needs_training = True
    name = "fparam"

    def act(self, env, reading,
            deterministic: bool = True, explore: float = 0.0) -> np.ndarray:
        obs = torch.as_tensor(make_obs(env, reading))
        was = self.net.training
        self.net.eval()
        with torch.no_grad():
            u = self.net(obs)
            if not deterministic and explore > 0.0:
                u = u + explore * torch.as_tensor(
                    self.rng.standard_normal(tuple(u.shape)),
                    dtype=torch.float32)
            eta = logits_to_eta(u).numpy()
        if was:
            self.net.train()
        return execute(eta, reading)
