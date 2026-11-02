# -*- coding: utf-8 -*-
"""神经网络执行器：模长 ∈ [0,1] × 物理步幅 ⊙ 场方向（+ 位置维偏转角）。

输入观测（22 维/机，与规模无关）：
    12 维环境观测（env.obs()/OBS_NORM）
    + 5 维场方向 s（3 符号 + 平面单位矢量 2 分量）
    + 5 维饱和残距 clip(|x*−x|/U, 0, 1)

输出（6 维/机）：
    m ∈ (0,1)^5   sigmoid——五维模长（0=冻结，1=满速）
    φ ∈ (−π, π)   tanh×π——位置方向偏转角（场方向为基准的旋转；
                  标量维无对应物，φ 只作用 xy）

合成：a_bpp = m_bpp ⊙ U_bpp ⊙ s_bpp ；a_xy = m_xy·U_xy·R(φ)·s_xy
纯乘法与旋转，数学上恒在物理界内，无 clip。

网络块 = ProjResBlock，逐位沿用旧管线（fc: d_in→d_out、proj 无偏置、
外层 GELU、Dropout(p=0)）；宽度族 (16,32,64,32,16) 同旧。
探索噪声加在 sigmoid 前的 logit 域（保界）。输出偏置初始化为调优
遗迹的等效模长（0.2/0.1/0.98³）——可学习参数的初值，非固定常数：
起步≈调优基线，训练自由移动。
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from config import OBS_NORM
from executors.base import Executor, FieldReading

OBS_DIM = 22
OUT_DIM = 6


class ProjResBlock(nn.Module):
    """投影残差块（组件照旧）：act(proj(x) + drop(act(LN(FC(x)))))。"""

    def __init__(self, d_in: int, d_out: int, p: float = 0.0):
        super().__init__()
        self.fc = nn.Linear(d_in, d_out)
        self.ln = nn.LayerNorm(d_out)
        self.proj = (nn.Identity() if d_in == d_out
                     else nn.Linear(d_in, d_out, bias=False))
        self.drop = nn.Dropout(p)
        self.act = nn.GELU()

    def forward(self, x):
        h = self.drop(self.act(self.ln(self.fc(x))))
        return self.act(self.proj(x) + h)


WIDTHS = (16, 32, 64, 32, 16)


class Net(nn.Module):
    """投影残差块 MLP：22 → 16 → 32 → 64 → 32 → 16 → 6。"""

    def __init__(self):
        super().__init__()
        dims = (OBS_DIM,) + WIDTHS
        self.blocks = nn.Sequential(*[
            ProjResBlock(dims[i], dims[i + 1]) for i in range(len(dims) - 1)])
        self.out = nn.Linear(dims[-1], OUT_DIM)
        nn.init.zeros_(self.out.weight)      # 零权重：起步输出=纯偏置
        # 偏置初值 = 调优遗迹等效模长（logit 域）+ 偏转角 0（不偏转起步）
        bias0 = np.log(np.array([0.2, 0.1, 0.98, 0.98, 0.98]) /
                       (1 - np.array([0.2, 0.1, 0.98, 0.98, 0.98])))
        with torch.no_grad():
            self.out.bias[:5] = torch.as_tensor(bias0, dtype=torch.float32)
            self.out.bias[5] = 0.0

    def forward(self, x):
        return self.out(self.blocks(x))


def make_obs(env, reading: FieldReading) -> np.ndarray:
    o = env.obs() / OBS_NORM[None, :]
    r = np.clip(reading.r_norm, 0.0, 1.0)
    return np.concatenate([o, reading.s, r], axis=1).astype(np.float32)


def compose(m: np.ndarray, phi: np.ndarray, reading: FieldReading) -> np.ndarray:
    """模长×步幅×方向（xy 旋转 φ）。纯乘法，恒在界内。"""
    step, s = reading.step, reading.s
    a = np.empty_like(s)
    a[:, 0] = m[:, 0] * step[0] * s[:, 0]
    a[:, 1] = m[:, 1] * step[1] * s[:, 1]
    a[:, 2] = m[:, 2] * step[2] * s[:, 2]
    c, sn = np.cos(phi), np.sin(phi)
    a[:, 3] = m[:, 3] * step[3] * (s[:, 3] * c - s[:, 4] * sn)
    a[:, 4] = m[:, 4] * step[4] * (s[:, 3] * sn + s[:, 4] * c)
    return a


class NetExecutor(Executor):
    needs_training = True
    name = "net"

    def __init__(self, args=None):
        self.args = args
        self.net = Net()
        self.rng = np.random.default_rng(getattr(args, "seed", 0) if args else 0)

    def act(self, env, reading: FieldReading,
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
            m = torch.sigmoid(u[:, :5]).numpy()
            phi = (np.pi * torch.tanh(u[:, 5])).numpy()
        if was:
            self.net.train()
        return compose(m, phi, reading)
