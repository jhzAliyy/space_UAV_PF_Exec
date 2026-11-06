# -*- coding: utf-8 -*-
"""神经网络执行器（2026-11-03 综合版）：乘性 η ⊙ (场目标 − 当前)。

三版合并定稿：
- 执行器参数化回到桌面版乘性形态：d = η ⊙ (x* − x)，η=exp(±ETA_AMP·tanh)
  硬界 [0.4966, 2.0138]——网络增益与缺口成正比（恢复期冲刺），且
  bias=0 ⇒ 起步 η≡1 = 纯势场逐位等价（天然热启动，确定性执行健康）。
- b/π 维沿用桌面版 PF_RELAX=0.2 定点松弛（1.0 与关联惯性打架）。
- 规模负效应剔除依赖 trainer 的 TD3（双 critic 取 min + 目标平滑 +
  延迟更新）与 C_logc 负载视图（无截断）——保留 workspace 版修复。

输入观测（22 维/机，与规模无关）：
    12 维环境观测（env.obs()/OBS_NORM）
    + 5 维场方向 s
    + 5 维饱和残距 clip(|x*−x|/U, 0, 1)

输出（5 维/机）：log-η（tanh 前的 logit），exp(±0.7·tanh) 后恒在硬界内。
探索噪声加在 logit 域（保界）。网络块 ProjResBlock、宽度族照旧。
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from config import OBS_NORM, PF_RELAX
from executors.base import Executor

OBS_DIM = 22
OUT_DIM = 5
ETA_AMP = 0.7                                   # η = exp(±0.7) = [0.4966, 2.0138]
ETA_MIN = float(np.exp(-ETA_AMP))
ETA_MAX = float(np.exp(+ETA_AMP))


class ProjResBlock(nn.Module):
    """投影残差块（组件照旧）：act(proj(x) + drop(act(LN(FC(x)))))."""

    def __init__(self, d_in: int, d_out: int, p: float = 0.0):
        super().__init__()
        self.fc = nn.Linear(d_in, d_out)
        self.ln = nn.LayerNorm(d_out)
        self.proj = (nn.Identity if d_in == d_out
                     else nn.Linear(d_in, d_out, bias=False))
        self.drop = nn.Dropout(p)
        self.act = nn.GELU()

    def forward(self, x):
        h = self.drop(self.act(self.ln(self.fc(x))))
        return self.act(self.proj(x) + h)


WIDTHS = (16, 32, 64, 32, 16)


class Net(nn.Module):
    """投影残差块 MLP：22 → 16 → 32 → 64 → 32 → 16 → 5（log-η 域）。"""

    def __init__(self):
        super().__init__()
        dims = (OBS_DIM,) + WIDTHS
        self.blocks = nn.Sequential(*[
            ProjResBlock(dims[i], dims[i + 1]) for i in range(len(dims) - 1)])
        self.out = nn.Linear(dims[-1], OUT_DIM)
        nn.init.zeros_(self.out.weight)      # 零权重 + 零偏置：
        nn.init.zeros_(self.out.bias)        # 起步 η≡1 = 纯势场（热启动）

    def forward(self, x):
        return self.out(self.blocks(x))


def logits_to_eta(u: torch.Tensor) -> torch.Tensor:
    """logit (…,5) → η ∈ [ETA_MIN, ETA_MAX]（tanh 域保界）。"""
    return torch.exp(ETA_AMP * torch.tanh(u))


def make_obs(env, reading) -> np.ndarray:
    o = env.obs() / OBS_NORM[None, :]
    r = np.clip(reading.r_norm, 0.0, 1.0)
    return np.concatenate([o, reading.s, r], axis=1).astype(np.float32)


def execute(eta: np.ndarray, reading) -> np.ndarray:
    """d = η ⊙ (x* − x)；b/π 维乘 PF_RELAX；clip 物理单步界。

    η≡1 且 PF_RELAX=1 时逐位等于桌面版纯势场；clip 兜底物理界。"""
    d = eta * (reading.target - reading.cur)
    d[:, 0] *= PF_RELAX
    d[:, 1] *= PF_RELAX
    return np.clip(d, -reading.step[None, :], reading.step[None, :])


class NetExecutor(Executor):
    needs_training = True
    name = "net"

    def __init__(self, args=None):
        self.args = args
        self.net = Net()
        self.rng = np.random.default_rng(getattr(args, "seed", 0) if args else 0)

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
