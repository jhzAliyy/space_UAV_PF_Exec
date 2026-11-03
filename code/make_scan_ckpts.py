# -*- coding: utf-8 -*-
"""A2a 前哨扫描：bias-only 常数网络 ckpt 生成器（零训练参数扫描）。

P1 诊断（2026-11-03）：burst n100 崩溃=总量约束下执行侧单机抢占连锁，
输入侧信息无关。本工具把 Net 输出层压成纯偏置（weight=0, bias=b），
tanh(b) 为常数 ⇒ η 常数 ⇒ fparam 全局参数常数——零训练即可扫
"手工全局阻尼参数"网格，回答 P4 前哨问题：5 参数空间里是否存在
压得住连锁的好点（好点存在 ⇒ A2a 学习版上限高；全灭 ⇒ 表达力不足）。

用法：cd code && python make_scan_ckpts.py ../artifacts/experiments
产出：artifacts/experiments/scan_lb{λ×100}_rl{relax×100}_gp{gπ×100}/ckpt/latest.pt
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

from executors.fparam_executor import P_MID, P_AMP
from executors.net_executor import Net

# 网格：λ_b（b 目标缩放）× relax（b/π 松弛）× g_π（价格步长增益）
GRID = [(lb, rl, gp)
        for lb in (0.85, 1.00)
        for rl in (0.10, 0.15, 0.20, 0.30)
        for gp in (0.5, 1.0)]


def main() -> int:
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
        "../artifacts/experiments")
    for lb, rl, gp in GRID:
        x = np.log(np.array([lb, rl, gp, 1.0, 1.0]) / P_MID) / P_AMP
        b = np.arctanh(np.clip(x, -0.999, 0.999))
        net = Net()
        with torch.no_grad():
            net.out.weight.zero_()
            net.out.bias.copy_(torch.tensor(b, dtype=torch.float32))
        task = f"scan_lb{int(lb * 100)}_rl{int(rl * 100)}_gp{int(gp * 100)}"
        d = root / task / "ckpt"
        d.mkdir(parents=True, exist_ok=True)
        torch.save({"policy": net.state_dict()}, d / "latest.pt")
        print(f"[scan] {task}: params=({lb}, {rl}, {gp}) logit={b.round(3)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
