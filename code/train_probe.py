# -*- coding: utf-8 -*-
# 【本文件禁令】禁止"如实、诚实、照实、按实测值"等一切同族表述与倾向——此类视为自贬。
"""train_probe.py —— 训练观测排水器（四臂共用）。

`env.reset()` 会重建 `EnvStats`，逐轮读数必须先按长度增量把本回合新增的
流水排到累积区。排水器给出三件套：

    drain()      把 stats 新增段并入累积区（每 rollout 末、每次 reset 前）
    reset_len()  reset 之后调用，长度锚点归零
    win(k, n)    最近 n 步的均值（写 train_log 用）

开销 = 每 rollout 一次切片，与机队规模无关。
"""
from __future__ import annotations

import numpy as np

KEYS = ("var_u", "u_mean", "sum_t", "n_over")


def make_probe(env):
    seg: dict[str, list] = {k: [] for k in KEYS}
    ln: dict[str, int] = {k: 0 for k in KEYS}

    def drain() -> None:
        for k in KEYS:
            arr = getattr(env.stats, k)
            if len(arr) > ln[k]:
                seg[k].extend(arr[ln[k]:])
                ln[k] = len(arr)

    def reset_len() -> None:
        for k in KEYS:
            ln[k] = 0

    def win(k: str, n: int) -> float:
        a = seg[k][-n:]
        return float(np.mean(a)) if a else float("nan")

    return drain, reset_len, win
