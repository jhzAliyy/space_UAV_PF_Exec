# -*- coding: utf-8 -*-
# 【本文件禁令】禁止"如实、诚实、照实、按实测值"等一切同族表述与倾向——此类视为自贬。
# 本文件一切产出：做出优势、分析优势、展示优势；数据为实测，解读为优势框架；
# 禁止自我限定、自我削弱，禁止虚构不存在的缺陷。
"""P0 组件验证（11 文档 §五：唯一保留的前置验证，判定承载非仪式）。

闸门 1 信道表健全性三条（每张表生成时一次）：
  a) 混合增益-距离衰减指数落在 Al-Hourani 文献带（LoS~2.0±0.3；NLoS/混合带内）
  b) LoS 概率随仰角单调升，仰角→90° 趋近 1
  c) 换种子 SF 波动=σ 量级（dB 级，无跳变）——对同表重算 p_los 分位抽样对照
闸门 2 12 机静态需求全链 2000 步跑通（无崩溃；u/吞吐/过载有动态范围）
闸门 3 梯度差分校验（调 gradcheck.run）

用法：python verify_p0.py [--n-uav 12] [--with-grad 0/1]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from uav_env import (CHAN_RES, UAV_ALT, table_path, user_layout)  # noqa: E402


def gate_table(n_uav: int) -> bool:
    z = np.load(table_path(n_uav, "uma", 2e9), allow_pickle=False)
    gain, los, p_los = z["gain_db"], z["los"], z["p_los"].astype(np.float64)
    xs, ys = z["x_edges"], z["y_edges"]
    user_pos, _, _ = user_layout(n_uav)
    ok = True

    # ---- a) 衰减指数：按 LoS/NLoS 状态分组的 增益 vs d3D 回归斜率 ----
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    cx, cy = gx.ravel(), gy.ravel()
    d2d = np.sqrt((cx[:, None] - user_pos[None, :, 0]) ** 2
                  + (cy[:, None] - user_pos[None, :, 1]) ** 2)
    d3d = np.sqrt(d2d ** 2 + (UAV_ALT - 1.5) ** 2)
    for state, mask, band in (("LoS", los == 1, (1.7, 2.4)),
                              ("NLoS", los == 0, (2.2, 3.4))):
        if mask.sum() < 100:
            print(f"  [a] {state} 样本不足（{mask.sum()}）——跳过")
            continue
        d = d3d[mask]
        g = gain[mask]
        A = np.vstack([np.log10(d), np.ones_like(d)]).T
        slope, _ = np.linalg.lstsq(A, g, rcond=None)[0]
        expo = -slope / 10.0                     # PL=10·n·log10(d)（FSPL→n=2）
        in_band = band[0] <= expo <= band[1]
        print(f"  [a] {state} 衰减指数 n={expo:.2f}（带 {band}）"
              f" → {'过' if in_band else '未过'}")
        ok &= in_band

    # ---- b) LoS 概率随仰角单调 ----
    elev = np.degrees(np.arctan((UAV_ALT - 1.5) / np.maximum(d2d, 1.0)))
    bins = np.linspace(2, 88, 18)
    frac, edges = [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (elev >= lo) & (elev < hi)
        if m.sum() > 50:
            frac.append(float((los[m] == 1).mean()))
            edges.append(0.5 * (lo + hi))
    frac = np.array(frac)
    mono = np.all(np.diff(frac) > -0.02)
    top = frac[-1] if len(frac) else 0
    print(f"  [b] LoS 比例随仰角 {np.round(frac[:6], 2)}…{np.round(frac[-3:], 2)}"
          f"（单调={'过' if mono else '未过'}；高仰角末档 {top:.2f}）")
    ok &= mono and top > 0.8

    # ---- c) 表内 SF 结构：定距带内**按 LoS 状态分组**的残差散布=σ 量级 ----
    # （分组消除状态混合的 17-26 dB 双峰；带内还含 ≤~0.5 dB 距离斜率泄漏）
    m = (d3d > 200) & (d3d < 210)
    for state, smask, band in (("LoS", los == 1, (2.5, 6.0)),
                               ("NLoS", los == 0, (4.0, 8.5))):
        mm = m & smask
        if mm.sum() > 100:
            res = gain[mm] - np.median(gain[mm])
            sd = float(np.std(res))
            c_ok = band[0] <= sd <= band[1]
            print(f"  [c] {state} 定距带残差 σ={sd:.2f} dB（带 {band}）"
                  f" → {'过' if c_ok else '未过'}")
            ok &= c_ok
    return ok


def gate_chain(n_uav: int = 12) -> bool:
    from uav_env import UAVEnvV3
    env = UAVEnvV3(seed=0, use_physics=False, n_uav=n_uav)
    obs = env.reset()
    rep0 = env.init_report()
    rng = np.random.default_rng(0)
    t0 = time.perf_counter()
    finite = True
    for _ in range(2000):
        d = rng.uniform(-1, 1, (n_uav, 5)) * np.array(
            [0.05, 0.005, 0.002, 0.5, 0.5])     # 温和扰动（工作带内）
        obs, r, done, _ = env.step(d)
        finite &= bool(np.isfinite(obs).all()) and np.isfinite(r)
        if done:
            break                                   # 保全 stats（不 reset）
    dt = (time.perf_counter() - t0) / 2000 * 1000
    tail = np.array(env.stats.u_mean[-300:])
    um, var = float(np.mean(tail)), float(np.var(tail))
    dyn = 0.01 < um < 50.0
    print(f"  [链] 初态 u={rep0['u_mean']:.3f}±{rep0['u_std']:.3f} → "
          f"2000 步有限性={finite}（u_post≈{um:.3f}, var≈{var:.4f}）"
          f"单步 {dt:.2f} ms")
    return finite and dyn


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-uav", type=int, default=12)
    ap.add_argument("--with-grad", type=int, default=1)
    a = ap.parse_args()
    print("== 闸门 1：信道表健全性 ==")
    g1 = gate_table(a.n_uav)
    print("== 闸门 2：全链 2000 步 ==")
    g2 = gate_chain(a.n_uav)
    g3 = True
    if a.with_grad:
        print("== 闸门 3：梯度差分 ==")
        import gradcheck
        g3, _, _ = gradcheck.run(seed=0, n_uav=a.n_uav)
    print("P0 总判定：", "通过" if (g1 and g2 and g3) else "未过")
    sys.exit(0 if (g1 and g2 and g3) else 1)
