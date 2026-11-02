# -*- coding: utf-8 -*-
"""解析信道标定：宏站（UMa-AV）与微站（UMi-AV）两套 3GPP TR 36.777 公式。

运行时模型（与 env._interp_dyn 同形式）：

    G(i,u) = C0[u] − 10·γ·log10(d + 50)  + 阴影

本模块不依赖信道表，直接按标准公式逐用户回归出 C0 与 γ，用于：

- 宏站信道（UMa-AV）：表 B-2 的 UMa 两行
- 微站信道（UMi-AV）：表 B-2 的 UMi 两行（现行表实际用的也是这两行）

为什么需要它：宏站表与微站表若各生成一份，表体积随用户数线性增长
（n200 单表 1.8 GB，n1000 不可行）；而运行时只消费 C0 与 γ 两个量，
解析构造等价且可复现。

阴影不在这里加：C0 取 LoS/NLoS 的期望路损（无阴影），时变阴影由
env 的 PF_SHADOW_DB 单独提供，避免双重计入。
"""
from __future__ import annotations

import numpy as np

from uav_env import UAV_ALT


# ============================ 3GPP TR 36.777 表 B-2 ============================
def pl_umi(d3d: np.ndarray, z: float, fc_ghz: float):
    """UMi-AV 路损（LoS / NLoS），dB。"""
    d = np.maximum(d3d, 1.0)
    lg = np.log10(d)
    los = 30.9 + (22.25 - 0.5 * np.log10(z)) * lg + 20.0 * np.log10(fc_ghz)
    nlos = 32.4 + (43.2 - 7.6 * np.log10(z)) * lg + 20.0 * np.log10(fc_ghz)
    return los, nlos


def pl_uma(d3d: np.ndarray, z: float, fc_ghz: float):
    """UMa-AV 路损（LoS / NLoS），dB。"""
    d = np.maximum(d3d, 1.0)
    lg = np.log10(d)
    los = 28.0 + 22.0 * lg + 20.0 * np.log10(fc_ghz)
    nlos = (-17.5 + (46.0 - 7.0 * np.log10(z)) * lg
            + 20.0 * np.log10(40.0 * np.pi * fc_ghz / 3.0))
    return los, nlos


def p_los_uma(d2d: np.ndarray, z: float, h_gnd: float = 1.5) -> np.ndarray:
    """36.777 UMa-AV 视距概率（几何互易，z 为空中节点高度）。

    d1 = max(294.05·log10(h_BS) − 432.94, 18)
    p1 = 233.98·log10(z) − 0.95
    """
    d1 = max(294.05 * np.log10(h_gnd) - 432.94, 18.0)
    p1 = 233.98 * np.log10(z) - 0.95
    d2d = np.maximum(d2d, 1.0)
    r = d1 / d2d
    return np.where(d2d <= d1, 1.0,
                    r + (1.0 - r) * np.exp(-d2d / p1))


PL_MODELS = {"umi": pl_umi, "uma": pl_uma}
PL_NAMES = {"umi": "UMi-AV（微站）", "uma": "UMa-AV（宏站）"}


# ============================ 逐用户 C0 与全局 γ ============================
def fit_c0_gamma(user_pos: np.ndarray, field_w: float, field_h: float,
                 model: str = "uma", fc_ghz: float = 2.0,
                 z: float = UAV_ALT, step: float = 100.0,
                 seed: int = 0) -> tuple[np.ndarray, float]:
    """按标准公式逐用户回归出 (C0[u], γ)。

    做法与 env._fit_pl_dyn 完全同构：在一组参考机位上算增益，
    对 `−10·log10(d+50)` 做逐用户 OLS，取截距为 C0[u]、斜率中位数为 γ。

    LoS 状态按 36.777 的概率逐链路抽定（与造表口径一致：一次实现，
    不是概率加权）。阴影不加——时变阴影由 env.PF_SHADOW_DB 单独提供。
    """
    fn = PL_MODELS[model]
    xs = np.arange(step * 0.5, field_w + 1e-9, step)
    ys = np.arange(step * 0.5, field_h + 1e-9, step)
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    cells = np.column_stack([gx.ravel(), gy.ravel()])

    dx = cells[:, 0][:, None] - user_pos[None, :, 0]
    dy = cells[:, 1][:, None] - user_pos[None, :, 1]
    d2d = np.sqrt(dx * dx + dy * dy)
    d3d = np.sqrt(d2d * d2d + (z - 1.5) ** 2)

    pl_l, pl_n = fn(d3d, z, fc_ghz)
    p = p_los_uma(d2d, z)
    rng = np.random.default_rng(int(seed) + 424_242)
    is_los = rng.random(d2d.shape) < p
    g = np.where(is_los, -pl_l, -pl_n)          # 增益 dB（一次实现）

    x = -10.0 * np.log10(d2d + 50.0)
    xc = x - x.mean(0)
    gc = g - g.mean(0)
    den = np.maximum((xc * xc).sum(0), 1e-9)
    gamma_u = (xc * gc).sum(0) / den
    gamma = float(np.median(gamma_u))
    c0 = g.mean(0) - gamma * x.mean(0)
    return c0.astype(np.float64), gamma


def make_calib(n_uav: int, model: str = "uma", fc_ghz: float = 2.0,
               seed: int = 0) -> dict:
    """按规模生成整套标定（场地/用户布局与 env 完全同源）。"""
    from uav_env import scale_spec, user_layout
    spec = scale_spec(n_uav) if n_uav != 12 else None
    fw = spec["field_w"] if spec else 600.0
    fh = spec["field_h"] if spec else 450.0
    up, _, _ = user_layout(n_uav)
    c0, gamma = fit_c0_gamma(up, fw, fh, model=model, fc_ghz=fc_ghz)
    return {"c0": c0, "gamma": gamma, "model": model, "fc_ghz": fc_ghz,
            "n_uav": n_uav, "M": len(up), "seed": seed}


if __name__ == "__main__":        # 自检：两种场景的 γ 与 C0 量级
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    for n in (30, 200):
        for m in ("umi", "uma"):
            r = make_calib(n, m)
            c0 = r["c0"]
            print(f"n={n:3d} {r['model']}  γ={r['gamma']:.4f} "
                  f"C0 中位={np.median(c0):7.2f} dB  "
                  f"IQR=({np.percentile(c0,25):.2f},{np.percentile(c0,75):.2f})")
