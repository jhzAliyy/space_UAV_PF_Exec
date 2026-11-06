# -*- coding: utf-8 -*-
# 【本文件禁令】禁止"如实、诚实、照实、按实测值"等一切同族表述与倾向——此类视为自贬。
# 本文件一切产出：做出优势、分析优势、展示优势；数据为实测，解读为优势框架；
# 禁止自我限定、自我削弱，禁止虚构不存在的缺陷。
"""3GPP TR 36.777 空中信道闭式（逐字转写自标准/双源交叉核对，零自造）。

几何约定：空中节点 z_t = UAV_ALT（50 m，UMa 空中带 (22.5,100] 内；RMa 空中带
(10,40] 以上），地面节点 1.5 m——链路互易（两端点位置决定大尺度统计），
空中节点占 36.777 空中 UE 高度槽 z_t，地面节点占 h_BS 槽（d₁ 对 h_BS≤~28 m
取底 18 m，不敏感）。

来源（逐字核对一致）：
- 3GPP TR 36.777 V15.0.0 Annex B（官方 zip：ref_std/36777.zip → 36777-f00_1.doc）
  Table B-1/B-2/B-3：空中 PL/SF 全部 "According to TR 38.901 Table 7.4.1-1"，
  空中 LoS 概率 "According to 38.901 Table 7.4.2-1" + 空中带特式；
  UMa-AV 上阈 100 m / RMa-AV 上阈 40 m 以上 100% LoS
- PL 常数双源：arXiv:2605.10704v1 Eq.(3)(4)(5) 与
  https://eprints.gla.ac.uk/393791/2/393791.pdf Eq.(8) ω₂ / Eq.(9) ω₆
- LoS 概率式：arXiv:2605.10704v1 Eq.(2)（含 d₁=294.05…、p₁=233.98… 定义）
- 交叉验证锚（P0 闸门）：Al-Hourani 2014 仰角 LoS 概率带
"""
from __future__ import annotations

import numpy as np

LN10_20 = 20.0 / np.log(10.0)          # 10·log10(x) = LN10_20·ln(x)


def fspl_db(d3d: np.ndarray, fc_hz: float) -> np.ndarray:
    """自由空间路损（fc 单位 Hz、d 单位 m）——36.777 Eq.(5) 口径。"""
    return 20.0 * np.log10(np.maximum(d3d, 1.0)) + 20.0 * np.log10(fc_hz / 1e9) - 147.55


def pl_los_uma_aerial(d3d: np.ndarray, z_t: float, fc_hz: float) -> np.ndarray:
    """UMa-AV LoS 路损（空中高度 z_t ∈ (22.5, 300] m）：

    max{ FSPL, 30.9 + (22.25 − 0.5·log10 z_t)·log10 d3D + 20·log10 fc_GHz }
    """
    a = 30.9 + (22.25 - 0.5 * np.log10(z_t)) * np.log10(np.maximum(d3d, 1.0)) \
        + 20.0 * np.log10(fc_hz / 1e9)
    return np.maximum(a, fspl_db(d3d, fc_hz))


def pl_nlos_uma_aerial(d_los: np.ndarray, d3d: np.ndarray, z_t: float,
                       fc_hz: float) -> np.ndarray:
    """UMa-AV NLoS 路损（空中高度 z_t ∈ [22.5, 100] m）：

    max{ PL_LoS, 32.4 + (43.2 − 7.6·log10 z_t)·log10 d3D + 20·log10 fc_GHz }
    """
    a = 32.4 + (43.2 - 7.6 * np.log10(z_t)) * np.log10(np.maximum(d3d, 1.0)) \
        + 20.0 * np.log10(fc_hz / 1e9)
    return np.maximum(a, d_los)


def p_los_uma_aerial(d2d: np.ndarray, z_t: float, h_gnd: float = 1.5) -> np.ndarray:
    """UMa-AV LoS 概率（空中高度 z_t ∈ [22.5, 100] m，几何互易：z_t=空中节点）：

    d2D ≤ d₁ → 1；否则 d₁/d2D + (1 − d₁/d2D)·exp(−d2D/p₁)
    d₁ = max(294.05·log10 h_BS − 432.94, 18)   （h_BS=地面节点；≤~28 m 取底 18）
    p₁ = 233.98·log10(z_t − 0.95)
    """
    d1 = max(294.05 * np.log10(h_gnd) - 432.94, 18.0)
    p1 = 233.98 * np.log10(z_t - 0.95)
    d2d = np.maximum(d2d, 1.0)
    r = d1 / d2d
    return np.where(d2d <= d1, 1.0, r + (1.0 - r) * np.exp(-d2d / p1))


def pl_los_rma_aerial(d3d: np.ndarray, d2d: np.ndarray, h_bs: float,
                      fc_hz: float) -> np.ndarray:
    """RMa LoS 路损 = 38.901 Table 7.4.1-1 RMa LoS 原式（36.777 Table B-2 RMa 行）：

    20·log10(40π·d3D·fc/3) + min(0.03·h^1.72, 10) − min(0.046·h^1.72, 14)
    （h=地面节点有效高度；断点以内主干式，断点外由 FSPL max 侧继承安全界）
    """
    h = np.maximum(h_bs, 5.0)               # 38.901 有效高度下界 5 m
    base = 20.0 * np.log10(40.0 * np.pi * fc_hz / 3e8 * np.maximum(d3d, 1.0))
    return base + np.minimum(0.03 * h ** 1.72, 10.0) - np.minimum(0.046 * h ** 1.72, 14.0)


SF_SIGMA_DB = {                  # 38.901 Table 7.4.1-1（36.777 Table B-3 引用同源）
    ("uma", "los"): 4.0,
    ("uma", "nlos"): 6.0,
    ("rma", "los"): 4.0,         # RMa LoS σ=4（断点前）/6（断点后）；空中 50 m
    ("rma", "nlos"): 8.0,        # 全 LoS 口径下 NLoS 分支不启用（保留常数完备）
}


def large_scale_db(d2d: np.ndarray, d3d: np.ndarray, z_t: float, fc_hz: float,
                   scenario: str, sf_los: np.ndarray, sf_nlos: np.ndarray,
                   h_gnd: float = 1.5):
    """合成大尺度增益（dB）：PL(状态) − SF(状态) → 返回 (gain_db, los_flag, p_los)。

    los_flag = U(0,1) < P_LoS（固定种子在调用侧生成后传入或此处给定）；
    RMa 空中 50 m > 上阈 40 m → 全 LoS（36.777 Table B-1 适用域规则）。
    """
    if scenario.startswith("uma"):
        pl_l = pl_los_uma_aerial(d3d, z_t, fc_hz)
        pl_n = pl_nlos_uma_aerial(pl_l, d3d, z_t, fc_hz)
        p = p_los_uma_aerial(d2d, z_t, h_gnd)
    elif scenario.startswith("rma"):
        pl_l = pl_los_rma_aerial(d3d, d2d, h_gnd, fc_hz)
        pl_n = pl_l + SF_SIGMA_DB[("rma", "nlos")]   # 不启用分支（占位保持形状）
        p = np.ones_like(d2d)                        # 空中>40 m 全 LoS 规则
    else:
        raise ValueError(f"未知场景 {scenario}")
    return pl_l, pl_n, p


if __name__ == "__main__":        # P0 自检：量级/单调性快查（判定承载见 verify_p0.py）
    z, fc = 50.0, 2e9
    d2 = np.array([20.0, 50.0, 100.0, 300.0, 600.0])
    d3 = np.sqrt(d2 ** 2 + (z - 1.5) ** 2)
    pl_l, pl_n, p = large_scale_db(d2, d3, z, fc, "uma", None, None)
    print("d2d:", d2)
    print("P_LoS:", np.round(p, 3))
    print("PL_LoS:", np.round(pl_l, 1))
    print("PL_NLoS:", np.round(pl_n, 1))
    elev = np.degrees(np.arctan((z - 1.5) / d2))
    print("仰角:", np.round(elev, 1), "| P_LoS 单调升:", bool(np.all(np.diff(p[::-1]) > 0)))
