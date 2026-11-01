# -*- coding: utf-8 -*-
"""势场数学对象（手写，不依赖任何被删除的实现，不import torch）。

定义
----
    Φ(u) = ½ Σ_{(i,j)∈E} W_ij (û_i − û_j)²

    û_i = m(u_i)          ← 观测映射（三档候选，见 MAPS）
    E   = { (i,j) : 机间距离 ≤ NEIGH_DIST }      ← env.adj 直接读

梯度闭式（u 是 b 的函数：u_i = c_i / b_i，c_i 与 b 无关）
----
    ∂Φ/∂û_i  = Σ_j W_ij (û_i − û_j)
    ∂û_i/∂u_i = m'(u_i)
    ∂u_i/∂b_i = −c_i / b_i²  =  −u_i / b_i
    ⇒ ∂Φ/∂b_i = −m'(u_i) (u_i/b_i) Σ_j W_ij (û_i − û_j)

场动作：Δb = −η · ∂Φ/∂b       （沿势能下降方向）

靶子（约束 Σb=B 下的全局最小点，闭式）
----
    b_i* = B_total · c_i / Σ_j c_j
    c_i = u_i · b_i           （环境自吐的量，实测与 b 无关，漂移 ≤2.7e-16）

三档观测映射
----
    A  clip(u,0,3)   u>3 段导数恒 0 → 梯度恒 0（次序全丢，缺陷根源）
    B  clip(u,0,30)  u>30 段导数恒 0 → 同上，只是阈值抬高
    C  对数拐点      u≤3 严格恒等；u>3 段 3+3ln(1+(u−3)/3)，导数 1/(1+(u−3)/3)
                     → 严格单调、不封顶
"""
from __future__ import annotations

import numpy as np

from uav_env import NEIGH_DIST


# --------------------------- 观测映射 ---------------------------
def map_a(u):
    """A：现行 clip(u,0,3)。返回 (û, dû/du)。"""
    u = np.asarray(u, dtype=np.float64)
    ue = np.clip(u, 0.0, 3.0)
    d = ((u > 0.0) & (u < 3.0)).astype(np.float64)
    return ue, d


def map_b(u):
    """B：clip(u,0,30)。"""
    u = np.asarray(u, dtype=np.float64)
    ue = np.clip(u, 0.0, 30.0)
    d = ((u > 0.0) & (u < 30.0)).astype(np.float64)
    return ue, d


def map_c(u):
    """C：对数拐点。u≤3 恒等；u>3 → 3+3ln(1+(u−3)/3)，导数 1/(1+(u−3)/3)。"""
    u = np.asarray(u, dtype=np.float64)
    hi = u > 3.0
    ue = np.where(hi, 3.0 + 3.0 * np.log1p((u - 3.0) / 3.0), u)
    d = np.where(hi, 1.0 / (1.0 + (u - 3.0) / 3.0), 1.0)
    return ue, d


MAPS = {"A_clip3": map_a, "B_clip30": map_b, "C_logc": map_c}


# --------------------------- 图与势能 ---------------------------
def edges_of(pos: np.ndarray, neigh: float = NEIGH_DIST) -> np.ndarray:
    """无向边表 (E,2)，i<j，由机位与邻域半径确定（与 env.adj 同源）。"""
    n = pos.shape[0]
    d = np.linalg.norm(pos[:, None] - pos[None], axis=2)
    return np.array([(i, j) for i in range(n) for j in range(i + 1, n)
                     if d[i, j] <= neigh and d[i, j] > 0.0], dtype=int).reshape(-1, 2)


def phi(ue: np.ndarray, E: np.ndarray, W: np.ndarray | None = None) -> float:
    """Φ = ½ Σ_{(i,j)∈E} W_ij (û_i−û_j)²。W 缺省等权 1。"""
    if E.size == 0:
        return 0.0
    i, j = E[:, 0], E[:, 1]
    w = np.ones(len(E)) if W is None else np.asarray(W, dtype=np.float64)
    return float(0.5 * np.sum(w * (ue[i] - ue[j]) ** 2))


def dphi_due(ue: np.ndarray, E: np.ndarray, W: np.ndarray | None = None) -> np.ndarray:
    """∂Φ/∂û_i = Σ_j W_ij (û_i − û_j)（闭式，对称图）。

    推导：Φ=½Σ W(u_i−u_j)²，边 (i,j) 对 u_i 的偏导为 ½·W·2(u_i−u_j)=W(u_i−u_j)；
    u_i 出现在所有与之相连的边上，符号随它落在边的哪一端而变。
    """
    n = ue.shape[0]
    g = np.zeros(n)
    if E.size == 0:
        return g
    i, j = E[:, 0], E[:, 1]
    w = np.ones(len(E)) if W is None else np.asarray(W, dtype=np.float64)
    diff = ue[i] - ue[j]
    np.add.at(g, i, w * diff)
    np.add.at(g, j, -w * diff)
    return g


def dphi_db(u: np.ndarray, b: np.ndarray, E: np.ndarray, map_fn,
            W: np.ndarray | None = None) -> np.ndarray:
    """∂Φ/∂b_i = −m'(u_i)·(u_i/b_i)·Σ_j W_ij(û_i−û_j)。

    若 m'=0（截断失效区）则该维梯度恰为 0——这就是缺陷的数学位置。
    """
    ue, dmdu = map_fn(u)
    g = dphi_due(ue, E, W)
    return -g * dmdu * (u / b)


def b_star(u: np.ndarray, b: np.ndarray, b_total: float) -> np.ndarray:
    """闭式最优带宽分配：b* = B·c/Σc，c=u·b。"""
    c = u * b
    return b_total * c / c.sum()


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """两向量夹角余弦（任一为零向量时返回 nan，不伪装成 0）。"""
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-30 or nb < 1e-30:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


def _rank(x: np.ndarray) -> np.ndarray:
    """平均秩（并列取均值），不依赖 scipy。"""
    x = np.asarray(x, dtype=np.float64)
    order = np.argsort(x, kind="mergesort")
    r = np.empty(len(x), dtype=np.float64)
    r[order] = np.arange(1, len(x) + 1, dtype=np.float64)
    xs = x[order]
    i = 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[j + 1] == xs[i]:
            j += 1
        if j > i:
            r[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return r


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman 秩相关（平均秩）。任一侧无变异（全并列）时返回 nan，
    不伪装成 0——那正是「截断把次序抹平」的情形，必须显式报出来。"""
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    if len(x) < 2:
        return float("nan")
    rx, ry = _rank(x), _rank(y)
    if rx.std() < 1e-12 or ry.std() < 1e-12:
        return float("nan")
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    return float((rx * ry).sum() / np.sqrt((rx * rx).sum() * (ry * ry).sum()))
