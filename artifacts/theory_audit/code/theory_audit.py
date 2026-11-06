# -*- coding: utf-8 -*-
"""theory_audit.py — 理论篇 D1/D2/D3/A6 在 n3~n100 的数值核验。

只读环境状态，不训练、不加载 ckpt、不改任何训练/评估口径。
单任务：
    python theory_audit.py --n 12 --scene flow --seed 0
输出：
    artifacts/theory_audit/out/theory_audit_n{n}_{scene}_s{seed}.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

# 环境变量必须在 import uav_env 之前设置
os.environ.setdefault("PF_CHANNEL", "umi")
os.environ.setdefault("PF_SHADOW_DB", "6")
os.environ.setdefault("PF_DXY_MAX", "0.5")
os.environ.setdefault("PF_REAL_OBS", "0")
os.environ.setdefault("PF_TORCH_THREADS", "1")

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "code"))

from uav_env import (  # noqa: E402
    UAVEnvV3, table_path, EPS_U, ALPHA_ASSOC, KAPPA, U_STAR,
)
from field import Field  # noqa: E402

LAMBDA = np.array([0.2, 0.2, 1.0, 1.0, 1.0])
TAU = 5.0
SNAPSHOTS = (700, 1500)
OUT_DIR = ROOT / "artifacts" / "theory_audit" / "out"


def bstar_from_c(c, b_total):
    """与 field.b_target 逐位同式的闭式带宽目标。"""
    bs = np.asarray(c, dtype=np.float64) / max(U_STAR, 1e-9)
    s = bs.sum()
    if s > b_total:
        bs = bs * (b_total / s)
    return bs


class State:
    """快照：只保留核验所需的环境量，避免持有整个 env。"""

    __slots__ = ("a", "b", "pi", "p", "u", "rho", "R_avail", "bh_cap",
                 "G_db", "dem", "n0_lin", "b_total")

    def __init__(self, env):
        self.a = env.assoc.copy()
        self.b = env.b.copy()
        self.pi = env.pi.copy()
        self.p = env.p.copy()
        self.u = env.u.copy()
        self.rho = env.rho.copy()
        self.R_avail = env.R_avail.copy()
        self.bh_cap = env.bh_cap.copy()
        self.G_db = env._G_db.copy()
        self.dem = env.dem.copy()
        self.n0_lin = float(env._n0_lin)
        self.b_total = float(env.b_total)


def assoc_step(state, a, tau=TAU):
    """复刻 uav_env._refresh_link 的一步关联更新（冻结速率口径）。"""
    g = 10.0 ** (state.G_db / 10.0)
    P = state.p * 10.0 ** (24.0 / 10.0)
    sig = P[:, None] * g
    agg = (a * sig).sum(axis=0)
    denom = state.n0_lin + agg[None, :] - a * sig
    sinr = sig / np.maximum(denom, 1e-30)
    r = np.log1p(sinr) / np.log(2.0)
    wgt = a * state.dem[None, :]
    bsh = state.b[:, None] * wgt / np.maximum(
        wgt.sum(axis=1, keepdims=True), 1e-9)
    R = bsh * r
    A = (a * R).sum(axis=1)
    scale = np.minimum(1.0, state.bh_cap / np.maximum(A, 1e-9))
    R2 = R * scale[:, None]
    R_avail = A * scale
    rho = a @ state.dem
    u = rho / np.maximum(R_avail, EPS_U)
    attn = (ALPHA_ASSOC * r
            + KAPPA * (R_avail / (rho + 1e-3))[:, None]
            + state.pi[:, None])
    x = np.exp(attn - attn.max(axis=0, keepdims=True))
    x /= x.sum(axis=0, keepdims=True)
    a_new = (1.0 - 1.0 / tau) * a + (1.0 / tau) * x
    return a_new, r, R2, R_avail, rho, u


def pure_action(env, field):
    rd = field.reading(env)
    return np.clip(LAMBDA[None, :] * (rd.target - rd.cur),
                   -rd.step[None, :], rd.step[None, :])


def d1_metrics(state):
    """D1：b* 对 c 的数值雅可比；秩一残差与非对角能量。"""
    c = state.u * state.b
    bs = bstar_from_c(c, state.b_total)
    n = len(c)
    J = np.zeros((n, n))
    for j in range(n):
        dc = max(abs(c[j]) * 1e-6, 1e-9)
        c2 = c.copy()
        c2[j] += dc
        J[:, j] = (bstar_from_c(c2, state.b_total) - bs) / dc
    sv = np.linalg.svd(J, compute_uv=False)
    ratio = float(sv[1] / max(sv[0], 1e-30)) if len(sv) > 1 else 0.0
    off = float(np.linalg.norm(J - np.diag(np.diag(J)))
                / max(np.linalg.norm(J), 1e-30))
    c_over_b = float(c.sum()) / state.b_total
    return dict(
        d1_sigma_ratio=ratio,
        d1_offdiag=off,
        mu=float(max(U_STAR, c_over_b)),
        c_sum_over_B=c_over_b,
        tier="budget" if c_over_b > U_STAR else "anchor",
    )


def d2_metrics(state, rng, n_probe=None):
    """D2：关联耦合项的随机化谱估计与谱范数/谱半径估计。"""
    n, K = state.a.shape
    l = int(n_probe or min(2 * n + 10, 60))
    V = rng.normal(size=(n, K, l))
    V /= np.maximum(
        np.linalg.norm(V.reshape(n * K, l), axis=0, keepdims=True), 1e-30)
    base = assoc_step(state, state.a)[0]
    eps = 1e-6
    JVs = np.empty((n, K, l))
    Y = np.empty((n, K, l))
    for i in range(l):
        ap = state.a + eps * V[:, :, i]
        JVs[:, :, i] = (assoc_step(state, ap)[0] - base) / eps
        Y[:, :, i] = JVs[:, :, i] - (1.0 - 1.0 / TAU) * V[:, :, i]
    Yf = Y.reshape(n * K, l)
    sv = np.linalg.svd(Yf, compute_uv=False)
    s_n = float(sv[n - 1]) if len(sv) >= n else 0.0
    s_n1 = float(sv[n]) if len(sv) > n else 0.0
    gap = float(s_n / max(s_n1, 1e-30))
    jv_norm = np.linalg.norm(JVs.reshape(n * K, l), axis=0)
    y_norm = np.linalg.norm(Yf, axis=0)
    coupling = float(np.mean(y_norm / np.maximum(jv_norm, 1e-30)))
    La = float(np.max(jv_norm))  # V 列已归一化

    # 谱半径：对 J 做幂迭代（D3 用谱半径，A6 用谱范数上界）
    v = rng.normal(size=(n, K))
    v /= max(np.linalg.norm(v), 1e-30)
    rho = 0.0
    for _ in range(20):
        jv = (assoc_step(state, state.a + eps * v)[0] - base) / eps
        nv = float(np.linalg.norm(jv))
        if nv <= 0:
            break
        rho = nv
        v = jv / nv
    return dict(
        d2_sigma_n=s_n,
        d2_sigma_n1=s_n1,
        d2_gap_ratio=gap,
        d2_coupling_energy=coupling,
        La=La,
        rho_a_power=float(rho),
        d2_n_probe=l,
    )


def d3_metrics(env, field, state, rng, n_probe=10):
    """D3：执行器谱半径、X* 对关联的敏感度、联合压缩因子。"""
    rho_X = float(np.max(np.abs(1.0 - LAMBDA)))
    a0 = env.assoc.copy()
    u0 = env.u.copy()
    rho0 = env.rho.copy()
    ra0 = env.R_avail.copy()
    X0 = field.target_state(env)
    vals = []
    eps = 1e-6
    for _ in range(n_probe):
        v = rng.normal(size=state.a.shape)
        nv = np.linalg.norm(v)
        if nv <= 0:
            continue
        v /= nv
        ap = state.a + eps * v
        an, r, R2, R_avail, rho, u = assoc_step(state, ap)
        env.assoc = an
        env.u = u
        env.rho = rho
        env.R_avail = R_avail
        X1 = field.target_state(env)
        vals.append(float(np.linalg.norm(X1 - X0) / eps))
    env.assoc = a0
    env.u = u0
    env.rho = rho0
    env.R_avail = ra0
    vals = np.asarray(vals) if vals else np.array([0.0])
    return dict(
        rho_X=rho_X,
        L_Xa_median=float(np.median(vals)),
        L_Xa_max=float(np.max(vals)),
    )


def audit_one(n, scene, seed, snapshots=SNAPSHOTS):
    lay = max(12, n)
    table = table_path(lay, "uma", 2e9)
    env = UAVEnvV3(
        tau_u=TAU, seed=seed, scene=scene, n_uav=n, t_comm=1,
        use_physics=False, p_ref_dbm=24.0, n0_dbm=-104.0,
        demand_scale=3.7648, scenario="uma", fc_hz=2e9,
        layout_n=lay, override_spec=None, table=table,
    )
    env.reset()
    field = Field()
    rng = np.random.default_rng(
        seed * 1009 + n * 31 + (0 if scene == "flow" else 1))
    rows = []
    t = 0
    max_step = max(snapshots)
    while t < max_step:
        env.step(pure_action(env, field))
        t += 1
        if t in snapshots:
            st = State(env)
            m1 = d1_metrics(st)
            m2 = d2_metrics(st, rng)
            m3 = d3_metrics(env, field, st, rng)
            rho_a = m2["rho_a_power"]
            Lxa = m3["L_Xa_median"]
            w = 1.0
            rho_J = max(rho_a + w * Lxa, m3["rho_X"])
            g = st.R_avail / (st.rho + 1e-3)
            rows.append(dict(
                n=n, scene=scene, seed=seed, step=t, tier=m1["tier"],
                mu=m1["mu"], c_sum_over_B=m1["c_sum_over_B"],
                max_cB=float(np.max(st.rho / np.maximum(st.bh_cap, 1e-9))),
                d1_sigma_ratio=m1["d1_sigma_ratio"],
                d1_offdiag=m1["d1_offdiag"],
                d2_sigma_n=m2["d2_sigma_n"],
                d2_sigma_n1=m2["d2_sigma_n1"],
                d2_gap_ratio=m2["d2_gap_ratio"],
                d2_coupling_energy=m2["d2_coupling_energy"],
                La=m2["La"], rho_a=rho_a, rho_X=m3["rho_X"],
                L_Xa_median=m3["L_Xa_median"], L_Xa_max=m3["L_Xa_max"],
                rho_J=rho_J,
                load_gap=float(np.max(st.u) - np.min(st.u)),
                load_std=float(np.std(st.u)),
                meanfield_residual=float(np.std(g) / max(np.mean(g), 1e-12)),
                u_mean=float(np.mean(st.u)),
                n_dead=int((st.u > 5).sum()),
                note=f"L_Xa_max={m3['L_Xa_max']:.6g};w={w:g}",
            ))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--scene", choices=["flow", "burst"], required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--snapshots", type=str, default="700,1500")
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()
    snaps = tuple(int(x) for x in args.snapshots.split(",") if x.strip())
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    rows = audit_one(args.n, args.scene, args.seed, snapshots=snaps)
    out = (Path(args.out) if args.out else
           OUT_DIR / f"theory_audit_n{args.n}_{args.scene}_s{args.seed}.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        with out.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    print(f"[theory_audit] n={args.n} scene={args.scene} seed={args.seed} "
          f"rows={len(rows)} elapsed={time.time() - t0:.1f}s out={out}")
    print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
