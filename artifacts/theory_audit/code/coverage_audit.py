# -*- coding: utf-8 -*-
"""coverage_audit.py — 学习执行器运行点覆盖率核验（n3~n100）。

每步记录 a_max / a_entropy；在 700/1500 步快照计算 rho_a（ARPACK）与
连续目标 L_Xa（b/π），输出“落在适用区”的步占比与快照占比。
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("PF_CHANNEL", "umi")
os.environ.setdefault("PF_SHADOW_DB", "6")
os.environ.setdefault("PF_DXY_MAX", "0.5")
os.environ.setdefault("PF_REAL_OBS", "0")
os.environ.setdefault("PF_TORCH_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "code"))

from uav_env import UAVEnvV3, table_path, U_STAR  # noqa: E402
from field import Field  # noqa: E402
from theory_audit import State, assoc_step, d3_metrics  # noqa: E402

OUT_DIR = ROOT / "artifacts" / "theory_audit" / "out"


def task_name(n, scene, seed):
    if n == 12:
        return f"{scene}_learned_tau5_s{seed}_umi_srand"
    return f"{scene}_n{n}_learned_tau5_s{seed}_umi_srand"


def _entropy(a):
    return float(np.mean(-(a * np.log(a + 1e-12)).sum(axis=0)))


def rho_a_arpack(state, rng):
    """关联块谱半径：ARPACK 优先，失败回退 50 步幂迭代。"""
    n, K = state.a.shape
    base = assoc_step(state, state.a)[0]
    eps = 1e-6
    N = n * K

    def matvec(vec):
        vv = np.asarray(vec, dtype=np.float64).reshape(n, K)
        return ((assoc_step(state, state.a + eps * vv)[0] - base)
                / eps).reshape(-1)

    try:
        from scipy.sparse.linalg import LinearOperator, eigs
        op = LinearOperator((N, N), matvec=matvec, dtype=np.float64)
        vals = eigs(op, k=1, which="LM", return_eigenvectors=False,
                    maxiter=200, tol=1e-3)
        return float(np.max(np.abs(vals))), "arpack"
    except Exception:
        v = rng.normal(size=(n, K))
        v /= max(np.linalg.norm(v), 1e-30)
        rho = 0.0
        for _ in range(50):
            jv = matvec(v.ravel()).reshape(n, K)
            nv = float(np.linalg.norm(jv))
            if nv <= 0:
                break
            rho = nv
            v = jv / nv
        return float(rho), "power"


def run_one(n, scene, seed, task=None, steps=1500, snaps=(700, 1500)):
    import torch
    torch.set_num_threads(1)
    from evaluate import Controller

    task = task or task_name(n, scene, seed)
    ck = ROOT / "artifacts" / "experiments" / task / "ckpt" / "latest.pt"
    if not ck.exists():
        return None

    lay = max(12, n)
    table = table_path(lay, "uma", 2e9)
    env = UAVEnvV3(
        tau_u=5.0, seed=seed, scene=scene, n_uav=n, t_comm=1,
        use_physics=False, p_ref_dbm=24.0, n0_dbm=-104.0,
        demand_scale=3.7648, scenario="uma", fc_hz=2e9,
        layout_n=lay, override_spec=None, table=table,
    )
    env.reset()
    ctrl = Controller("learned", env, seed, task)
    field = Field()
    rng = np.random.default_rng(seed * 1009 + n * 31 + (0 if scene == "flow" else 1))

    amax, aent, umean, ndead = [], [], [], []
    snap = {}
    for t in range(1, steps + 1):
        act = ctrl.ctrl(env)
        env.step(act)
        amax.append(float(env.assoc.max()))
        aent.append(_entropy(env.assoc))
        umean.append(float(np.mean(env.u)))
        ndead.append(int((env.u > 5).sum()))
        if t in snaps:
            st = State(env)
            rho, method = rho_a_arpack(st, rng)
            m3 = d3_metrics(env, field, st, rng)
            lxa = m3["L_Xa_median"]
            rho_j = (max(rho + lxa, m3["rho_X"])
                     if np.isfinite(lxa) else float("nan"))
            snap[t] = dict(
                rho_a=rho, rho_a_method=method, rho_J=rho_j,
                L_Xa=lxa, rho_X=m3["rho_X"],
                a_max=float(env.assoc.max()), a_entropy=_entropy(env.assoc),
                rho_min=float(np.min(env.rho)),
                R_avail_min=float(np.min(env.R_avail)),
                u_mean=float(np.mean(env.u)),
                n_dead=int((env.u > 5).sum()),
            )

    amax = np.asarray(amax)
    row = dict(
        n=n, scene=scene, seed=seed, task=task, mark="",
        steps=steps,
        frac_amax_lt_099=float(np.mean(amax < 0.99)),
        frac_amax_lt_0999=float(np.mean(amax < 0.999)),
        frac_amax_lt_09999=float(np.mean(amax < 0.9999)),
        amax_median=float(np.median(amax)),
        amax_min=float(amax.min()),
        amax_max=float(amax.max()),
        a_entropy_median=float(np.median(aent)),
        u_mean=float(np.mean(umean)),
        n_dead_max=int(max(ndead)),
    )
    for t, d in snap.items():
        for k, v in d.items():
            row[f"{k}_{t}"] = v
        row[f"in_region_{t}"] = bool(
            np.isfinite(d["rho_J"]) and d["rho_J"] < 1.0)
    vals = [snap[t]["rho_a"] for t in snaps if t in snap]
    row["frac_snap_rho_a_lt_1"] = float(np.mean([v < 1.0 for v in vals])) if vals else float("nan")
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--scene", choices=["flow", "burst"], required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--task", type=str, default=None)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    row = run_one(args.n, args.scene, args.seed, args.task, args.steps)
    if row is None:
        print(f"[coverage] missing ckpt n={args.n} {args.scene} s{args.seed}")
        return
    out = Path(args.out) if args.out else OUT_DIR / f"coverage_n{args.n}_{args.scene}_s{args.seed}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(row.keys()))
        w.writeheader()
        w.writerow(row)
    print(f"[coverage] n={args.n} scene={args.scene} seed={args.seed} "
          f"elapsed={time.time() - t0:.1f}s out={out}")
    print(json.dumps(row, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
