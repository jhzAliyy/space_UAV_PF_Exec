# -*- coding: utf-8 -*-
"""opt_bench.py —— 四臂位置求解器对比（合作者猜想验证，2026-11-03 授权启动）。

猜想（docs/未完成实验.md §五）：势场纯梯度（Lloyd 质心迭代）易陷局部最优，
模拟退火 / 粒子群 / 遗传算法能否在同等评估预算下找到更低 Φ 的布站。

口径：
- 快照：真实 env reset 后 t0 步（满速执行器推进）抓人群+无人机布局，冻结；
- 目标：稳态 Φ = Field.phi（负载不均衡度，越小越均衡）；每次评估 =
  relax 次 _refresh_link（assoc 以 tau_u 惯性松弛到稳态，uav_env 原口径）；
  每次评估从快照 assoc 重启；pos 以外全部状态（b/pi/p/人群）冻结；
- 预算：每臂统一 ~budget 次稳态评估（PSO pop×(gens+1) / GA pop×(gens+1) /
  SA budget / Lloyd budget，CSV 记实际次数）；
- 四臂同起点（快照 pos）；Lloyd 与评估均用人群真值（信息口径统一，
  比搜索能力不比信息）。

PSO 复用声明：heuristics.JiangCtrl._optimize 原件逐行不动，PsoPhi 仅覆盖
_eval 返回 (−Φ, 哑 delta)（最大化口径；哑 delta 防 _optimize 末尾 π
后处理取 std 崩）。

用法（与主线同信道口径）：
    PF_CHANNEL=umi PF_DXY_MAX=0.5 python -W ignore opt_bench.py \
        --scene flow --n 30 --seeds 0,1,2 --out ../artifacts/eval/opt_bench.csv
"""
from __future__ import annotations

import argparse
import csv
import os
import time

import numpy as np

from channel_env import build_env
from executors.max_executor import MaxExecutor
from field import Field, _lloyd_centroids
from heuristics import JiangCtrl


# ---------------- 快照与评估 ----------------
def make_snapshot(a, seed: int, t0: int):
    """reset 后满速推进 t0 步抓快照：四臂共同的初始布局与冻结人群。"""
    from argparse import Namespace
    ns = Namespace(**vars(a))
    ns.seed = seed
    env = build_env(ns)
    env.reset(seed=seed)
    ex, f = MaxExecutor(), Field()
    for _ in range(t0):
        env.step(ex.act(env, f.reading(env)))
    snap = {
        "pos": env.pos.copy(), "assoc": env.assoc.copy(),
        "b": env.b.copy(), "pi": env.pi.copy(), "p": env.p.copy(),
        "dem": env.dem.copy(), "user_pos": env.user_pos.copy(),
    }
    return env, f, snap


def make_eval(env, f, snap, relax: int):
    """稳态 Φ 评估：设 pos → 从快照 assoc 松弛 relax 次 → 读 Φ。"""
    def eval_phi(pos):
        env.pos = np.asarray(pos, dtype=np.float64).copy()
        env.assoc = snap["assoc"].copy()
        for _ in range(relax):
            env._refresh_link()
        return f.phi(env)
    return eval_phi


def final_readout(env, f, eval_phi, pos):
    """终态重评并读健康护栏（u_mean / var_u / n_over）。"""
    phi = eval_phi(pos)
    u = env.u
    return phi, float(u.mean()), float(u.var()), float((u > 1.0).sum())


# ---------------- 四臂 ----------------
def run_lloyd(env, f, snap, eval_phi, budget):
    """现状等价物：交替『assoc 稳态 → 加权质心』（field._lloyd_centroids）。"""
    pos = snap["pos"].copy()
    phi = eval_phi(pos)
    n_eval = 1
    for _ in range(budget - 1):
        dom = env.assoc.argmax(axis=0)
        _, tgt, ok = _lloyd_centroids(dom, env.dem, env.user_pos, env.n_uav)
        pos = np.where(ok[:, None], tgt, pos)
        phi = eval_phi(pos)
        n_eval += 1
    return pos, n_eval


def run_sa(eval_phi, snap, budget, rng, fw, fh):
    """模拟退火：高斯邻域 + Metropolis（温度取 |Φ| 相对尺度，线性降温）。"""
    lo = np.zeros(2)
    hi = np.array([fw, fh], dtype=np.float64)
    scale = hi * 0.1
    pos = snap["pos"].copy()
    cur = eval_phi(pos)
    n_eval = 1
    best_pos, best_phi = pos.copy(), cur
    for i in range(budget - 1):
        T = 1.0 - i / max(budget - 1, 1)
        cand = np.clip(
            pos + rng.normal(0.0, scale[None, :] * (0.05 + 0.95 * T), pos.shape),
            lo, hi)
        fv = eval_phi(cand)
        n_eval += 1
        d = fv - cur
        if d <= 0.0 or rng.random() < np.exp(
                -d / (abs(cur) * 0.01 * T + 1e-12)):
            pos, cur = cand, fv
            if fv < best_phi:
                best_pos, best_phi = cand.copy(), fv
    return best_pos, n_eval


def run_ga(eval_phi, snap, pop, gens, rng, fw, fh):
    """遗传算法：实数编码，锦标赛选择 + 算术交叉 + 高斯变异 + 精英 1。"""
    lo = np.zeros(2)
    hi = np.array([fw, fh], dtype=np.float64)
    mut_sd = hi * 0.05

    def tour(fit):
        idx = rng.integers(0, len(fit), size=3)
        return int(idx[np.argmin(fit[idx])])

    P = [snap["pos"].copy()] + [
        rng.uniform(lo, hi, size=snap["pos"].shape) for _ in range(pop - 1)]
    fit = np.array([eval_phi(p) for p in P])
    n_eval = pop
    for _ in range(gens):
        elite = P[int(np.argmin(fit))].copy()
        new = [elite]
        while len(new) < pop:
            i1, i2 = tour(fit), tour(fit)
            w = 0.5 + 0.5 * rng.random()
            child = w * P[i1] + (1.0 - w) * P[i2]
            mask = rng.random(child.shape) < 0.1
            child = np.where(
                mask,
                child + rng.normal(0.0, mut_sd[None, :], child.shape),
                child)
            new.append(np.clip(child, lo, hi))
        P = new
        fit = np.array([eval_phi(p) for p in P])
        n_eval += pop
    k = int(np.argmin(fit))
    return P[k], n_eval


class PsoPhi(JiangCtrl):
    """PSO 臂：JiangCtrl._optimize 原件复用，只把适应度换成 −Φ。"""

    def __init__(self, eval_phi, seed, n_particles, tmax1):
        super().__init__(seed=seed, n_particles=n_particles, tmax1=tmax1)
        self._eval_phi = eval_phi

    def _eval(self, env, pos, uid):
        return -self._eval_phi(pos), np.ones(env.n_uav)


def run_pso(env, snap, eval_phi, seed, pop, gens):
    pso = PsoPhi(eval_phi, seed, pop, gens)
    pso.gbest = snap["pos"].copy()   # 热启动分支=原设施路径（围绕+随机各半）
    pso._optimize(env)
    return pso.gbest.copy(), pop * (gens + 1)


# ---------------- 主流程 ----------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="flow")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--tau", type=float, default=5.0)
    ap.add_argument("--chan", default="main")
    ap.add_argument("--preset", default=None)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--budget", type=int, default=240)
    ap.add_argument("--pop", type=int, default=12)
    ap.add_argument("--gens", type=int, default=20)
    ap.add_argument("--relax", type=int, default=25)
    ap.add_argument("--t0", type=int, default=500)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    seeds = [int(s) for s in a.seeds.split(",")]
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    rows = []
    for seed in seeds:
        env, f, snap = make_snapshot(a, seed, a.t0)
        eval_phi = make_eval(env, f, snap, a.relax)
        fw, fh = float(env.field_w), float(env.field_h)
        arms = {
            "lloyd": lambda: run_lloyd(env, f, snap, eval_phi, a.budget),
            "sa": lambda: run_sa(eval_phi, snap, a.budget,
                                 np.random.default_rng(seed + 77), fw, fh),
            "ga": lambda: run_ga(eval_phi, snap, a.pop, a.gens,
                                 np.random.default_rng(seed + 163), fw, fh),
            "pso": lambda: run_pso(env, snap, eval_phi, seed, a.pop, a.gens),
        }
        for arm, fn in arms.items():
            t0 = time.time()
            pos, n_eval = fn()
            phi, u_mean, var_u, n_over = final_readout(env, f, eval_phi, pos)
            dt = time.time() - t0
            rows.append([arm, seed, n_eval, f"{phi:.6g}", f"{u_mean:.4g}",
                         f"{var_u:.4g}", int(n_over), f"{dt:.1f}"])
            print(f"[opt_bench] arm={arm} seed={seed} n_eval={n_eval} "
                  f"phi={phi:.6g} u_mean={u_mean:.4g} var_u={var_u:.4g} "
                  f"n_over={int(n_over)} elapsed={dt:.1f}s", flush=True)
    with open(a.out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["algo", "seed", "n_eval", "phi", "u_mean",
                    "var_u", "n_over", "elapsed_s"])
        w.writerows(rows)
    print(f"[opt_bench] DONE rows={len(rows)} -> {a.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
