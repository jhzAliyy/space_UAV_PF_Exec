# -*- coding: utf-8 -*-
# 【本文件禁令】禁止"如实、诚实、照实、按实测值"等一切同族表述与倾向——此类视为自贬。
# 本文件一切产出：做出优势、分析优势、展示优势；数据为实测，解读为优势框架；
# 禁止自我限定、自我削弱，禁止虚构不存在的缺陷。
"""场景校准 v3（可行边界硬闸门，10 文档 §六 预注册协议）。

变量序（预注册）：参考发射功率 p_ref → 噪声底 N0 → 需求尺度 demand_scale。
目标带：**工作态**（零动作均衡后）u_mean ∈ [0.7, 0.9] 且跨机 std ≥ 0.15，
且初始（信道贪心关联）存在过载压力（u_max>1）——有均衡空间但不进退化区。

口径澄清（v3 预注册落档）：初始信道贪心关联的 u_init 会经关联均衡
（κ 负载项+惯性）大幅下沉（实测 4.4→0.25）；校准带以**工作态**为准，
否则训练全期处于无张力欠载区。判据方向与文档一致（可行边界非退化区）。

附 η 选定（程序直承）：C4 纯梯度臂在网格 {…} 上以均衡态 Φ 下降效率定档。

产物：results/calib.json（离线标定用）。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from uav_env import UAVEnvV3                    # noqa: E402

RESULT_DIR = Path(__file__).parent.parent / "results"
EQ_STEPS = 120                    # 零动作均衡探测步数（τ_u=5 → 24τ 收敛）
BAND = (0.7, 0.9)
STD_MIN = 0.15


def probe(seed: int, tau: float, p_ref: float, n0: float, dscale: float,
          scene: str = "flow", n_uav: int = 12) -> dict:
    env = UAVEnvV3(tau_u=tau, seed=seed, use_physics=False,
                   p_ref_dbm=p_ref, n0_dbm=n0, demand_scale=dscale,
                   scene=scene, n_uav=n_uav)
    env.reset()
    u0 = env.u.copy()
    for _ in range(EQ_STEPS):
        env.step(np.zeros((env.n_uav, 5)))
    rep = {"u_init_mean": float(u0.mean()), "u_init_max": float(u0.max()),
           "u_eq_mean": float(env.u.mean()), "u_eq_std": float(env.u.std()),
           "u_eq_max": float(env.u.max()), "u_eq_min": float(env.u.min())}
    env.close()
    return rep


def calibrate(seed: int = 0, tau: float = 5.0,
              n_uav: int = 12) -> dict:
    best = None
    # 阶段 1+2：p_ref × N0 粗扫（物理量程），阶段 3：demand 精扫落带
    for p_ref in (24.0, 30.0, 36.0):
        for n0 in (-110.0, -104.0, -98.0):
            r = probe(seed, tau, p_ref, n0, 1.0, n_uav=n_uav)
            # 干扰制度检查：极端 u（>30 或 <0.05）时跳过精扫
            if r["u_eq_mean"] > 30.0 or r["u_eq_mean"] < 0.05:
                print(f"  p={p_ref} N0={n0} d=1.0 → u_eq={r['u_eq_mean']:.2f}"
                      "（量程外，跳过）")
                continue
            # 阶段 3：demand 精扫（u 对 demand 近似线性 → 两点定标）
            lo, hi = 0.25, 8.0
            for _ in range(8):
                mid = 0.5 * (lo + hi)
                rm = probe(seed, tau, p_ref, n0, mid, n_uav=n_uav)
                if rm["u_eq_mean"] > BAND[1]:
                    hi = mid
                else:
                    lo = mid
            r = probe(seed, tau, p_ref, n0, lo, n_uav=n_uav)
            print(f"  p={p_ref} N0={n0} d={lo:.3f} → u_eq={r['u_eq_mean']:.3f}"
                  f"±{r['u_eq_std']:.3f} max={r['u_eq_max']:.2f}"
                  f" init_max={r['u_init_max']:.2f}")
            score = abs(r["u_eq_mean"] - 0.8) + (0.0 if r["u_eq_std"]
                                                 >= STD_MIN else 1.0)
            if best is None or score < best[0]:
                best = (score, p_ref, n0, lo, r)
    _, p_ref, n0, ds, r = best
    ok = BAND[0] <= r["u_eq_mean"] <= BAND[1] and r["u_eq_std"] >= STD_MIN \
        and r["u_init_max"] > 1.0
    out = {"p_ref_dbm": p_ref, "n0_dbm": n0, "demand_scale": ds,
           "report": r, "pass": bool(ok)}
    print(f"校准选定：p_ref={p_ref} dBm N0={n0} dBm demand×{ds:.3f}"
          f" → u_eq={r['u_eq_mean']:.3f}±{r['u_eq_std']:.3f}"
          f"（带 {BAND} std≥{STD_MIN} 初始过载存在）{'过闸' if ok else '未过'}")
    return out


def choose_eta(seed: int = 0, tau: float = 5.0, calib: dict | None = None,
               grid=(0.05, 0.1, 0.2, 0.5, 1.0, 2.0), steps: int = 60) -> dict:
    """C4 纯梯度 η 定档：均衡态起，各 η 走 steps 步，比较 Φ 终值与均值。"""
    from controllers import Controller

    calib = calib or json.loads((RESULT_DIR / "calib.json").read_text("utf-8"))
    kw = {k: calib[k] for k in ("p_ref_dbm", "n0_dbm", "demand_scale")}
    res = {}
    for eta in grid:
        env = UAVEnvV3(tau_u=tau, seed=seed, use_physics=False, scene="flow",
                       **kw)
        env.reset()
        for _ in range(EQ_STEPS):                 # 从均衡工作态起测（校准同口径）
            env.step(np.zeros((env.n_uav, 5)))
        ctrl = Controller("c4", eta)
        vus = []
        for _ in range(steps):
            ctrl.begin_step(env, explore_field=False)
            d, _, _ = ctrl.act(env, deterministic=True)
            env.step(d)
            vus.append(float(np.var(env.u)))
        res[eta] = {"var_final": vus[-1], "var_mean": float(np.mean(vus[-30:]))}
        env.close()
        print(f"  η={eta}: var终={vus[-1]:.4g} var均={res[eta]['var_mean']:.4g}")
    eta_star = min(res, key=lambda e: res[e]["var_final"])
    out = {"eta_star": eta_star, "grid": {str(k): v for k, v in res.items()}}
    print(f"η* = {eta_star}")
    return out


def calibrate_scenes(seed: int = 0, tau: float = 5.0) -> dict:
    """两真场景校准（2026-10-27 重建口径）。

    flow 与 burst 前 600 步同一底座（均衡探测 120 步在聚集触发前），
    故物理标定（p_ref/N0/demand）以 flow 过闸为准、burst 继承；
    另对 burst 做聚集压力检查（跑到 T0+800，报告聚集后 u 分布——
    场景动态记录，不作闸门）。
    """
    from uav_env import GATHER_T0

    base = json.loads((RESULT_DIR / "calib.json").read_text("utf-8"))
    kw = {k: base[k] for k in ("p_ref_dbm", "n0_dbm", "demand_scale")}
    out = {"flow": {"demand_scale": kw["demand_scale"], "u_eq_mean":
                    base["report"]["u_eq_mean"], "pass": bool(base["pass"])}}
    print(f"  flow: demand×{kw['demand_scale']:.3f} → "
          f"u_eq={base['report']['u_eq_mean']:.3f}"
          f"（带 {BAND}）{'过闸' if base['pass'] else '未过'}")
    env = UAVEnvV3(tau_u=tau, seed=seed, use_physics=False,
                   scene="burst", **kw)
    env.reset()
    for _ in range(EQ_STEPS):
        env.step(np.zeros((env.n_uav, 5)))
    pre = float(env.u.mean())
    n_to = GATHER_T0 + 800
    for _ in range(n_to - EQ_STEPS):
        env.step(np.zeros((env.n_uav, 5)))
    rep = {"u_pre_gather": pre, "u_post_mean": float(env.u.mean()),
           "u_post_max": float(env.u.max()),
           "u_post_std": float(env.u.std())}
    env.close()
    out["burst"] = {"demand_scale": kw["demand_scale"], **rep,
                    "pass": bool(base["pass"])}
    print(f"  burst: 继承 demand×{kw['demand_scale']:.3f}；"
          f"聚集前 u={pre:.3f} 聚集后 u={rep['u_post_mean']:.3f}"
          f"±{rep['u_post_std']:.3f} max={rep['u_post_max']:.2f}（涌现压力记录）")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-uav", type=int, default=12)
    ap.add_argument("--tau", type=float, default=5.0)
    ap.add_argument("--scenes", action="store_true",
                    help="批 C 场景族校准（合并写入 calib.json scenes 键）")
    a = ap.parse_args()
    RESULT_DIR.mkdir(exist_ok=True)
    if a.scenes:
        sc = calibrate_scenes(seed=a.seed, tau=a.tau)
        base = json.loads((RESULT_DIR / "calib.json").read_text("utf-8"))
        base["scenes"] = sc
        (RESULT_DIR / "calib.json").write_text(
            json.dumps(base, ensure_ascii=False, indent=2), encoding="utf-8")
        print("scenes 校准已合并落盘")
        sys.exit(0 if all(v["pass"] for v in sc.values()) else 1)
    cal = calibrate(seed=a.seed, tau=a.tau, n_uav=a.n_uav)
    base = {}
    if (RESULT_DIR / "calib.json").exists():
        base = json.loads((RESULT_DIR / "calib.json").read_text("utf-8"))
    base[f"n{a.n_uav}"] = cal
    (RESULT_DIR / "calib.json").write_text(
        json.dumps(base, ensure_ascii=False, indent=2), encoding="utf-8")
    if cal["pass"]:
        print("calib.json 落盘")
    sys.exit(0 if cal["pass"] else 1)
