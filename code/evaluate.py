# -*- coding: utf-8 -*-
"""统一评估器：注册表驱动，与训练完全同一条链（field.reading → 执行器）。

列表行格式（空行与 # 注释跳过）：
    <algo> <tau> <seed> <scene> <t_comm> <anchor> <chan> <n> [ckpt_task] [preset]
示例：
    field 5 0 flow 1 0 main 12
    learned 5 0 flow 1 0 main 12 flow_n12_learned_tau5_s0
    tuned 5 0 static 1 0 main 7 - cheng7
    ddpga 5 0 flow 1 0 main 30 flow_n30_ddpga_tau5_s0_umi_srand

algo ∈ 场族别名 field/tuned/learned（→ 执行器 max/tuned/net）｜
        SOTA 臂 ddpga/dqnm/mfdqn/tmarl ｜ 免训基线 rand/fixed/greedy/rr/shen/jiang

口径照旧：确定性执行、三档 2000/3000/4000 步、pre=[700,1000]、
post=末 300 步、cov 含回合初值、e_fly/travel、逐场曲线 CSV。
真实观测挡板与训练同开关（PF_REAL_OBS=1）。训练臂按 ckpt_task
加载 latest.pt（从零唯一模式，无版本兼容分支）。
"""
from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

import numpy as np
import torch

from uav_env import CKPT_ROOT, RESULT_DIR, collect_nb, decode_ddp, norm_obs, \
    select81

# ckpt 读取根（顺序查找）：归档格（experiments/<task>/ckpt，纪律位置）
# 优先，训练暂存区（code/ckpt）兜底（在训/未归档件）。
EVAL_CKPT_ROOTS = (CKPT_ROOT.parent.parent / "artifacts" / "experiments",
                   CKPT_ROOT)


def _task_dir(task: str) -> Path:
    for root in EVAL_CKPT_ROOTS:
        d = root / task / "ckpt"
        if (d / "latest.pt").exists():
            return d
    return EVAL_CKPT_ROOTS[0] / task / "ckpt"   # 不存在也返回（报缺件路径）
from field import Field
from executors import EXECUTORS
from executors.net_executor import Net
from trainer import RealObsGate
from heuristics import make_baseline

EVAL_STEPS = (2000, 3000, 4000)
COV_DB = 5.0                       # 覆盖判定门限（SINR ≥ 5 dB，照旧）

# 场族别名 → 执行器注册键
FIELD_ALIAS = {"field": "max", "tuned": "tuned", "learned": "net"}
BASELINES = ("rand", "fixed", "greedy", "rr", "shen", "jiang")
SOTA_ARMS = ("ddpga", "dqnm", "mfdqn", "tmarl")

FIELDS = ["var_u_pre", "var_u_post", "u_mean_pre", "u_mean_post",
          "sum_t_pre", "sum_t_post", "n_over_pre", "n_over_post",
          "var_u_clip_pre", "var_u_clip_post", "n_dead_pre", "n_dead_post",
          "cov_pre", "cov_post", "algo", "tau", "seed", "scene", "t_comm",
          "anchor", "chan", "n_uav", "preset", "e_fly", "travel",
          "ckpt_task", "steps"]


def load_net(task: str) -> Net:
    ck = _task_dir(task) / "latest.pt"
    if not ck.exists():
        raise FileNotFoundError(task)
    sd = torch.load(ck, map_location="cpu", weights_only=False)
    net = Net()
    net.load_state_dict(sd["policy"])
    net.eval()
    return net


def run_name(mode, tau, seed, scene, n_uav, chan):
    """ckpt 任务名（缺省时按旧命名约定推导）。"""
    sig = os.environ.get("PF_SHADOW_DB", "random").strip().lower()
    stag = "srand" if sig in ("random", "rand", "r") else f"s{sig}"
    parts = [scene]
    if n_uav != 12:
        parts.append(f"n{n_uav}")
    parts.append(mode)
    parts.append(f"tau{int(tau)}_s{seed}")
    if chan != "main":
        parts.append(chan)
    parts.append(stag)
    return "_".join(parts)


class Controller:
    """评估分派：场族执行器链、SOTA 臂、免训基线。"""

    def __init__(self, algo, env, seed, ckpt_task=None):
        self.mode = algo
        self.rng = np.random.default_rng(seed)
        self.net = None
        self.ctrl = None
        if algo in FIELD_ALIAS or algo in EXECUTORS:
            key = FIELD_ALIAS.get(algo, algo)
            ex = EXECUTORS[key]()
            if key == "net":                     # learned 臂：加载 ckpt
                task = ckpt_task or run_name(
                    "learned", 5.0, seed, env.scene, env.n_uav, "main")
                ex.net = load_net(task)
            f = Field()

            def ctrl(env, _ex=ex, _f=f):
                rd = _f.reading(env)
                return _ex.act(env, rd) if not _ex.needs_training \
                    else _ex.act(env, rd, deterministic=True)
            self.ctrl = ctrl
        elif algo in BASELINES:
            self.ctrl = make_baseline(algo, seed)
        else:
            self.net = self._load_arm(algo, seed, env, ckpt_task)

    def _load_arm(self, mode, seed, env, ckpt_task):
        tdir = _task_dir(ckpt_task or run_name(
            mode, 5.0, seed, env.scene, env.n_uav, "main"))
        ck = tdir / "latest.pt"
        if not ck.exists():
            raise FileNotFoundError(tdir.name)
        sd = torch.load(ck, map_location="cpu")
        if mode == "dqnm":
            from dqnm import QNet, DQNWrap
            policy = QNet()
            policy.load_state_dict(sd["policy"])
            return DQNWrap(policy)
        if mode == "ddpga":
            from ddpga import Actor as DDPActor, DDPGWrap
            policy = DDPActor.from_ckpt(sd["policy"])
            policy.load_state_dict(sd["policy"])
            return DDPGWrap(policy)
        if mode == "mfdqn":
            from mfdqn import MFWrap, QNet as MFQNet
            policy = MFQNet.from_ckpt(sd["policy"])
            policy.load_state_dict(sd["policy"])
            return MFWrap(policy, env.n_uav, sd.get("phi_rel", None))
        from tmarl import Actor as TActor, TWrap
        policy = TActor()
        policy.load_state_dict(sd["policy"])
        return TWrap(policy)

    def reset(self):
        """回合边界：有状态的智能体（累计量、平均场）归零。"""
        for obj in (self.net, self.ctrl):
            if obj is not None and hasattr(obj, "reset"):
                obj.reset()

    def act(self, env, deterministic=True):
        """统一返回三元组 (d, d_copy, aux)——旧 Controller 约定。"""
        if self.ctrl is not None:
            fn = getattr(self.ctrl, "act", None)
            if fn is not None:                     # 对象控制器（基线）
                return fn(env, deterministic)
            return self.ctrl(env), None, None      # 场族闭包
        if self.mode == "dqnm":
            q = self.net.q(norm_obs(env.obs()))
            _, d = select81(q, env, 0.0, self.rng)
            return d, d.copy(), np.zeros(env.n_uav)
        if self.mode in ("ddpga", "mfdqn"):
            d = self.net.act_polar(env)
            return d, d.copy(), np.zeros(env.n_uav)
        obs = norm_obs(env.obs())
        nb, mask = collect_nb(obs, env.adj, 12, env.pos)
        d = decode_ddp(self.net.act_polar(obs, nb, mask))
        return d, d.copy(), np.zeros(env.n_uav)


def rollout(env, ctrl, tag, steps):
    """确定性执行 steps 步；pre=[700,1000]，post=末 300 步（含 cov 初值、
    e_fly/travel、逐场曲线 CSV——口径照旧）。"""
    gate = RealObsGate(env, env.seed)
    env.reset()
    gate.snap = None
    ctrl.reset()
    prev = env.pos.copy()
    travel = 0.0
    cov = [float(np.mean(10 * np.log10(np.maximum(
        env._sinr.max(axis=0), 1e-30)) >= COV_DB))]
    for _ in range(steps):
        d, _, _ = gate(lambda: ctrl.act(env, deterministic=True))
        env.step(d)                       # 超过 EPISODE_LEN 也继续跑，不重置
        travel += float(np.abs(env.pos - prev).sum())
        prev = env.pos.copy()
        cov.append(float(np.mean(10 * np.log10(np.maximum(
            env._sinr.max(axis=0), 1e-30)) >= COV_DB)))
    st = env.stats
    lo, hi = steps - 300, steps
    agg = lambda xs, a, b: float(np.mean(xs[a:b]))          # noqa: E731
    w = {}
    for k, xs in (("var_u", st.var_u), ("u_mean", st.u_mean),
                  ("sum_t", st.sum_t), ("n_over", st.n_over),
                  ("var_u_clip", st.var_u_clip), ("n_dead", st.n_dead)):
        w[f"{k}_pre"] = agg(xs, 700, 1000)
        w[f"{k}_post"] = agg(xs, lo, hi)
    w["cov_pre"] = float(np.mean(cov[700:1000]))
    w["cov_post"] = float(np.mean(cov[lo:hi]))
    cur_dir = RESULT_DIR / "curves" / env.scene
    cur_dir.mkdir(parents=True, exist_ok=True)
    with open(cur_dir / f"{tag}_s{steps}.csv", "w", encoding="utf-8") as fp:
        fp.write("t,u_mean,var_u,var_u_clip,n_dead,sum_t,n_over,cov\n")
        for t, (um, vv, vc, nd, stt, no, cv) in enumerate(zip(
                st.u_mean, st.var_u, st.var_u_clip, st.n_dead, st.sum_t,
                st.n_over, cov)):
            fp.write(f"{t},{um:.6g},{vv:.6g},{vc:.6g},{nd},{stt:.6g},"
                     f"{no},{cv:.4f}\n")
    w.update({"e_fly": float(np.sum(st.e_fly)), "travel": travel})
    return w


def evaluate_line(parts):
    algo, tau, seed, scene = parts[0], float(parts[1]), int(parts[2]), parts[3]
    t_comm, anchor = int(parts[4]), float(parts[5])
    chan, n = parts[6], int(parts[7])
    ckpt_task = parts[8] if len(parts) > 8 and parts[8] not in ("-", "") else None
    preset = parts[9] if len(parts) > 9 and parts[9] not in ("-", "") else None

    import argparse
    a = argparse.Namespace(
        algo=algo, tau=tau, seed=seed, scene=scene, n=n, chan=chan,
        preset=preset, t_comm=t_comm)
    from channel_env import build_env
    env = build_env(a)

    ctrl = Controller(algo, env, seed, ckpt_task)
    tag = ckpt_task or run_name(algo, tau, seed, scene, n, chan)
    if preset:
        tag = f"{tag}_{preset}"
    rows = []
    for steps in EVAL_STEPS:
        w = rollout(env, ctrl, tag, steps)
        w.update({"algo": algo, "tau": tau, "seed": seed, "scene": scene,
                  "t_comm": t_comm, "anchor": anchor, "chan": chan,
                  "n_uav": n, "preset": preset or "",
                  "ckpt_task": ckpt_task or "", "steps": steps})
        print(f"[{tag}] steps={steps} u_mean={w['u_mean_post']:.4f} "
              f"var_u={w['var_u_post']:.3e} sum_t={w['sum_t_post']:.1f} "
              f"n_over={w['n_over_post']:.2f}", flush=True)
        rows.append(w)
    env.close()
    return rows


def main() -> int:
    lst, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [ln.strip() for ln in lst.read_text(encoding="utf-8").splitlines()
             if ln.strip() and not ln.strip().startswith("#")]
    need = len(lines) * len(EVAL_STEPS)
    rows = []
    for ln in lines:
        parts = ln.split()
        try:
            rows.extend(evaluate_line(parts))
        except FileNotFoundError as e:
            print(f"[eval] 缺 ckpt，跳过 {e}", flush=True)
        except Exception as e:                  # noqa: BLE001
            print(f"[eval] 失败 {' '.join(parts)}: "
                  f"{type(e).__name__}: {e}", flush=True)
    if rows:
        with open(out, "w", newline="", encoding="utf-8") as fp:
            wtr = csv.DictWriter(fp, fieldnames=FIELDS)
            wtr.writeheader()
            for r in rows:
                wtr.writerow({k: r.get(k, "") for k in FIELDS})
    ok = len(rows) == need
    print(f"[eval] {'评估完成' if ok else '评估未完成'} "
          f"{len(rows)}/{need} 条 → {out}", flush=True)
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
