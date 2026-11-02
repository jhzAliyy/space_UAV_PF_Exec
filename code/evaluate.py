# -*- coding: utf-8 -*-
"""统一评估：读清单，跑满 2000 步，写 CSV 与逐场曲线。

清单行格式（空格分隔）：
    mode tau seed scene t_comm anchor chan n_uav [ckpt_task] [preset]

mode ∈ {pfrl, pfg, rand, fixed, greedy, rr, shen, jiang, ddpga, dqnm, mfdqn, tmarl}
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import torch

from uav_env import (ACT_DIM, CKPT_ROOT, EPISODE_LEN, N0_DBM, P_REF_DBM,
                     RESULT_DIR, UAVEnvV3, baseline_scene, collect_nb,
                     decode_ddp, norm_obs, select81)
from field import GlobalField
from heuristics import make_baseline
from pfrl import ResidualActor, ExecCtrl

CHAN_MAP = {"main": (None, None), "uma35": (None, 3.5e9),
            "rma2": ("rma", None), "rma35": ("rma", 3.5e9)}
COV_DB = 5.0


def run_name(mode, tau, seed, scene, n_uav, chan):
    import os
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
    """评估分派：项目方法、纯势场、免训基线、四个训练臂。"""

    def __init__(self, mode, env, seed, ckpt_task=None):
        self.mode = mode
        self.rng = np.random.default_rng(seed)
        self.net = None
        self.ctrl = None
        if mode == "pfrl":
            tdir = CKPT_ROOT / (ckpt_task or run_name(
                mode, 5.0, seed, env.scene, env.n_uav, "main"))
            ck = tdir / "latest.pt"
            if not ck.exists():
                raise FileNotFoundError(tdir.name)
            sd = torch.load(ck, map_location="cpu", weights_only=False)
            # v2 残差口径：ckpt 必须带 res_frac；缺键 = 旧版，直接报错。
            # b 维尺度（rizvi3 修复）：训练端落盘 env.b_scale，评估端断言
            # 与评估环境一致后按它重建残差上界，防场景/尺度错配。
            # 缺键 = 修复前 ckpt（在跑 9 格），其 b 维尺度恒为全局 5.0，
            # 且这些场景 env.b_scale 也是 5.0——回退 5.0 语义精确。
            from pfrl import act_scale_of
            _bs = float(sd.get("b_scale", 5.0))
            assert abs(_bs - env.b_scale) < 1e-9, (
                f"ckpt b_scale={_bs} 与评估环境 b_scale={env.b_scale} 不一致"
                f"（task={ckpt_task}）——场景或代码版本错配")
            actor = ResidualActor(widths=tuple(sd["widths"]),
                                  p=float(sd["dropout_p"]),
                                  res_frac=float(sd["res_frac"]),
                                  act_scale=act_scale_of(env))
            actor.load_state_dict(sd["policy"])
            # 臂配置从 ckpt 复原：训练用哪支场，评估就用哪支
            field = GlobalField(mode=sd["field_mode"],
                                map_name=(sd["map_name"] or None),
                                u_star=sd["u_star"])
            eta = float(sd["eta_fixed"])
            self.ctrl = ExecCtrl(actor, field, eta_fixed=eta,
                                 with_pos=bool(sd["with_pos"]))
        elif mode == "pfg":
            self.ctrl = None
            self.field = GlobalField(mode="closed", u_star=0.8)
        elif mode in ("rand", "fixed", "greedy", "rr", "shen", "jiang"):
            self.ctrl = make_baseline(mode, seed)
        else:
            self.net = self._load_arm(mode, seed, env, ckpt_task)

    def _load_arm(self, mode, seed, env, ckpt_task):
        from ddpga import Actor as DDPActor, DDPGWrap
        tdir = CKPT_ROOT / (ckpt_task or run_name(
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
        """回合边界：把有状态的智能体（累计量、平均场、上一步动作）归零。"""
        for obj in (self.net, self.ctrl):
            if obj is not None and hasattr(obj, "reset"):
                obj.reset()

    def begin_step(self, env, explore_field=False):
        pass

    def act(self, env, deterministic=True):
        if self.ctrl is not None:
            return self.ctrl.act(env, deterministic)
        if self.mode == "pfg":
            d = self.field.act(env)
            return d, np.zeros_like(d), np.zeros(env.n_uav)
        if self.net is None:
            return np.zeros((env.n_uav, ACT_DIM), np.float32), None, None
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
    """确定性执行 steps 步；pre = [700,1000]，post = 末 300 步。"""
    from pfrl import RealObsGate
    gate = RealObsGate(env, env.seed)
    env.reset()
    gate.snap = None
    ctrl.reset()                          # 有状态智能体的回合边界
    prev = env.pos.copy()
    travel = 0.0
    cov = [float(np.mean(10 * np.log10(np.maximum(
        env._sinr.max(axis=0), 1e-30)) >= COV_DB))]
    for _ in range(steps):
        ctrl.begin_step(env, explore_field=False)
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
    with open(cur_dir / f"{tag}_s{steps}.csv", "w", encoding="utf-8") as f:
        f.write("t,u_mean,var_u,var_u_clip,n_dead,sum_t,n_over,cov\n")
        for t, (um, vv, vc, nd, stt, no, cv) in enumerate(zip(
                st.u_mean, st.var_u, st.var_u_clip, st.n_dead, st.sum_t,
                st.n_over, cov)):
            f.write(f"{t},{um:.6g},{vv:.6g},{vc:.6g},{nd},{stt:.6g},{no},{cv:.4f}\n")
    w.update({"e_fly": float(np.sum(st.e_fly)), "travel": travel})
    return w


EVAL_STEPS = (2000, 3000, 4000)


def evaluate_line(parts):
    mode, tau, seed, scene = parts[0], float(parts[1]), int(parts[2]), parts[3]
    t_comm, anchor = int(parts[4]), float(parts[5])
    chan, n_uav = parts[6], int(parts[7])
    ckpt_task = parts[8] if len(parts) > 8 and parts[8] else None
    preset = parts[9] if len(parts) > 9 and parts[9] else None

    spec = baseline_scene(preset)
    scen, fc = CHAN_MAP[chan]
    env = UAVEnvV3(tau_u=tau, seed=seed, scene=scene, n_uav=n_uav,
                   t_comm=t_comm, layout_n=max(12, n_uav),
                   use_physics=(spec is None), override_spec=spec,
                   scenario=scen, fc_hz=fc,
                   p_ref_dbm=P_REF_DBM, n0_dbm=N0_DBM,
                   demand_scale=(spec or {}).get("demand_scale", 3.7648))
    ctrl = Controller(mode, env, seed, ckpt_task)
    tag = ckpt_task or run_name(mode, tau, seed, scene, n_uav, chan)
    if preset:
        tag = f"{tag}_{preset}"
    rows = []
    for steps in EVAL_STEPS:
        w = rollout(env, ctrl, tag, steps)
        w.update({"mode": mode, "tau": tau, "seed": seed, "scene": scene,
                  "t_comm": t_comm, "anchor": anchor, "chan": chan,
                  "n_uav": n_uav, "preset": preset or "",
                  "ckpt_task": ckpt_task or "", "steps": steps})
        print(f"[{tag}] steps={steps} u_mean={w['u_mean_post']:.4f} "
              f"var_u={w['var_u_post']:.3e} sum_t={w['sum_t_post']:.1f} "
              f"n_over={w['n_over_post']:.2f}", flush=True)
        rows.append(w)
    env.close()
    return rows


FIELDS = ["var_u_pre", "var_u_post", "u_mean_pre", "u_mean_post",
          "sum_t_pre", "sum_t_post", "n_over_pre", "n_over_post",
          "var_u_clip_pre", "var_u_clip_post", "n_dead_pre", "n_dead_post",
          "cov_pre", "cov_post", "mode", "tau", "seed", "scene", "t_comm",
          "anchor", "chan", "n_uav", "preset", "e_fly", "travel",
          "ckpt_task", "steps"]


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
        except Exception as e:            # noqa: BLE001
            # 单条失败不拖垮整批：大声打出类型与消息，继续下一条
            print(f"[eval] 失败 {' '.join(parts)}: "
                  f"{type(e).__name__}: {e}", flush=True)
    if rows:
        with open(out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k, "") for k in FIELDS})
    if len(rows) == need:
        print(f"[eval] 评估完成 {len(rows)} 条 → {out}", flush=True)
        return 0
    # 行数不足：调度器不得把本批标记为完成，缺的格按日志逐条补
    print(f"[eval] 评估未完成 {len(rows)}/{need} 条 → {out}", flush=True)
    return 2


if __name__ == "__main__":
    sys.exit(main())
