# -*- coding: utf-8 -*-
"""PF-Exec：势场给方向，执行器网络给四个归一化速率比例。

结构对标 cleanrl 的 ddpg_continuous_action.py：超参集中在 Args、
网络类同文件、主循环写在 __main__ 里、分段注释固定。

执行器契约（唯一）：
    输入 12 维观测，逐维除以 OBS_NORM → 同一量级；
    输出 4 维 σ = [σ_b, σ_π, σ_p, σ_xy]，sigmoid 保证落在 (0,1)；
    动作 = σ ⊙ (场均衡点 − 当前状态)，方向恒等于场，网络只定快慢。
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.stdout.reconfigure(encoding="utf-8")   # 日志固定 UTF-8，调度脚本按字面量匹配

from config import SIG_LO
from field import GlobalField
from replay import ReplayBuffer
from uav_env import (CKPT_EVERY, DB_MAX, DEMAND_SCALE_ANCHOR, DP_MAX,
                     DPI_MAX, EARLY_SHORT, EARLY_THR, EARLY_WIN, EPISODE_LEN,
                     N0_DBM, OBS_DIM, OBS_NORM, P_REF_DBM, PF_RELAX,
                     SIG_DIM, TOTAL_ROLLOUTS, U_STAR_DEFAULT, CKPT_ROOT,
                     UAVEnvV3, baseline_scene)

# ---------------------------------------------------------------- 超参


@dataclass
class Args:
    # ---- 实验 ----
    seed: int = 0
    scene: str = "flow"              # flow | burst | static
    n_uav: int = 12
    tau_u: float = 5.0
    t_comm: int = 1
    anchor: float = 0.0
    total_rollouts: int = TOTAL_ROLLOUTS
    rollout_len: int = 256
    task: str = ""
    # ---- 势场 ----
    u_star: float = U_STAR_DEFAULT
    field_mode: str = "closed"       # closed | natgrad | grad
    map_name: str = ""               # "" | A_clip3 | B_clip30 | C_logc
    with_pos: bool = True
    preset: str = ""
    # ---- 执行器 ----
    learning_rate_actor: float = 1e-3
    learning_rate_critic: float = 1e-3
    gamma: float = 0.99
    tau: float = 0.005
    batch_size: int = 64
    learning_starts: int = 2048      # 环境步：此前从动作空间均匀采样
    policy_frequency: int = 2        # cleanrl 同值：actor 每 2 个环境步更新一次
    buffer_rollouts: int = 30
    exploration_noise: float = 0.1   # σ 域噪声幅度。σ 量程为 1，故即 ±10%；
                                     # cleanrl 按半量程定标，等价幅度是 0.05
    widths: tuple = (16, 32, 64, 32, 16)
    dropout_p: float = 0.0           # 标准 DDPG/TD3 无 dropout（cleanrl 同）
    eta_fixed: float = -1.0          # >=0 时固定比例（消融臂），<0 表示学
    # ---- TD3（cleanrl td3_continuous_action.py 同款） ----
    policy_noise: float = 0.2        # 目标策略平滑噪声（动作量程 1 的比例）
    noise_clip: float = 0.5          # 平滑噪声截断
    # ---- 训练设施 ----
    ckpt_every: int = CKPT_EVERY
    early_win: int = EARLY_WIN
    early_short: int = EARLY_SHORT
    early_thr: float = EARLY_THR
    device: str = "cpu"
    fresh: bool = False


# ---------------------------------------------------------------- 网络


class ProjResBlock(nn.Module):
    """投影残差块：act(proj(x) + act(LN(FC(x))))。"""

    def __init__(self, d_in: int, d_out: int, p: float):
        super().__init__()
        self.fc = nn.Linear(d_in, d_out)
        self.ln = nn.LayerNorm(d_out)
        self.proj = (nn.Identity() if d_in == d_out
                     else nn.Linear(d_in, d_out, bias=False))
        self.drop = nn.Dropout(p)
        self.act = nn.GELU()

    def forward(self, x):
        h = self.drop(self.act(self.ln(self.fc(x))))
        return self.act(self.proj(x) + h)


class ExecActor(nn.Module):
    """o_i → σ ∈ [lo,1)^4 = [σ_b, σ_π, σ_p, σ_xy]。

    输出层线性给 logit，forward 里 lo + (1−lo)·sigmoid 出比例。
    速度下限 lo（config.SIG_LO）是 2026-11-01 规模负效应修复的核心：
    网络只允许学"比下限更快"，永远不能把跟队速度关死（悬停由场把
    缺口归零实现，与此正交）。sig_lo 随 ckpt 落盘，评估端按同值重建；
    缺 sig_lo 的旧 ckpt 不兼容，直接报错，不做回退。
    """

    def __init__(self, obs_dim: int = OBS_DIM, out_dim: int = SIG_DIM,
                 widths: tuple = (16, 32, 64, 32, 16), p: float = 0.1,
                 sig_lo=None):
        super().__init__()
        dims = (obs_dim,) + tuple(widths)
        self.blocks = nn.Sequential(*[
            ProjResBlock(dims[i], dims[i + 1], p)
            for i in range(len(dims) - 1)])
        self.out = nn.Linear(dims[-1], out_dim)
        nn.init.uniform_(self.out.weight, -3e-3, 3e-3)
        nn.init.constant_(self.out.bias, 0.0)     # logit 0 → σ = 中点
        lo = np.asarray(SIG_LO if sig_lo is None else sig_lo,
                        dtype=np.float32)
        assert lo.shape == (out_dim,), f"sig_lo 形状 {lo.shape} != {out_dim}"
        # persistent=False：lo 不进 state_dict；评估端必须从 ckpt 的
        # sig_lo 键重建同一支下限，缺键即报错。
        self.register_buffer("lo", torch.from_numpy(lo), persistent=False)

    def logits(self, x):
        return self.out(self.blocks(x))

    def forward(self, x):
        return self.lo + (1.0 - self.lo) * torch.sigmoid(self.logits(x))


class Critic(nn.Module):
    """Q(o, σ) → 标量。σ 是 actor 输出的四维增益。"""

    def __init__(self, obs_dim: int = OBS_DIM, act_dim: int = SIG_DIM,
                 widths: tuple = (16, 32, 64, 32, 16), p: float = 0.1):
        super().__init__()
        dims = (obs_dim + act_dim,) + tuple(widths)
        self.blocks = nn.Sequential(*[
            ProjResBlock(dims[i], dims[i + 1], p)
            for i in range(len(dims) - 1)])
        self.out = nn.Linear(dims[-1], 1)

    def forward(self, o, a):
        return self.out(self.blocks(torch.cat([o, a], dim=1))).squeeze(-1)


# ---------------------------------------------------------------- 契约


def pf_obs(env) -> np.ndarray:
    """观测定标：逐维除以 OBS_NORM（n12 纯势场 p95），12 维同一量级。"""
    return (env.obs() / OBS_NORM[None, :]).astype(np.float32)


def cur_state(env) -> np.ndarray:
    """当前状态向量 x（与 target_state 同坐标系）。"""
    x = np.zeros((env.n_uav, 5))
    x[:, 0] = env.b
    x[:, 1] = env.pi
    x[:, 2] = env.p
    base = (env._phys._setpoint if getattr(env, "_phys", None) is not None
            else env.pos)
    x[:, 3:5] = base
    return x


def scale_action(a_field: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """纯势场动作 (N,5) × 逐维增益 σ (N,4) → 实际动作 (N,5)。

    σ 的 xy 一位同时乘位置两列。a_field 已在动作界内，σ∈[0,1]，
    因此结果自然在界内，不需要额外 clip。
    """
    s = np.asarray(sigma, dtype=np.float32)
    a = np.array(a_field, copy=True)
    a[:, 0] *= s[:, 0]
    a[:, 1] *= s[:, 1]
    a[:, 2] *= s[:, 2]
    a[:, 3:5] *= s[:, 3:4]
    return a


def build_action(env, field: GlobalField, sigma: np.ndarray,
                 with_pos: bool = True):
    """σ ∈ (0,1)^4 → 实际动作。唯一实现，训练与评估共用。

    纯势场动作 a_field = field.act(env) 是唯一基准；网络只乘增益 σ。
    σ=1 时逐位等于纯势场。返回 (a, σ, a_field)。
    """
    a_field = field.act(env, with_pos=with_pos)
    return scale_action(a_field, sigma), np.asarray(
        sigma, dtype=np.float32), a_field


class ExecCtrl:
    """采集器：训练与评估同一路径。"""

    def __init__(self, actor, field, rng=None, exploration_noise=0.15,
                 eta_fixed=-1.0, with_pos=True):
        self.actor, self.field = actor, field
        self.rng = rng or np.random.default_rng(0)
        self.expl = float(exploration_noise)
        self.eta_fixed = float(eta_fixed)
        self.with_pos = bool(with_pos)

    def act(self, env, deterministic: bool = False, a_field=None):
        if self.eta_fixed >= 0.0:                 # 消融：固定比例
            sigma = np.full((env.n_uav, SIG_DIM), self.eta_fixed,
                            dtype=np.float32)
        else:
            net = self.actor
            was_training = net.training
            net.eval()                            # 采集关 dropout
            with torch.no_grad():
                s_t = net(torch.as_tensor(pf_obs(env), dtype=torch.float32))
                if not deterministic and self.expl > 0.0:
                    # cleanrl 同款：噪声加在**动作域**并按动作量程定标。
                    # 动作空间是 σ∈[lo,1)，量程 1；越下界的噪声截回 lo。
                    lo = getattr(net, "lo", None)
                    lo_t = (lo if lo is not None
                            else torch.zeros(SIG_DIM))
                    s_t = s_t + self.expl * torch.as_tensor(
                        self.rng.standard_normal(tuple(s_t.shape)),
                        dtype=torch.float32)
                    s_t = torch.maximum(s_t, lo_t + 1e-4).clamp(
                        max=1.0 - 1e-4)
                sigma = s_t.numpy()
            if was_training:
                net.train()
        if a_field is None:
            return build_action(env, self.field, sigma, self.with_pos)
        a = scale_action(a_field, sigma)
        return a, np.asarray(sigma, dtype=np.float32), a_field


def soft_update(net, target, tau: float) -> None:
    for p, pt in zip(net.parameters(), target.parameters()):
        pt.data.mul_(1.0 - tau).add_(tau * p.data)


# ---------------------------------------------------------------- 训练


def make_env(args: Args):
    spec = baseline_scene(args.preset) or {}
    env = UAVEnvV3(tau_u=args.tau_u, seed=args.seed, scene=args.scene,
                   n_uav=args.n_uav, t_comm=args.t_comm,
                   layout_n=max(12, args.n_uav),
                   use_physics=baseline_scene(args.preset) is None,
                   override_spec=baseline_scene(args.preset),
                   p_ref_dbm=P_REF_DBM, n0_dbm=N0_DBM,
                   demand_scale=spec.get("demand_scale", DEMAND_SCALE_ANCHOR))
    return env


def run_name(args: Args) -> str:
    chan = os.environ.get("PF_CHANNEL", "table")
    sig = os.environ.get("PF_SHADOW_DB", "random").strip().lower()
    stag = "srand" if sig in ("random", "rand", "r") else f"s{sig}"
    parts = [args.scene]
    if args.t_comm != 1:
        parts.append(f"tc{args.t_comm}")
    if args.anchor > 0:
        parts.append(f"anch{args.anchor:g}")
    if args.n_uav != 12:
        parts.append(f"n{args.n_uav}")
    parts.append("pfrl")
    parts.append(f"tau{int(args.tau_u)}_s{args.seed}")
    if chan != "main":
        parts.append(chan)
    parts.append(stag)
    return "_".join(parts)


def train(args: Args) -> Path:
    task = args.task or run_name(args)
    tdir = CKPT_ROOT / task
    tdir.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    env = make_env(args)
    u_star = None if args.anchor > 0 else args.u_star

    actor, a_t = ExecActor(widths=tuple(args.widths), p=args.dropout_p), \
        ExecActor(widths=tuple(args.widths), p=args.dropout_p)
    # TD3 双 critic（cleanrl td3_continuous_action.py）：min(Q1,Q2) 抑制
    # 过估与错位——实测旧单 critic 给坏策略 q≈68、好策略 q≈9（倒挂），
    # actor 跟随伪梯度把 σ 推向敏感区。
    critic, c2, c_t, c2_t = (Critic(act_dim=SIG_DIM,
                                    widths=tuple(args.widths),
                                    p=args.dropout_p)
                             for _ in range(4))
    a_t.eval()
    c_t.eval()
    c2_t.eval()
    a_t.load_state_dict(actor.state_dict())
    c_t.load_state_dict(critic.state_dict())
    c2_t.load_state_dict(c2.state_dict())
    opt_a = torch.optim.Adam(actor.parameters(), lr=args.learning_rate_actor)
    opt_c = torch.optim.Adam(list(critic.parameters())
                             + list(c2.parameters()),
                             lr=args.learning_rate_critic)
    rb = ReplayBuffer(args.buffer_rollouts * args.rollout_len * args.n_uav,
                      OBS_DIM, SIG_DIM)
    field = GlobalField(mode=args.field_mode,
                        map_name=args.map_name or None, u_star=u_star)

    # 2026-11-01 用户裁定：续跑机制证伪（warm_n 重 warm-up 无先例、
    # 实测修复无效），启动必须跑完。本函数恒 fresh：每次启动从 r1 开始，
    # 不加载任何 ckpt/replay/日志状态。--fresh 是显式标志，恒 fresh。
    start = 0
    ckpt = tdir / "latest.pt"

    ctrl_rng = np.random.default_rng(args.seed)
    ctrl = ExecCtrl(actor, field, rng=ctrl_rng,
                    exploration_noise=args.exploration_noise,
                    eta_fixed=args.eta_fixed, with_pos=args.with_pos)

    ret_hist: list[float] = []
    log_path = tdir / "train_log.csv"
    log_f = open(log_path, "w", newline="", encoding="utf-8")
    w = csv.writer(log_f)
    w.writerow(["rollout", "env_steps", "ret_rollout", "mean_ret20",
                "c_loss", "q_mean",
                "sigma_mean", "learn_steps", "seconds",
                "var_u", "u_mean", "sum_t", "n_over"])
    log_f.flush()
    snaps = tdir / "snaps"
    snaps.mkdir(parents=True, exist_ok=True)

    env.reset()
    obs = pf_obs(env)
    a_field = field.act(env, with_pos=args.with_pos)
    rets: list = []
    cl_v = qm_v = sm_v = 0.0
    n_learn = 0
    t0 = time.time()

    seg_keys = ("var_u", "u_mean", "sum_t", "n_over")
    seg: dict = {k: [] for k in seg_keys}
    seg_len: dict = {k: 0 for k in seg_keys}

    def drain_stats() -> None:
        for k in seg_keys:
            arr = getattr(env.stats, k)
            if len(arr) > seg_len[k]:
                seg[k].extend(arr[seg_len[k]:])
                seg_len[k] = len(arr)

    def win(k: str, n: int) -> float:
        a = seg[k][-n:]
        return float(np.mean(a)) if a else float("nan")

    def save(ro: int) -> dict:
        obj = {"policy": actor.state_dict(),
               "critic": critic.state_dict(),
               "critic2": c2.state_dict(),
               "target_policy": a_t.state_dict(),
               "target_critic": c_t.state_dict(),
               "target_critic2": c2_t.state_dict(),
               "opt_a": opt_a.state_dict(),
               "opt_c": opt_c.state_dict(),
               "rollout": ro, "u_star": u_star, "sig_dim": SIG_DIM,
               # 臂配置随权重落盘：评估端必须按同一支场重建，否则
               # --map / --field-mode / --no-pos 臂评估出来的不是该臂。
               "field_mode": args.field_mode,
               "map_name": args.map_name or "",
               "with_pos": bool(args.with_pos),
               "anchor": float(args.anchor),
               # 固定比例臂必须随权重落盘：该臂训练期 σ 恒为设定值，
               # actor 从未作用于环境，评估端用 actor 等于换了策略。
               "eta_fixed": float(args.eta_fixed),
               "widths": list(args.widths),
               "dropout_p": float(args.dropout_p),
               # 速度下限随权重落盘：评估端必须用同一支下限；
               # 缺 sig_lo 的 ckpt 不兼容，不做回退。
               "sig_lo": np.asarray(SIG_LO, dtype=np.float32).tolist()}
        torch.save(obj, ckpt)
        return obj

    total = args.total_rollouts
    n_step = start * args.rollout_len          # 全局环境步（预热/策略更新节拍）
    lo_np = np.asarray(SIG_LO, dtype=np.float32)
    lo_t4 = torch.as_tensor(lo_np)
    n_bad = 0                                   # nan/inf 护栏计数
    for ro in range(start, total):
        n_learn = 0
        for _ in range(args.rollout_len):
            # ALGO LOGIC: 预热段从动作域均匀采样（cleanrl 同款）。
            # 动作域是 σ∈[lo,1)，采样同域：lo + (1−lo)·U(0,1)。
            if n_step < args.learning_starts:
                sigma = (lo_np + (1.0 - lo_np)
                         * rng.random((env.n_uav, SIG_DIM))).astype(np.float32)
                a = scale_action(a_field, sigma)
            else:
                a, sig, _ = ctrl.act(env, deterministic=False,
                                     a_field=a_field)
                sigma = sig.astype(np.float32)
            next_obs, r, done, _ = env.step(a)
            next_a_field = field.act(env, with_pos=args.with_pos)
            nobs = pf_obs(env)
            # nan/inf 护栏（2026-11-01）：n100 曾出现整轮 nan 自愈轮
            # （r125/r250）。非有限转移一旦被采样会永久毒死网络——
            # 坚决不入池，只告警。物理来源另查（见分析报告 E6.4）。
            if (np.isfinite(nobs).all() and np.isfinite(r).all()
                    and np.isfinite(a).all() and np.isfinite(a_field).all()
                    and np.isfinite(next_a_field).all()
                    and np.isfinite(sigma).all()):
                rb.add(obs, sigma, r, nobs, float(done))
                rets.append(r)
            else:
                n_bad += 1
                print(f"[{task}] WARN 非有限转移 t={env.t} "
                      f"obs_finite={np.isfinite(nobs).all()} "
                      f"r_finite={np.isfinite(r).all()} 累计 {n_bad}，"
                      f"该步不入池", flush=True)
            obs = nobs
            a_field = next_a_field
            sm_v = float(np.mean(sigma))
            n_step += 1
            # ALGO LOGIC: TD3 更新（cleanrl td3_continuous_action.py）。
            # fresh 语义：learning_starts 后 buffer 必然够 batch。
            if n_step > args.learning_starts:
                o, a, rr, no, dn = rb.sample(args.batch_size, rng)
                o_t, no_t = torch.as_tensor(o), torch.as_tensor(no)
                a_in = torch.as_tensor(a)
                r_t, dn_t = torch.as_tensor(rr), torch.as_tensor(dn)
                actor.train()
                critic.train()
                c2.train()
                with torch.no_grad():
                    # 目标策略平滑：在 σ 域加噪、截到 σ 域上下界。
                    noise = (torch.randn_like(a_in) * args.policy_noise
                             ).clamp(-args.noise_clip, args.noise_clip)
                    a_next = torch.maximum(
                        a_t(no_t) + noise, lo_t4 + 1e-4).clamp(max=1.0 - 1e-4)
                    q1n = c_t(no_t, a_next)
                    q2n = c2_t(no_t, a_next)
                    y = r_t + args.gamma * torch.minimum(q1n, q2n) * (1.0 - dn_t)
                q1 = critic(o_t, a_in)
                q2 = c2(o_t, a_in)
                c_loss = F.mse_loss(q1, y) + F.mse_loss(q2, y)
                opt_c.zero_grad()
                c_loss.backward()
                nn.utils.clip_grad_norm_(
                    list(critic.parameters()) + list(c2.parameters()), 10.0)
                opt_c.step()
                # cleanrl：actor 与目标网络按 policy_frequency 延迟更新
                if n_step % args.policy_frequency == 0:
                    a_pred = actor(o_t)
                    a_loss = -critic(o_t, a_pred).mean()
                    opt_a.zero_grad()
                    a_loss.backward()
                    nn.utils.clip_grad_norm_(actor.parameters(), 10.0)
                    opt_a.step()
                    soft_update(actor, a_t, args.tau)
                    soft_update(critic, c_t, args.tau)
                    soft_update(c2, c2_t, args.tau)
                cl_v, qm_v = float(c_loss.detach()), float(q1.detach().mean())
                n_learn += 1
            # TRY NOT TO MODIFY: 回合边界
            if done:
                drain_stats()
                env.reset()
                obs = pf_obs(env)
                a_field = field.act(env, with_pos=args.with_pos)
                for k in seg_keys:
                    seg[k].clear()
                    seg_len[k] = 0
        drain_stats()
        sd_save = save(ro)
        if (ro + 1) % args.ckpt_every == 0 or ro + 1 == total:
            torch.save(sd_save, snaps / f"snap_r{ro + 1}.pt")
            (tdir / "progress.json").write_text(json.dumps({
                "task": task, "rollout": ro + 1, "total": total,
                "env_steps": (ro + 1) * args.rollout_len,
                "mean_ret_recent": float(np.mean(rets[-EPISODE_LEN:])),
                "c_loss": round(cl_v, 5), "q_mean": round(qm_v, 3),
                "learn_steps": n_learn, "sigma_mean": round(sm_v, 4),
                "n_bad_transitions": n_bad,
                "var_u": round(win("var_u", args.rollout_len), 6),
                "u_mean": round(win("u_mean", args.rollout_len), 4),
                "sum_t": round(win("sum_t", args.rollout_len), 1),
                "n_over": round(win("n_over", args.rollout_len), 2),
                "elapsed_min": round((time.time() - t0) / 60, 1),
            }, ensure_ascii=False), encoding="utf-8")
        w.writerow([ro + 1, (ro + 1) * args.rollout_len,
                    round(float(np.mean(rets[-args.rollout_len:])), 4),
                    round(float(np.mean(rets[-20 * args.rollout_len:])), 4),
                    round(cl_v, 5), round(qm_v, 3), round(sm_v, 4), n_learn,
                    round(time.time() - t0, 1),
                    round(win("var_u", args.rollout_len), 6),
                    round(win("u_mean", args.rollout_len), 4),
                    round(win("sum_t", args.rollout_len), 1),
                    round(win("n_over", args.rollout_len), 2)])
        log_f.flush()
        if True:  # 逐步打印：每轮一行，不静默
            print(f"[{task}] {ro+1}/{total} "
                  f"ret={np.mean(rets[-20*args.rollout_len:]):.3f} "
                  f"cl={cl_v:.4f} q={qm_v:.2f} sigma={sm_v:.3f} "
                  f"learn={n_learn} {(time.time()-t0)/60:.1f}min", flush=True)
        ret_r = float(np.mean(rets[-args.rollout_len:]))
        ret_hist.append(ret_r)
        # 早停：近 S 轮均值 vs 近 W 轮均值。
        # 只有两者相差在 ±EARLY_THR 内（无提升、也无明显退化）才停；
        # 近 S 轮明显低于近 W 轮 = 崩溃，继续训练，不许早停。
        W = args.early_win
        S = args.early_short
        t = ro + 1
        if t % 25 == 0 and t >= 2 * W:
            short = float(np.mean(ret_hist[-S:]))
            long = float(np.mean(ret_hist[-W:]))
            diff = short - long
            band = args.early_thr * max(abs(long), 1e-6)
            if abs(diff) <= band:
                print(f"[{task}] 早停@{ro+1}轮 近{S}轮={short:.4f} "
                      f"近{W}轮={long:.4f} 差={diff:+.4f} "
                      f"|差|<{args.early_thr:.1%} 无提升", flush=True)
                save(ro)
                (tdir / "progress.json").write_text(json.dumps({
                    "task": task, "rollout": ro + 1, "total": total,
                    "env_steps": (ro + 1) * args.rollout_len, "early_stop": True,
                    "mean_ret_recent": float(np.mean(rets[-EPISODE_LEN:])),
                    "elapsed_min": round((time.time() - t0) / 60, 1),
                }, ensure_ascii=False), encoding="utf-8")
                break
    log_f.close()
    env.close()
    return tdir


# ---------------------------------------------------------------- CLI


def parse_args() -> Args:
    d = Args()
    ap = argparse.ArgumentParser(description="PF-Exec 训练")
    ap.add_argument("--scene", default=d.scene)
    ap.add_argument("--n-uav", type=int, default=d.n_uav)
    ap.add_argument("--seed", type=int, default=d.seed)
    ap.add_argument("--tau", type=float, default=d.tau_u)
    ap.add_argument("--tcomm", type=int, default=d.t_comm)
    ap.add_argument("--anchor", type=float, default=d.anchor)
    ap.add_argument("--total-rollouts", type=int, default=d.total_rollouts)
    ap.add_argument("--u-star", type=float, default=d.u_star)
    ap.add_argument("--task", default=d.task)
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--early-win", type=int, default=d.early_win)
    ap.add_argument("--early-short", type=int, default=d.early_short)
    ap.add_argument("--early-thr", type=float, default=d.early_thr)
    ap.add_argument("--eta-fixed", type=float, default=d.eta_fixed)
    ap.add_argument("--field-mode", default=d.field_mode,
                    choices=["closed", "natgrad", "grad"])
    ap.add_argument("--map", default=d.map_name,
                    choices=["A_clip3", "B_clip30", "C_logc"])
    ap.add_argument("--no-pos", action="store_true")
    ap.add_argument("--no-anchor", action="store_true")
    ap.add_argument("--preset", default=d.preset)
    ap.add_argument("--mode", default="pfrl", choices=["pfrl"])
    a = ap.parse_args()
    d.scene, d.n_uav, d.seed, d.tau_u = a.scene, a.n_uav, a.seed, a.tau
    d.t_comm, d.anchor = a.tcomm, a.anchor
    d.total_rollouts, d.u_star = a.total_rollouts, a.u_star
    d.task, d.fresh = a.task or "", a.fresh
    d.early_win, d.early_short, d.early_thr = (
        a.early_win, a.early_short, a.early_thr)
    d.eta_fixed = a.eta_fixed
    d.field_mode, d.map_name = a.field_mode, a.map
    d.with_pos = not a.no_pos
    if a.no_anchor:
        d.anchor = 1.0            # anchor>0 → u_star=None
    d.preset = a.preset or ""
    return d


if __name__ == "__main__":
    train(parse_args())
