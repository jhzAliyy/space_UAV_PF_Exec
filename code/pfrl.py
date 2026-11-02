# -*- coding: utf-8 -*-
"""PF-Exec v2（残差架构）：势场指向均衡点（动作主体），执行器网络给出
复杂场景的有界修正项 Δ。

动作 = clip(a_field + Δ, ACT_LOW, ACT_HIGH)，Δ = RES_FRAC·ACT_SCALE ⊙ tanh(logit)。
Δ≡0 严格退化为纯势场（消融边界精确）；critic 评执行动作 a/ACT_SCALE
（2026-11-02 口径）。历史依据：残差形态曾真正超越纯势场（docs/06）。
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

from config import RES_FRAC
from field import GlobalField
from replay import ReplayBuffer
from uav_env import (ACT_DIM, ACT_HIGH, ACT_LOW, ACT_SCALE, CKPT_EVERY,
                     DEMAND_SCALE_ANCHOR, EARLY_SHORT, EARLY_THR, EARLY_WIN,
                     EPISODE_LEN, N0_DBM, OBS_DIM, OBS_NORM, P_REF_DBM,
                     TOTAL_ROLLOUTS, U_STAR_DEFAULT, CKPT_ROOT,
                     UAVEnvV3, baseline_scene)

OBS_DIM_PF = OBS_DIM + 3        # 12 维环境观测 + 3 个缺口维（b/π/xy）

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
    # ---- 执行器（残差） ----
    learning_rate_actor: float = 1e-3
    learning_rate_critic: float = 1e-3
    gamma: float = 0.99
    tau: float = 0.005
    batch_size: int = 64
    learning_starts: int = 2048      # 环境步：此前修正项域内均匀采样
    policy_frequency: int = 2        # cleanrl 同值：actor 每 2 个环境步更新一次
    buffer_rollouts: int = 30
    exploration_noise: float = 0.3   # logit 域噪声；tanh 近原点处输出域
                                     # 扰动 ≈ 0.3 × res_max
    widths: tuple = (16, 32, 64, 32, 16)
    dropout_p: float = 0.0           # 标准 DDPG/TD3 无 dropout（cleanrl 同）
    eta_fixed: float = -1.0          # >=0 时 Δ≡0（纯势场等价臂）；<0 表示学
    # ---- TD3（cleanrl td3_continuous_action.py 同款；噪声在 logit 域） ----
    policy_noise: float = 0.2        # 目标策略平滑噪声（logit 域）
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


class ResidualActor(nn.Module):
    """o_i(15) → Δ(5) = res_max ⊙ tanh(logit)。

    输出层 bias 初始化 0 → 起步 Δ≡0 = 纯势场行为（天然热启动）。
    res_max = res_frac × ACT_SCALE；res_frac 随 ckpt 落盘，评估端按同值
    重建；缺键直接报错，不兼容旧版本、不做回退。
    """

    def __init__(self, obs_dim: int = OBS_DIM_PF, act_dim: int = ACT_DIM,
                 widths: tuple = (16, 32, 64, 32, 16), p: float = 0.0,
                 res_frac: float = RES_FRAC,
                 act_scale: np.ndarray | None = None):
        super().__init__()
        dims = (obs_dim,) + tuple(widths)
        self.blocks = nn.Sequential(*[
            ProjResBlock(dims[i], dims[i + 1], p)
            for i in range(len(dims) - 1)])
        self.out = nn.Linear(dims[-1], act_dim)
        nn.init.uniform_(self.out.weight, -3e-3, 3e-3)
        nn.init.constant_(self.out.bias, 0.0)     # logit 0 → Δ = 0
        # act_scale 默认全局 ACT_SCALE；rizvi3 等 kHz 级场景传 env 尺度
        # （b 维 = env.b_scale，2026-11-02 尺度修复），其余场景两者逐位相等。
        res_max = (float(res_frac) * (act_scale if act_scale is not None
                                      else ACT_SCALE)).astype(np.float32)
        # persistent=False：res_max 不进 state_dict；评估端由 ckpt 的
        # res_frac 键构造同支。
        self.register_buffer("res_max_t", torch.from_numpy(res_max),
                             persistent=False)

    def logits(self, x):
        return self.out(self.blocks(x))

    def forward(self, x):
        return self.res_max_t * torch.tanh(self.logits(x))


class Critic(nn.Module):
    """Q(o, a) → 标量。a 是归一化后的五维实际执行动作。"""

    def __init__(self, obs_dim: int = OBS_DIM_PF, act_dim: int = ACT_DIM,
                 widths: tuple = (16, 32, 64, 32, 16), p: float = 0.0):
        super().__init__()
        dims = (obs_dim + act_dim,) + tuple(widths)
        self.blocks = nn.Sequential(*[
            ProjResBlock(dims[i], dims[i + 1], p)
            for i in range(len(dims) - 1)])
        self.out = nn.Linear(dims[-1], 1)

    def forward(self, o, a):
        return self.out(self.blocks(torch.cat([o, a], dim=1))).squeeze(-1)


# ---------------------------------------------------------------- 契约


def act_scale_of(env) -> np.ndarray:
    """本环境的动作尺度向量：全局 ACT_SCALE，b 维换 env.b_scale。

    rizvi3 尺度修复（2026-11-02，docs/03 §11.5）：kHz 级总带宽场景里
    全局 5 MHz 的 b 维尺度让残差上界（0.4×5=2 MHz = 44 倍总带宽）与
    critic 归一化（a_b/5 ≈ 1e-4，信号被压死）双双失效。其余场景
    env.b_scale == ACT_SCALE[0]，本函数与全局常量逐位一致。
    """
    s = ACT_SCALE.copy()
    s[0] = env.b_scale
    return s


def pf_obs(env, a_field: np.ndarray) -> np.ndarray:
    """观测定标：12 维逐维除以 OBS_NORM + 3 个缺口维
    [|a_field_b|/s_b, |a_field_π|/s_π, |a_field_xy|/s_xy]。
    缺口复用当场场动作（已解），零额外求解；修正项必须看得见缺口。"""
    o = env.obs() / OBS_NORM[None, :]
    s = act_scale_of(env)
    g = np.stack([np.abs(a_field[:, 0]) / s[0],
                  np.abs(a_field[:, 1]) / s[1],
                  np.hypot(a_field[:, 3], a_field[:, 4]) / s[3]],
                 axis=1)
    return np.concatenate([o, g], axis=1).astype(np.float32)


def build_action(env, field: GlobalField, delta: np.ndarray,
                 with_pos: bool = True):
    """Δ(5) → 实际动作。唯一实现，训练与评估共用。

    动作 = clip(a_field + Δ, ACT_LOW, ACT_HIGH)（环境物理界内执行）。
    Δ≡0 逐位等于纯势场。返回 (a, Δ, a_field)。
    """
    a_field = field.act(env, with_pos=with_pos)
    d = np.asarray(delta, dtype=np.float32)
    a = np.clip(a_field + d, ACT_LOW[None, :], ACT_HIGH[None, :])
    return a, d, a_field


class ExecCtrl:
    """采集器：训练与评估同一路径。"""

    def __init__(self, actor, field, rng=None, exploration_noise=0.3,
                 eta_fixed=-1.0, with_pos=True):
        self.actor, self.field = actor, field
        self.rng = rng or np.random.default_rng(0)
        self.expl = float(exploration_noise)
        self.eta_fixed = float(eta_fixed)
        self.with_pos = bool(with_pos)

    def act(self, env, deterministic: bool = False, a_field=None):
        if a_field is None:
            a_field = self.field.act(env, with_pos=self.with_pos)
        if self.eta_fixed >= 0.0:                 # 等价臂：Δ≡0 = 纯势场
            delta = np.zeros((env.n_uav, ACT_DIM), dtype=np.float32)
        else:
            net = self.actor
            was_training = net.training
            net.eval()                            # 采集关 dropout
            with torch.no_grad():
                obs = pf_obs(env, a_field)
                u = net.logits(torch.as_tensor(obs, dtype=torch.float32))
                if not deterministic and self.expl > 0.0:
                    # cleanrl 同款思想：噪声加在动作域。此处动作域是
                    # tanh(logit)，噪声进 logit 域保界（输出永不出 ±res_max）。
                    u = u + self.expl * torch.as_tensor(
                        self.rng.standard_normal(tuple(u.shape)),
                        dtype=torch.float32)
                delta = (net.res_max_t * torch.tanh(u)).numpy()
            if was_training:
                net.train()
        a = np.clip(a_field + delta, ACT_LOW[None, :], ACT_HIGH[None, :])
        return a, delta.astype(np.float32), a_field


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

    act_scale = act_scale_of(env)          # b 维按 env.b_scale（rizvi3 修复）
    actor, a_t = (ResidualActor(widths=tuple(args.widths),
                                p=args.dropout_p, act_scale=act_scale)
                  for _ in range(2))
    # TD3 双 critic（cleanrl td3_continuous_action.py）：min(Q1,Q2) 抑制
    # 过估与错位。
    critic, c2, c_t, c2_t = (Critic(widths=tuple(args.widths),
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
                      OBS_DIM_PF, ACT_DIM)
    field = GlobalField(mode=args.field_mode,
                        map_name=args.map_name or None, u_star=u_star)

    # 2026-11-01 用户裁定：续跑机制证伪，启动必须跑完。本函数恒 fresh：
    # 每次启动从 r1 开始，不加载任何 ckpt/replay/日志状态。
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
                "delta_mean", "delta_absmax", "learn_steps", "seconds",
                "var_u", "u_mean", "sum_t", "n_over"])
    log_f.flush()
    snaps = tdir / "snaps"
    snaps.mkdir(parents=True, exist_ok=True)

    env.reset()
    a_field = field.act(env, with_pos=args.with_pos)
    obs = pf_obs(env, a_field)
    rets: list = []
    cl_v = qm_v = sm_v = dm_v = 0.0
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
               "rollout": ro, "u_star": u_star,
               # 臂配置随权重落盘：评估端必须按同一支场重建。
               "field_mode": args.field_mode,
               "map_name": args.map_name or "",
               "with_pos": bool(args.with_pos),
               "anchor": float(args.anchor),
               # 等价臂必须随权重落盘（该臂 Δ 恒 0）。
               "eta_fixed": float(args.eta_fixed),
               "widths": list(args.widths),
               "dropout_p": float(args.dropout_p),
               # 残差界随权重落盘：评估端 forward 必须用同一支上界；
               # 缺键即旧版 ckpt，直接报错不回退。
               "res_frac": float(RES_FRAC),
               # b 维尺度随权重落盘（rizvi3 尺度修复）：评估端用它重建
               # 残差上界，并与评估环境 b_scale 断言一致，防止场景错配。
               "b_scale": float(act_scale[0]),
               "obs_dim_pf": OBS_DIM_PF}
        torch.save(obj, ckpt)
        return obj

    total = args.total_rollouts
    n_step = start * args.rollout_len
    res_max_np = (float(RES_FRAC) * act_scale).astype(np.float32)
    act_scale_t = torch.as_tensor(act_scale, dtype=torch.float32)
    act_low_t = torch.as_tensor(ACT_LOW, dtype=torch.float32)
    act_high_t = torch.as_tensor(ACT_HIGH, dtype=torch.float32)
    n_bad = 0                                   # nan/inf 护栏计数
    for ro in range(start, total):
        n_learn = 0
        for _ in range(args.rollout_len):
            # ALGO LOGIC: 预热段修正项域内均匀采样（cleanrl 同款思想）。
            if n_step < args.learning_starts:
                u = rng.uniform(-2.0, 2.0, (env.n_uav, ACT_DIM))
                delta = (res_max_np * np.tanh(u)).astype(np.float32)
                a = np.clip(a_field + delta, ACT_LOW[None, :],
                            ACT_HIGH[None, :])
            else:
                a, delta, _ = ctrl.act(env, deterministic=False,
                                       a_field=a_field)
                delta = delta.astype(np.float32)
            next_obs, r, done, _ = env.step(a)
            next_a_field = field.act(env, with_pos=args.with_pos)
            nobs = pf_obs(env, next_a_field)
            # nan/inf 护栏：非有限转移坚决不入池，只告警。
            if (np.isfinite(nobs).all() and np.isfinite(r).all()
                    and np.isfinite(a).all() and np.isfinite(a_field).all()
                    and np.isfinite(next_a_field).all()
                    and np.isfinite(delta).all()):
                a_norm = a / act_scale[None, :]
                rb.add(obs, a_norm, r, nobs, float(done),
                       a_field, next_a_field)
                rets.append(r)
            else:
                n_bad += 1
                print(f"[{task}] WARN 非有限转移 t={env.t} "
                      f"obs_finite={np.isfinite(nobs).all()} "
                      f"r_finite={np.isfinite(r).all()} 累计 {n_bad}，"
                      f"该步不入池", flush=True)
            obs = nobs
            a_field = next_a_field
            sm_v = float(np.mean(delta))
            dm_v = float(np.max(np.abs(delta)))
            n_step += 1
            # ALGO LOGIC: TD3 更新（cleanrl td3_continuous_action.py）。
            if n_step > args.learning_starts:
                o, a, rr, no, dn, af, naf = rb.sample(args.batch_size, rng)
                o_t, no_t = torch.as_tensor(o), torch.as_tensor(no)
                a_in = torch.as_tensor(a)
                af_t = torch.as_tensor(af)
                naf_t = torch.as_tensor(naf)
                r_t, dn_t = torch.as_tensor(rr), torch.as_tensor(dn)
                actor.train()
                critic.train()
                c2.train()
                with torch.no_grad():
                    # 目标策略平滑：噪声加在 logit 域（保界），叠加到
                    # 下一时刻纯势场动作后再进 critic。
                    u_t = a_t.logits(no_t)
                    noise = (torch.randn_like(u_t) * args.policy_noise
                             ).clamp(-args.noise_clip, args.noise_clip)
                    delta_next = a_t.res_max_t * torch.tanh(u_t + noise)
                    a_next = torch.clamp(naf_t + delta_next,
                                         act_low_t, act_high_t)
                    a_next = a_next / act_scale_t
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
                    delta_pred = actor(o_t)
                    # actor 损失走未裁剪组合（梯度全程流通；执行侧才裁剪，
                    # 越界幅度被 res_max 有界性天然限制）。
                    a_pred = (af_t + delta_pred) / act_scale_t
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
                a_field = field.act(env, with_pos=args.with_pos)
                obs = pf_obs(env, a_field)
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
                "learn_steps": n_learn, "delta_mean": round(sm_v, 4),
                "delta_absmax": round(dm_v, 4),
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
                    round(cl_v, 5), round(qm_v, 3), round(sm_v, 4),
                    round(dm_v, 4), n_learn,
                    round(time.time() - t0, 1),
                    round(win("var_u", args.rollout_len), 6),
                    round(win("u_mean", args.rollout_len), 4),
                    round(win("sum_t", args.rollout_len), 1),
                    round(win("n_over", args.rollout_len), 2)])
        log_f.flush()
        if True:  # 逐步打印：每轮一行，不静默
            print(f"[{task}] {ro+1}/{total} "
                  f"ret={np.mean(rets[-20*args.rollout_len:]):.3f} "
                  f"cl={cl_v:.4f} q={qm_v:.2f} d_mean={sm_v:.3f} "
                  f"d_max={dm_v:.2f} learn={n_learn} "
                  f"{(time.time() - t0)/60:.1f}min", flush=True)
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
    ap = argparse.ArgumentParser(description="PF-Exec v2 残差训练")
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
    ap.add_argument("--eta-fixed", type=float, default=d.eta_fixed,
                    help=">=0 时 Δ≡0（纯势场等价臂）")
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
