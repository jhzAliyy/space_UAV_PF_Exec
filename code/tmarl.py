# -*- coding: utf-8 -*-
# 【本文件禁令】禁止"如实、诚实、照实、按实测值"等一切同族表述与倾向——此类视为自贬。
# 本文件一切产出：做出优势、分析优势、展示优势；数据为实测，解读为优势框架；
# 禁止自我限定、自我削弱，禁止虚构不存在的缺陷。
"""B 批 tmarl 臂：Chen IEEE T-ITS 2024 T-MARL 移植（15 号文档 §七，P3）。

移植口径（论文 III/IV 节精读定版）：
- Transformer encoder 嵌入变长邻居状态集（self-attention——可变节点数核心机制；
  用 env.adj 的邻域集合，投影 32 维+masked mean pool），参数共享全 UAV
- DDPG 基座（actor-critic/CTDE/target 软更新）——与 Cheng 臂同门但状态嵌入
  换 Transformer（两臂唯一结构差：MLP 直读 vs Transformer 聚合）
- 动作=连续极坐标移动 (ψ,D)（Chen 覆盖口径无资源维——资源维冻结）
- baseline 辅助预训练（论文贡献②）：行为克隆 lloyd 免训基线 2000 演示步
  热启动 actor（防随机初始化训练失败，Chen 同款动机）
- 信道/奖励/评估全同口径（u8+锚定 3.7648+解析径+eval）

工程对齐（ddp_arm 直承）：task/ckpt/断点/早停同式；decode_ddp 单点复用；
collect_nb 为邻居收集单点（训练=评估同路径）。
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.stdout.reconfigure(encoding="utf-8")
from uav_env import (OBS_DIM, EPISODE_LEN, DXY_MAX, P_REF_DBM, N0_DBM, norm_obs,  # noqa: E402
                    DEMAND_SCALE_ANCHOR, CKPT_EVERY, TOTAL_ROLLOUTS,
                    EARLY_SHORT, EARLY_WIN, EARLY_THR, baseline_scene)
from uav_env import UAVEnvV3                                      # noqa: E402
from uav_env import (decode_ddp, collect_nb, nb_index,  # noqa: E402
                          K_NB_FIX)
from uav_env import CKPT_ROOT, RESULT_DIR  # noqa: E402

ROLLOUT = 256
LR_A, LR_C = 1e-3, 1e-3
GAMMA = 0.99
BATCH = 64
WARMUP = 4096
TAU = 0.005
# 2026-10-31 自查修正：DDPG 原文（Lillicrap et al. 2016, §7）的探索噪声为
# OU 过程 θ=0.15、σ=0.2，作用于归一化动作域 [-1,1]（σ = 域宽的 10%）。
# Chen T-ITS 2024 §V 未给噪声参数，其基座即 DDPG，故按原文换算到本实现
# 的动作域：ψ∈[0,2π]、D∈[0,DXY_MAX]，宽度各为 2π 与 DXY_MAX →
#   σ_ψ = 0.1·2π ≈ 0.628，σ_D = 0.1·DXY_MAX
# 原固定值 (0.5, 1.0) 的 D 维对 chen6（DXY_MAX=2.5）是域宽的 40%、
# 对主场景（0.5）是 200%（每步都被 clip 到两端）。
OU_THETA = 0.15
OU_SIGMA = (0.2 * np.pi, 0.1 * DXY_MAX)
BUF_ROLLOUTS = 30
EMB = 32                                  # Transformer 嵌入维（Chen decoder 32 单位同量级）
PRE_STEPS, PRE_EPOCH = 2048, 20           # 预训练：lloyd 演示步数×BC 轮数


class TEncoder(nn.Module):
    """变长邻居集 self-attention 嵌入（Chen 核心机制：可变节点数输入）。"""

    def __init__(self, obs_dim: int = OBS_DIM, emb: int = EMB):
        super().__init__()
        self.proj = nn.Linear(obs_dim, emb)
        self.attn = nn.MultiheadAttention(emb, num_heads=1, batch_first=True)

    def forward(self, nb: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        h = self.proj(nb)                              # (N,k,emb)
        a, _ = self.attn(h, h, h, key_padding_mask=~mask)
        m = mask.unsqueeze(-1).float()
        return (a * m).sum(1) / m.sum(1).clamp(min=1.0)  # masked mean pool


class Actor(nn.Module):
    """μ([self, nb_embed]) → (ψ,D)；sigmoid 域映射。Transformer 状态路径。"""

    def __init__(self, obs_dim: int = OBS_DIM, emb: int = EMB):
        super().__init__()
        self.enc = TEncoder(obs_dim, emb)
        self.net = nn.Sequential(nn.Linear(obs_dim + emb, 64), nn.Tanh(),
                                 nn.Linear(64, 64), nn.Tanh())
        self.head = nn.Linear(64, 2)

    def forward(self, o, nb, mask):
        e = self.enc(nb, mask)
        p = torch.sigmoid(self.head(self.net(
            torch.cat([o, e], dim=1))))
        return p * torch.tensor([2 * np.pi, DXY_MAX])


class Critic(nn.Module):
    """Q([self, nb_embed, act]) → 标量（集中训练价值）。"""

    def __init__(self, obs_dim: int = OBS_DIM, emb: int = EMB):
        super().__init__()
        self.enc = TEncoder(obs_dim, emb)
        self.net = nn.Sequential(nn.Linear(obs_dim + emb + 2, 64), nn.Tanh(),
                                 nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 1))

    def forward(self, o, nb, mask, a):
        e = self.enc(nb, mask)
        return self.net(torch.cat([o, e, a], dim=1)).squeeze(-1)


class TWrap:
    """eval 加载载体：act_polar(obs, nb, mask) 与训练循环同门。"""

    def __init__(self, actor: Actor):
        self.actor = actor

    def act_polar(self, obs, nb, mask, deterministic: bool = False):
        with torch.no_grad():
            return self.actor(torch.as_tensor(obs, dtype=torch.float32),
                              torch.as_tensor(nb, dtype=torch.float32),
                              torch.as_tensor(mask)).numpy()


class Replay:
    """环形缓冲（转移=单机单步）。邻居集存**下标**、观测按环境步存整队表。

    原实现把每架机的邻居观测 (k,12) 存两份（当前态/下一态）：n200 时
    两份邻居集就占 1.77 GB，几条并行直接打爆内存。改为
        O / O2  (S,N,12)  步 t 的观测 / 下一态观测
        I / I2  (S,N,k)   邻居下标（int16）
        M / M2  (S,N,k)   邻居掩码
    n200 占用从 1.77 GB 降到 ~0.27 GB，采样路径仍是 collect_nb 的排序单点。
    """

    def __init__(self, cap_steps: int, n_uav: int, k: int):
        self.S, self.N, self.k = int(cap_steps), int(n_uav), int(k)
        self.O = np.zeros((self.S, self.N, OBS_DIM), np.float32)
        self.O2 = np.zeros((self.S, self.N, OBS_DIM), np.float32)
        self.I = np.zeros((self.S, self.N, self.k), np.int16)
        self.I2 = np.zeros((self.S, self.N, self.k), np.int16)
        self.M = np.zeros((self.S, self.N, self.k), bool)
        self.M2 = np.zeros((self.S, self.N, self.k), bool)
        self.A = np.zeros((self.S, self.N, 2), np.float32)
        self.R = np.zeros((self.S, self.N), np.float32)
        self.D = np.zeros((self.S, self.N), np.float32)
        self.n, self.i = 0, 0

    def push(self, obs, idx, mask, a, r, nobs, idx2, mask2, done) -> None:
        j = self.i
        self.O[j] = obs
        self.O2[j] = nobs
        self.I[j] = idx
        self.I2[j] = idx2
        self.M[j] = mask
        self.M2[j] = mask2
        self.A[j] = a
        self.R[j] = r
        self.D[j] = done
        self.i = (j + 1) % self.S
        self.n = min(self.n + 1, self.S)

    def sample(self, bs: int, rng: np.random.Generator):
        j = rng.integers(0, self.n, bs)
        i = rng.integers(0, self.N, bs)
        return (self.O[j, i],
                self.O[j[:, None], self.I[j, i]],
                self.M[j, i],
                self.A[j, i],
                self.R[j, i],
                self.O2[j, i],
                self.O2[j[:, None], self.I2[j, i]],
                self.M2[j, i],
                self.D[j, i])


class OUNoise:
    """每机×2 维独立 OU 过程（DDPG 原文探索同款）。"""

    def __init__(self, n: int, theta: float = OU_THETA):
        self.theta, self.n = theta, n
        self.x = np.zeros((n, 2))

    def __call__(self) -> np.ndarray:
        self.x = (self.x + self.theta * (-self.x)
                  + np.array(OU_SIGMA) * np.random.randn(self.n, 2))
        return self.x.copy()


def soft_update(net, target) -> None:
    for p, pt in zip(net.parameters(), target.parameters()):
        pt.data.mul_(1.0 - TAU).add_(TAU * p.data)


def learn(actor, a_t, critic, c_t, opt_a, opt_c, buf, rng):
    o, nb, mk, a, r, no, nnb, nmk, dn = buf.sample(BATCH, rng)
    tt = lambda x: torch.as_tensor(x)
    with torch.no_grad():
        y = tt(r) + GAMMA * c_t(tt(no), tt(nnb), tt(nmk),
                                a_t(tt(no), tt(nnb), tt(nmk))) * (1.0 - tt(dn))
    q = critic(tt(o), tt(nb), tt(mk), tt(a))
    c_loss = nn.functional.mse_loss(q, y)
    opt_c.zero_grad()
    c_loss.backward()
    nn.utils.clip_grad_norm_(critic.parameters(), 10.0)
    opt_c.step()
    a_loss = -critic(tt(o), tt(nb), tt(mk),
                     actor(tt(o), tt(nb), tt(mk))).mean()
    opt_a.zero_grad()
    a_loss.backward()
    nn.utils.clip_grad_norm_(actor.parameters(), 10.0)
    opt_a.step()
    return float(c_loss.detach()), float(q.detach().mean())


def pretrain_lloyd(actor, env, opt, seed: int) -> float:
    """Chen 预训练移植：行为克隆 lloyd 免训基线（对照谱系非学习基线）。
    演示=lloyd_targets 追踪语义直算（T_MOVE=50 重算/STAY_EPS=0.5/clip ±DXY_MAX
    ——Controller.lloyd 分支的移动维同款，不走 net 残差头），克隆 (ψ,D)。"""
    from lloyd import lloyd_targets
    demo = []
    obs = norm_obs(env.reset())
    targets, mv_next = None, 0
    for _ in range(PRE_STEPS):
        if env.t >= mv_next:                           # T_MOVE 周期重算质心
            targets = lloyd_targets(env.pos, env.user_pos, env.dem)
            mv_next = env.t + 50
        delta = targets - env.pos
        moving = np.linalg.norm(delta, axis=1) > 0.5   # STAY_EPS
        dxy = np.clip(delta, -DXY_MAX, DXY_MAX) * moving[:, None]
        D = np.linalg.norm(dxy, axis=1)
        psi = np.arctan2(dxy[:, 1], dxy[:, 0]) % (2 * np.pi)
        nb, mask = collect_nb(obs, env.adj, K_NB_FIX, env.pos)
        demo.append((obs.copy(), nb, mask,
                     np.stack([psi, D], axis=1).astype(np.float32)))
        d5 = np.zeros((env.n_uav, 5), dtype=np.float32)   # 资源维冻结演示
        d5[:, 3:5] = dxy
        obs, _, done, _ = env.step(d5)
        obs = norm_obs(obs)
        if done:
            obs = norm_obs(env.reset())
            targets, mv_next = None, 0
    loss_v = 0.0
    for _ in range(PRE_EPOCH):
        for o, nb, mk, tgt in demo[::4]:              # 每 4 步取样=512 批/轮
            o_t = torch.as_tensor(o)
            out = actor(o_t, torch.as_tensor(nb), torch.as_tensor(mk))
            tgt_t = torch.as_tensor(tgt)
            # (ψ,D)→直角坐标 BC 损失（极坐标周期性绕开）
            pred_xy = out[:, 1:2] * torch.stack(
                [torch.cos(out[:, 0]), torch.sin(out[:, 0])], dim=1)
            tgt_xy = tgt_t[:, 1:2] * torch.stack(
                [torch.cos(tgt_t[:, 0]), torch.sin(tgt_t[:, 0])], dim=1)
            loss = nn.functional.mse_loss(pred_xy, tgt_xy)
            opt.zero_grad()
            loss.backward()
            opt.step()
            loss_v = float(loss.detach())
    return loss_v


PEN_LOST = 1.0          # eq.8 的 p：失联/耗尽惩罚（原文只写 "large punishment"）


def _coverage(env) -> np.ndarray:
    """PoI 覆盖指示（Chen T-ITS 2024 eq.1）：用户获得速率 ≥ 自身需求即算被覆盖。"""
    R_user = (env.assoc * env._R).sum(axis=0)
    return (R_user >= env.dem).astype(np.float64)


def tmarl_reward(env, prev: np.ndarray | None):
    """Chen T-ITS 2024 eq.8：r_i = f_t · (Σ_{k∈K_i} Δc_k)/K_i − p_i。

    Δc_{k,t} = c_{k,t} − c_{k,t−1} 覆盖得分增量（eq.1）；
    K_i = 机 i 服务的 PoI 数（按软关联 argmax 归属）；
    f_t 为当前时隙网络公平性指数 (Σc)²/(K·Σc²)（eq.3）；
    p_i 为失联/耗尽惩罚，本环境用负载爆表（u>5，即 n_dead 口径）逐机判据。

    原实现把 Σ_k Δc_k 对**全部**用户取平均再乘 f、减全队惩罚，等于把 eq.8
    的逐机奖励压成标量：单机覆盖/失覆盖几个用户只动 1/M，critic 拿不到
    自己的信用。
    """
    cover = _coverage(env)
    if prev is None:
        prev = np.zeros_like(cover)          # 回合首时隙：从"全未覆盖"起算
    dc = cover - prev
    dom = env.assoc.argmax(axis=0)           # (M,) 每个用户归属的机
    n = env.n_uav
    inc = np.bincount(dom, weights=dc, minlength=n)        # Σ_{k∈K_i} Δc_k
    cnt = np.maximum(np.bincount(dom, minlength=n), 1)     # K_i
    s = float(cover.sum())
    f = (s * s) / (len(cover) * float((cover * cover).sum()) + 1e-12)
    p = PEN_LOST * (env.u > 5.0).astype(np.float64)
    return (f * inc / cnt - p).astype(np.float32), cover


def run_name(mode: str, tau: float, seed: int, scene: str,
             n_uav: int = 12, chan: str = "main") -> str:
    """{scene}[_n{N}]_{mode}_tau{tau}_s{seed}_{channel}_{shadow}。"""
    import os as _os
    c = _os.environ.get("PF_CHANNEL", "table")
    sig = _os.environ.get("PF_SHADOW_DB", "random").strip().lower()
    stag = "srand" if sig in ("random", "rand", "r") else f"s{sig}"
    parts = [scene]
    if n_uav != 12:
        parts.append(f"n{n_uav}")
    parts.append(mode)
    parts.append(f"tau{int(tau)}_s{seed}")
    if c != "main":
        parts.append(c)
    parts.append(stag)
    return "_".join(parts)


def train(mode: str, tau: float, seed: int,
          scene: str = "flow", t_comm: int = 1, anchor: float = 0.0,
          n_uav: int = 12, chan: str = "main",
          calib: dict | None = None, total_rollouts: int = TOTAL_ROLLOUTS,
          resume: bool = True, task: str | None = None,
          preset: str | None = None) -> Path:
    assert mode == "tmarl", "本训练器只服务 tmarl 臂"
    early_stop: bool = True
    task = task or run_name(mode, tau, seed, scene, n_uav=n_uav, chan=chan)
    tdir = CKPT_ROOT / task
    tdir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    # 锚定（2026-10-27 用户裁定：原始锚定方法，与 ppo 同口径——校准废止，
    # 不读 calib.json）
    _bs = baseline_scene(preset) or {}
    calib = {"p_ref_dbm": P_REF_DBM, "n0_dbm": N0_DBM,
             "demand_scale": _bs.get("demand_scale")
             or (calib["demand_scale"] if calib else DEMAND_SCALE_ANCHOR)}
    chan_map = {"main": (None, None), "uma35": (None, 3.5e9),
                "rma2": ("rma", None), "rma35": ("rma", 3.5e9)}
    scen, fc = chan_map[chan]
    env = UAVEnvV3(tau_u=tau, seed=seed, scene=scene,
                   n_uav=n_uav, t_comm=t_comm,
                   scenario=scen, fc_hz=fc,
                   use_physics=(baseline_scene(preset) is None),
                   override_spec=baseline_scene(preset), **calib)
    actor, a_t = Actor(), Actor()
    critic, c_t = Critic(), Critic()
    a_t.load_state_dict(actor.state_dict())
    c_t.load_state_dict(critic.state_dict())
    opt_a = torch.optim.Adam(actor.parameters(), lr=LR_A)
    opt_c = torch.optim.Adam(critic.parameters(), lr=LR_C)
    kmax = K_NB_FIX                                   # 邻居集固定宽（度数动态
    buf = Replay(BUF_ROLLOUTS * ROLLOUT, n_uav, kmax)  # 涨会炸定长缓冲）；容量按环境步
    ou = OUNoise(n_uav)
    start = 0

    ckpt = tdir / "latest.pt"
    if resume and ckpt.exists():
        sd = torch.load(ckpt, map_location="cpu")
        actor.load_state_dict(sd["policy"])
        a_t.load_state_dict(sd["policy"])
        critic.load_state_dict(sd["critic"])
        c_t.load_state_dict(sd["critic"])
        opt_a.load_state_dict(sd["opt_a"])
        opt_c.load_state_dict(sd["opt_c"])
        start = sd["rollout"] + 1
        print(f"[{task}] 断点恢复 → rollout {start}")
    elif start == 0:                                   # Chen 预训练（仅首启）
        pl = pretrain_lloyd(actor, env, opt_a, seed)
        a_t.load_state_dict(actor.state_dict())
        print(f"[{task}] lloyd 行为克隆预训练完成 bc_loss={pl:.4f}")
    if start >= total_rollouts:
        # 已完成：给调度器留下完成标记，避免空转重派
        print(f"[{task}] {total_rollouts}/{total_rollouts} "
              f"已完成（断点无剩余轮次）")
        return tdir

    wrap = TWrap(actor)
    log_f = open(tdir / "train_log.csv", "a", newline="", encoding="utf-8")
    w = csv.writer(log_f)
    if start == 0:
        w.writerow(["rollout", "env_steps", "mean_ret20", "c_loss", "q_mean",
                    "seconds", "var_u", "u_mean", "sum_t", "n_over"])
        log_f.flush()
    from train_probe import make_probe
    _drain, _reset_len, _win = make_probe(env)

    obs = norm_obs(env.reset())
    prev_cover = None
    rets: list[float] = []
    ret_curve: list[float] = []
    cl_v = qm_v = 0.0
    t0 = time.time()
    for ro in range(start, total_rollouts):
        for _ in range(ROLLOUT):
            idx, mask = nb_index(env.adj, K_NB_FIX, env.pos)
            polar = wrap.act_polar(obs, obs[idx], mask)
            polar = polar + ou()
            polar[:, 0] %= 2 * np.pi
            polar[:, 1] = np.clip(polar[:, 1], 0.0, DXY_MAX)
            d = decode_ddp(polar)
            nobs, _, done, _ = env.step(d)
            nobs = norm_obs(nobs)
            r, prev_cover = tmarl_reward(env, prev_cover)   # eq.8（论文奖励本体）
            idx2, mask2 = nb_index(env.adj, K_NB_FIX, env.pos)
            buf.push(obs, idx, mask, polar.astype(np.float32), r,
                     nobs, idx2, mask2, float(done))
            obs = nobs
            rets.append(r)
            if buf.n >= WARMUP:
                cl_v, qm_v = learn(actor, a_t, critic, c_t, opt_a, opt_c,
                                   buf, rng)
                soft_update(actor, a_t)
                soft_update(critic, c_t)
            if done:
                _drain()
                obs = norm_obs(env.reset())
                prev_cover = None
                _reset_len()
        _drain()
        if (ro + 1) % CKPT_EVERY == 0 or ro + 1 == total_rollouts:
            torch.save({"policy": actor.state_dict(),
                        "critic": critic.state_dict(),
                        "opt_a": opt_a.state_dict(), "opt_c": opt_c.state_dict(),
                        "rollout": ro, "mode": mode, "tau": tau},
                       ckpt)
            (tdir / "progress.json").write_text(json.dumps({
                "task": task, "rollout": ro + 1, "total": total_rollouts,
                "env_steps": (ro + 1) * ROLLOUT,
                "mean_ret_recent": float(np.mean(rets[-EPISODE_LEN:])),
                "c_loss": round(cl_v, 5), "q_mean": round(qm_v, 3),
                "elapsed_min": round((time.time() - t0) / 60, 1),
            }, ensure_ascii=False), encoding="utf-8")
        w.writerow([ro + 1, (ro + 1) * ROLLOUT,
                    round(float(np.mean(rets[-20 * ROLLOUT:])), 4),
                    round(cl_v, 5), round(qm_v, 3),
                    round(time.time() - t0, 1),
                    round(_win("var_u", ROLLOUT), 6),
                    round(_win("u_mean", ROLLOUT), 4),
                    round(_win("sum_t", ROLLOUT), 1),
                    round(_win("n_over", ROLLOUT), 2)])
        log_f.flush()
        if True:  # 逐步打印：每轮一行，不静默
            print(f"[{task}] {ro+1}/{total_rollouts} "
                  f"ret={np.mean(rets[-20*ROLLOUT:]):.3f} cl={cl_v:.4f} "
                  f"q={qm_v:.2f} {(time.time()-t0)/60:.1f}min", flush=True)
        ret_curve.append(float(np.mean(rets[-20 * ROLLOUT:])))
        # 早停：滑动判定窗 200 回合；只解除硬回合数下限，判据不变。
        if ((ro + 1) % 25 == 0
                and len(rets) >= 2 * EARLY_WIN * ROLLOUT):
            short = float(np.mean(rets[-EARLY_SHORT * ROLLOUT:]))
            long = float(np.mean(rets[-EARLY_WIN * ROLLOUT:]))
            diff = short - long
            band = EARLY_THR * max(abs(long), 1e-6)
            if abs(diff) <= band:
                print(f"[{task}] 早停@{ro+1}轮 近{EARLY_SHORT}轮={short:.4f} "
                      f"近{EARLY_WIN}轮={long:.4f} 差={diff:+.4f} "
                      f"|差|<{EARLY_THR:.1%} 无提升", flush=True)
                torch.save({"policy": actor.state_dict(),
                            "critic": critic.state_dict(),
                            "opt_a": opt_a.state_dict(),
                            "opt_c": opt_c.state_dict(),
                            "rollout": ro, "mode": mode, "tau": tau}, ckpt)
                (tdir / "progress.json").write_text(json.dumps({
                    "task": task, "rollout": ro + 1, "total": total_rollouts,
                    "env_steps": (ro + 1) * ROLLOUT, "early_stop": True,
                    "mean_ret_recent": float(np.mean(rets[-EPISODE_LEN:])),
                    "elapsed_min": round((time.time() - t0) / 60, 1),
                }, ensure_ascii=False), encoding="utf-8")
                break
    log_f.close()
    env.close()
    return tdir


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True)
    ap.add_argument("--tau", type=float, default=5.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--scene", default="flow")
    ap.add_argument("--tcomm", type=int, default=1)
    ap.add_argument("--anchor", type=float, default=0.0)
    ap.add_argument("--n-uav", type=int, default=12)
    ap.add_argument("--chan", default="main")
    ap.add_argument("--total-rollouts", type=int, default=TOTAL_ROLLOUTS)
    ap.add_argument("--task", default=None,
                    help="任务名覆盖（tag 行；worker/模块化接口）")
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--preset", default=None,
                    help="基线原生场景复刻名（config.BASELINE_SCENES）")
    args = ap.parse_args()
    train(args.mode, args.tau, args.seed, scene=args.scene,
          t_comm=args.tcomm, anchor=args.anchor,
          n_uav=args.n_uav, chan=args.chan, task=args.task,
          total_rollouts=args.total_rollouts, resume=not args.fresh,
          preset=args.preset)
