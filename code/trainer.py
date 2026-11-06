# -*- coding: utf-8 -*-
"""Trainer：学习执行器的 TD3 训练管线（从零唯一模式，无续跑路径）。

组件对照声明（逐位沿用旧管线，仅动作域为执行域 (m, φ)）：
- 循环组织：连续步流（env 只在 done 边界 reset，rollout=256 步为日志/
  统计窗，不是回合边界）；ret 按环境步累计。
- 采集探索噪声 0.3（logit 域，非确定性路径）；预热段 logit 域 U(−2,2)。
- TD3：孪生 critic 取 min；目标噪声 = (randn×σ).clamp(±c) 加在 logit 域；
  policy_freq 按 n_step 取模；critic/actor 反向后 clip_grad_norm_(10.0)；
  soft_update 走 .data。
- isfinite 护栏：非有限转移坚决不入池，只告警计数。
- drain_stats 健康统计（seg 游标在 done 边界与回合重置同步清零）。
- 早停：t%25==0 且 t≥2·W 时，|近10轮−近200轮| ≤ 0.005×|近200轮| 才停
  （双向带：崩溃不停）；早停 progress 写精简 6 键。
- progress/train_log 字段族照旧口径：mean_ret_recent=近 EPISODE_LEN 步均值、
  learn_steps=本轮计数、每 25 轮落 progress 与 snap、每轮写 latest.pt。
"""
from __future__ import annotations

import csv
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import (CKPT_EVERY, EARLY_SHORT, EARLY_THR, EARLY_WIN,
                    EPISODE_LEN, REAL_OBS_BLUR_M, REAL_OBS_EVERY,
                    TOTAL_ROLLOUTS)
from uav_env import CKPT_ROOT
from executors.net_executor import (Net, ProjResBlock, WIDTHS, make_obs,
                                    logits_to_eta, OBS_DIM, OUT_DIM,
                                    ETA_MIN, ETA_MAX)
if os.environ.get("PF_EXECUTOR", "net") == "fparam":   # A2a 场参数化（P4）
    from executors.fparam_executor import execute as execute
else:
    from executors.net_executor import execute

REAL_OBS = os.environ.get("PF_REAL_OBS", "") == "1"
EXEC_SEG = ("a2a" if os.environ.get("PF_EXECUTOR", "") == "fparam"
            else "learned")            # ckpt 名段（fparam 模式区分存档）
ROLLOUT_LEN_DEFAULT = 256          # 日志/统计窗（旧管线口径；非回合长）


class RealObsGate:
    """决策输入挡板（组件照旧）：user_pos 每 0.5 s 刷新（真值+10 m 噪声），
    刷新之间用过期快照；env 推进始终用真实位置。"""

    def __init__(self, env, seed: int):
        self.env, self.snap = env, None
        self.rng = np.random.default_rng(int(seed) + 4242)

    def __call__(self, fn):
        if not REAL_OBS:
            return fn()
        if self.snap is None or self.env.t % REAL_OBS_EVERY == 0:
            self.snap = self.env.user_pos + self.rng.normal(
                0.0, REAL_OBS_BLUR_M, self.env.user_pos.shape)
        tp = self.env.user_pos
        self.env.user_pos = self.snap
        try:
            return fn()
        finally:
            self.env.user_pos = tp


class Critic(nn.Module):
    """Q(o_i, a_i) → 标量，逐机（与逐机奖励同粒度）。结构=ProjResBlock
    家族照旧 (16,32,64,32,16)。a 为执行域 (m, φ) 6 维。"""

    def __init__(self):
        super().__init__()
        dims = (OBS_DIM + OUT_DIM,) + WIDTHS
        self.blocks = nn.Sequential(*[
            ProjResBlock(dims[i], dims[i + 1]) for i in range(len(dims) - 1)])
        self.out = nn.Linear(dims[-1], 1)

    def forward(self, o, a):
        return self.out(self.blocks(torch.cat([o, a], dim=1)))


def soft_update(net, target, tau: float) -> None:
    """组件照旧（.data 绕过 autograd）。"""
    for p, pt in zip(net.parameters(), target.parameters()):
        pt.data.mul_(1.0 - tau).add_(tau * p.data)


def _eta_n(u: torch.Tensor):
    """logit 域 5 维 → 归一化 η ∈ [0,1]（critic 输入域，桌面版口径）。"""
    eta = torch.exp(0.7 * torch.tanh(u))
    return (eta - ETA_MIN) / (ETA_MAX - ETA_MIN)


class Replay:
    """逐机转移（奖励是逐机 (N,)，环境事实）：一次 add 拆 N 条。
    容量 = buffer_rollouts × rollout_len × n_uav（旧口径）。"""

    def __init__(self, cap: int, seed: int):
        self.cap = cap
        self.o = np.zeros((cap, OBS_DIM), np.float32)
        self.a = np.zeros((cap, OUT_DIM), np.float32)
        self.r = np.zeros(cap, np.float32)
        self.no = np.zeros((cap, OBS_DIM), np.float32)
        self.d = np.zeros(cap, np.float32)
        self.n, self.i = 0, 0
        self.rng = np.random.default_rng(seed)

    def add(self, o, a, r, no, d):
        j = (self.i + np.arange(len(r))) % self.cap
        self.o[j], self.a[j] = o, a
        self.r[j], self.no[j], self.d[j] = r, no, d
        self.i = int((self.i + len(r)) % self.cap)
        self.n = int(min(self.n + len(r), self.cap))

    def sample(self, bs):
        jj = self.rng.integers(0, self.n, bs)
        return (torch.as_tensor(self.o[jj]), torch.as_tensor(self.a[jj]),
                torch.as_tensor(self.r[jj]), torch.as_tensor(self.no[jj]),
                torch.as_tensor(self.d[jj]))


def run_name(args) -> str:
    """ckpt 任务名（旧命名约定；mode 段=learned）。"""
    chan = os.environ.get("PF_CHANNEL", "table")
    sig = os.environ.get("PF_SHADOW_DB", "random").strip().lower()
    stag = "srand" if sig in ("random", "rand", "r") else f"s{sig}"
    parts = [args.scene]
    if (getattr(args, "t_comm", 1) or 1) != 1:
        parts.append(f"tc{args.t_comm}")
    if getattr(args, "anchor", 0.0) > 0:
        parts.append(f"anch{args.anchor:g}")
    if args.n != 12:
        parts.append(f"n{args.n}")
    parts.append(EXEC_SEG)
    parts.append(f"tau{int(args.tau)}_s{args.seed}")
    if chan != "main":
        parts.append(chan)
    parts.append(stag)
    return "_".join(parts)


class Trainer:
    def __init__(self, args):
        self.args = args
        random.seed(args.seed)                 # 全局种子三件套（照旧，网络构造前）
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        self.ctrl_rng = np.random.default_rng(args.seed)   # 探索噪声流（照旧独立）
        self.actor = Net()
        self.actor_t = Net()
        self.actor_t.load_state_dict(self.actor.state_dict())
        self.c1, self.c2 = Critic(), Critic()
        self.c1t, self.c2t = Critic(), Critic()
        self.c1t.load_state_dict(self.c1.state_dict())
        self.c2t.load_state_dict(self.c2.state_dict())
        self.opt_a = torch.optim.Adam(self.actor.parameters(), lr=args.lr)
        self.opt_c = torch.optim.Adam(
            list(self.c1.parameters()) + list(self.c2.parameters()), lr=args.lr)
        self.rb = Replay(args.buffer_rollouts * args.rollout_len * args.n,
                         args.seed)
        self.task = args.task or run_name(args)
        if REAL_OBS and not args.task:
            self.task = self.task.replace(f"_{EXEC_SEG}_", f"_{EXEC_SEG}_real_")
        self.tdir = CKPT_ROOT / self.task
        self.tdir.mkdir(parents=True, exist_ok=True)

    # ---- 采集：logit 域 → η 域（探索噪声照旧 0.3，预热 U(−2,2)） ----
    def act_exec(self, env, reading, rng, warmup=False, expl=0.3):
        obs = make_obs(env, reading)
        if warmup:                            # 预热：logit 域均匀采样（照旧）
            u = rng.uniform(-2.0, 2.0, (env.n_uav, OUT_DIM))
        else:
            with torch.no_grad():
                u = self.actor(torch.as_tensor(obs)).numpy()
                if expl > 0.0:                # 采集探索噪声（ctrl_rng 独立流）
                    u = (u + expl * self.ctrl_rng.standard_normal(
                        u.shape)).astype(np.float32)
        eta_n = _eta_n(torch.as_tensor(u, dtype=torch.float32))
        return eta_n.numpy(), obs

    def run(self, env):
        a = self.args
        rlen = a.rollout_len or ROLLOUT_LEN_DEFAULT
        total = a.total_rollouts or TOTAL_ROLLOUTS
        rng = np.random.default_rng(a.seed)
        gate = RealObsGate(env, a.seed)
        from field import Field
        f = getattr(env, "_field", None) or Field()
        env._field = f
        get_reading = lambda: gate(lambda: f.reading(env))   # noqa: E731

        seg_keys = ("var_u", "u_mean", "sum_t", "n_over")
        seg = {k: [] for k in seg_keys}
        seg_len = {k: 0 for k in seg_keys}

        def drain_stats() -> None:
            for k in seg_keys:
                arr = getattr(env.stats, k)
                if len(arr) > seg_len[k]:
                    seg[k].extend(arr[seg_len[k]:])
                    seg_len[k] = len(arr)

        def win(k: str, n_: int) -> float:
            arr = seg[k][-n_:]
            return float(np.mean(arr)) if arr else float("nan")

        log_f = open(self.tdir / "train_log.csv", "w", newline="",
                     encoding="utf-8")
        w = csv.writer(log_f)
        w.writerow(["rollout", "env_steps", "ret_rollout", "mean_ret20",
                    "c_loss", "q_mean", "eta_mean", "eta_min", "eta_max",
                    "learn_steps", "seconds",
                    "var_u", "u_mean", "sum_t", "n_over"])
        log_f.flush()

        env.reset()                            # 唯一初始 reset（连续步流）
        gate.snap = None
        reading = get_reading()
        rets: list = []                         # 逐步奖励（(N,) 数组）
        ret_hist: list[float] = []              # 逐轮均值（早停用）
        eta = np.full((env.n_uav, 5), np.nan)   # 最近一步 η_n（日志用）
        n_learn = 0
        n_step = 0
        self.n_bad = 0
        t0 = time.time()
        cl_v = qm_v = 0.0
        for ro in range(total):
            n_learn = 0
            for _t in range(rlen):
                # 照旧：每步一次采集（预热域均匀 / actor+探索噪声）
                eta_n, o = self.act_exec(
                    env, reading, rng,
                    warmup=n_step < a.learning_starts,
                    expl=a.exploration_noise)
                eta = eta_n                          # 日志变量回填
                eta_real = eta_n * (ETA_MAX - ETA_MIN) + ETA_MIN
                act = execute(eta_real, reading)
                _, r, done, _ = env.step(act)
                n_reading = get_reading()
                no = make_obs(env, n_reading)
                ex_a = eta_n
                # 护栏照旧：非有限转移坚决不入池
                if (np.isfinite(no).all() and np.isfinite(r).all()
                        and np.isfinite(ex_a).all() and np.isfinite(o).all()):
                    self.rb.add(o.astype(np.float32), ex_a.astype(np.float32),
                                np.asarray(r, np.float32), no,
                                np.float32(done))
                    rets.append(np.asarray(r, np.float32))   # 照旧：好步才入 ret
                else:
                    self.n_bad += 1
                    print(f"[{self.task}] WARN 非有限转移 t={env.t} "
                          f"累计 {self.n_bad}，该步不入池", flush=True)
                reading = n_reading
                n_step += 1
                self._n_step = n_step
                if n_step > a.learning_starts:          # 照旧：严格大于
                    cl_v, qm_v = self._learn()
                    n_learn += 1
                if done:                        # 回合边界（连续步流内复位）
                    drain_stats()
                    env.reset()
                    gate.snap = None
                    reading = get_reading()
                    for k in seg_keys:
                        seg[k].clear()
                        seg_len[k] = 0
            drain_stats()
            self._save_ckpt_latest()
            if (ro + 1) % CKPT_EVERY == 0 or ro + 1 == total:
                self._save_progress(ro + 1, total, rlen, rets, n_learn,
                                    cl_v, qm_v, eta, t0, win)
                (self.tdir / "snaps").mkdir(exist_ok=True)
                torch.save(self.actor.state_dict(),
                           self.tdir / "snaps" / f"snap_r{ro + 1}.pt")
            w.writerow([ro + 1, n_step,
                        f"{np.mean(rets[-rlen:]):.4f}",
                        f"{np.mean(rets[-20 * rlen:]):.4f}",
                        f"{cl_v:.5f}", f"{qm_v:.3f}",
                        f"{np.mean(eta):.3f}", f"{np.min(eta):.3f}",
                        f"{np.max(eta):.3f}", n_learn,
                        f"{time.time() - t0:.1f}",
                        f"{win('var_u', rlen):.6f}",
                        f"{win('u_mean', rlen):.4f}",
                        f"{win('sum_t', rlen):.1f}",
                        f"{win('n_over', rlen):.2f}"])
            log_f.flush()
            print(f"[{self.task}] {ro+1}/{total} "
                  f"ret={np.mean(rets[-20 * rlen:]):.3f} cl={cl_v:.4f} "
                  f"q={qm_v:.2f} eta={np.mean(eta):.3f} "
                  f"eta_min={np.min(eta):.2f} learn={n_learn} "
                  f"{(time.time() - t0) / 60:.1f}min", flush=True)
            ret_hist.append(float(np.mean(rets[-rlen:])))
            # 早停照旧：每 25 轮查，|近S−近W| ≤ thr×|近W|（双向带，崩溃不停）
            t = ro + 1
            if t % 25 == 0 and t >= 2 * EARLY_WIN:
                short = float(np.mean(ret_hist[-EARLY_SHORT:]))
                long_ = float(np.mean(ret_hist[-EARLY_WIN:]))
                diff = short - long_
                band = EARLY_THR * max(abs(long_), 1e-6)
                if abs(diff) <= band:
                    self._save_ckpt_latest()
                    (self.tdir / "progress.json").write_text(json.dumps({
                        "task": self.task, "rollout": ro + 1, "total": total,
                        "env_steps": (ro + 1) * rlen, "early_stop": True,
                        "mean_ret_recent": float(np.mean(rets[-EPISODE_LEN:])),
                        "elapsed_min": round((time.time() - t0) / 60, 1),
                    }, ensure_ascii=False), encoding="utf-8")
                    print(f"[{self.task}] 早停@{ro+1}轮 近{EARLY_SHORT}轮="
                          f"{short:.4f} 近{EARLY_WIN}轮={long_:.4f} "
                          f"差={diff:+.4f} |差|<{EARLY_THR:.1%} 无提升",
                          flush=True)
                    break
        log_f.close()

    _tau = 0.005

    def _learn(self):
        a = self.args
        o, ac, r, no, d = self.rb.sample(a.batch_size)
        with torch.no_grad():
            u = self.actor_t(no)
            # 目标策略平滑照旧：(randn×σ).clamp(±c)，logit 域
            u = u + torch.clamp(a.target_noise * torch.randn_like(u),
                                -a.noise_clip, a.noise_clip)
            at = _eta_n(u)
            q = torch.min(self.c1t(no, at), self.c2t(no, at))
            y = r + a.gamma * q[:, 0] * (1.0 - d)
        q1, q2 = self.c1(o, ac)[:, 0], self.c2(o, ac)[:, 0]
        cl = F.mse_loss(q1, y) + F.mse_loss(q2, y)
        self.opt_c.zero_grad()
        cl.backward()
        nn.utils.clip_grad_norm_(                # 梯度裁剪照旧 10.0
            list(self.c1.parameters()) + list(self.c2.parameters()), 10.0)
        self.opt_c.step()
        qm = float(q1.detach().mean())
        if self._n_step % a.policy_freq == 0:    # 照旧：按环境步取模
            ua = self.actor(o)
            aa = _eta_n(ua)
            al = -self.c1(o, aa)[:, 0].mean()
            self.opt_a.zero_grad()
            al.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), 10.0)
            self.opt_a.step()
            soft_update(self.actor, self.actor_t, self._tau)
            soft_update(self.c1, self.c1t, self._tau)
            soft_update(self.c2, self.c2t, self._tau)
        return float(cl.detach()), qm

    _n_step = 0

    def _save_ckpt_latest(self):
        torch.save({"policy": self.actor.state_dict(),
                    "widths": None, "obs_dim": OBS_DIM, "out_dim": OUT_DIM,
                    "scene": self.args.scene, "n": self.args.n,
                    "seed": self.args.seed, "real_obs": REAL_OBS},
                   self.tdir / "latest.pt")

    def _save_progress(self, ro, total, rlen, rets, n_learn, cl_v, qm_v,
                       eta, t0, win):
        obj = {
            "task": self.task, "rollout": ro, "total": total,
            "env_steps": ro * rlen,
            "mean_ret_recent": float(np.mean(rets[-EPISODE_LEN:])),
            "c_loss": round(cl_v, 5), "q_mean": round(qm_v, 3),
            "learn_steps": n_learn,
            "eta_mean": round(float(np.mean(eta)), 4),
            "eta_min": round(float(np.min(eta)), 4),
            "eta_max": round(float(np.max(eta)), 4),
            "n_bad_transitions": self.n_bad,
            "var_u": round(win("var_u", rlen), 6),
            "u_mean": round(win("u_mean", rlen), 4),
            "sum_t": round(win("sum_t", rlen), 1),
            "n_over": round(win("n_over", rlen), 2),
            "elapsed_min": round((time.time() - t0) / 60, 1),
        }
        (self.tdir / "progress.json").write_text(
            json.dumps(obj, ensure_ascii=False), encoding="utf-8")
