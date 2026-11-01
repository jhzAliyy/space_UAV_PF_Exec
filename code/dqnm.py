# -*- coding: utf-8 -*-
# 【本文件禁令】禁止"如实、诚实、照实、按实测值"等一切同族表述与倾向——此类视为自贬。
# 本文件一切产出：做出优势、分析优势、展示优势；数据为实测，解读为优势框架；
# 禁止自我限定、自我削弱，禁止虚构不存在的缺陷。
"""B 批 dqnm 臂：Rizvi TMLCN 2025 掩码 SDQN 移植（15 号文档 §七，P1）。

移植口径（算法进我们的环境，信道/评估/判据全同口径）：
- 动作格 81 = (8 向移动+悬停) ×3 功率档 ×3 带宽档；Δπ≡0（Rizvi 无波束维）
- 无效动作掩码：动作使下一态越物理界 → 无效（uav_env.invalid_mask81 单点）
- ε-greedy 探索在掩码内均匀采样（Rizvi 同款）；Bellman target 同掩码
- 参数共享中心式 Q：单机 obs(11) → Q(81)（旧 PPO 网络同构容量 64×64 tanh）
- 超参对齐 Rizvi：lr=1e-3、γ=0.99、batch 64、Huber、target 慢同步

工程对齐（ppo 直承）：
- run_name/ckpt 目录/train_log.csv/progress.json 同式 → eval 无缝评估
- 断点 resume（rollout 续跑；replay 不落盘——重启即重采，ε 按绝对进度重算）
- 早停同 PPO 判据（平缓即停，2026-10-25 裁定）
- 动作选择与 Controller.dqnm 分支同走 select81（训练=评估同路径）
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
from uav_env import (OBS_DIM, EPISODE_LEN, P_REF_DBM, N0_DBM,        # noqa: E402
                    DEMAND_SCALE_ANCHOR, CKPT_EVERY, TOTAL_ROLLOUTS,
                    EARLY_WIN, EARLY_THR, baseline_scene)
from uav_env import UAVEnvV3                                    # noqa: E402
from uav_env import (N_ACT81, invalid_mask81, select81, norm_obs)  # noqa: E402
from uav_env import CKPT_ROOT, RESULT_DIR  # noqa: E402

ROLLOUT = 256
LR = 1e-3                       # Rizvi SDQN 同值
GAMMA = 0.99
BATCH = 64
WARMUP = 4096                   # 转移数：buffer 预热（首梯度前）
TARGET_SYNC = 1000              # 环境步：target 网络同步
EPS_MIN, EPS_FRAC = 0.01, 0.4   # ε 线性衰减至 0.01，衰减段=前 40% 总步
GRAD_CLIP = 10.0
BUF_ROLLOUTS = 30               # replay 深度=30 rollout 等效（随 n_uav 缩放）


class QNet(nn.Module):
    """旧 PPO 网络同构容量（64×64 tanh → 81）——信息集与容量双公平。"""

    def __init__(self, obs_dim: int = OBS_DIM, n_act: int = N_ACT81):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(obs_dim, 64), nn.Tanh(),
                                 nn.Linear(64, 64), nn.Tanh(),
                                 nn.Linear(64, n_act))

    def forward(self, x):
        return self.net(x)


class DQNWrap:
    """eval 加载载体（v2 验证件范式：训练器导出，禁止内联重写）。"""

    def __init__(self, qnet: QNet):
        self.qnet = qnet

    def q(self, obs: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            return self.qnet(torch.as_tensor(obs, dtype=torch.float32)).numpy()


class Replay:
    """环形缓冲（转移=单机单步）；含下一态掩码（Bellman target 同掩码忠实移植）。"""

    def __init__(self, cap: int):
        self.cap = cap
        self.obs = np.zeros((cap, OBS_DIM), dtype=np.float32)
        self.act = np.zeros(cap, dtype=np.int64)
        self.rew = np.zeros(cap, dtype=np.float32)
        self.nobs = np.zeros((cap, OBS_DIM), dtype=np.float32)
        self.done = np.zeros(cap, dtype=np.float32)
        self.nmask = np.zeros((cap, N_ACT81), dtype=bool)
        self.n, self.i = 0, 0

    def push(self, obs, a_idx, r, nobs, done, nmask) -> None:
        m = len(a_idx)
        idx = (self.i + np.arange(m)) % self.cap
        self.obs[idx] = obs
        self.act[idx] = a_idx
        self.rew[idx] = r
        self.nobs[idx] = nobs
        self.done[idx] = done
        self.nmask[idx] = nmask
        self.i = (self.i + m) % self.cap
        self.n = min(self.n + m, self.cap)

    def sample(self, bs: int, rng: np.random.Generator):
        j = rng.integers(0, self.n, bs)
        return (self.obs[j], self.act[j], self.rew[j],
                self.nobs[j], self.done[j], self.nmask[j])


def learn(qnet, target, opt, buf, rng) -> tuple[float, float]:
    o, a, r, no, dn, nm = buf.sample(BATCH, rng)
    o_t = torch.as_tensor(o)
    no_t = torch.as_tensor(no)
    a_t = torch.as_tensor(a)
    r_t = torch.as_tensor(r)
    dn_t = torch.as_tensor(dn)
    q = qnet(o_t).gather(1, a_t[:, None]).squeeze(1)
    with torch.no_grad():
        q_next_all = target(no_t)
        q_next_all[~torch.as_tensor(nm)] = -1e9          # target 侧同掩码
        y = r_t + GAMMA * q_next_all.max(axis=1).values * (1.0 - dn_t)
    loss = nn.functional.smooth_l1_loss(q, y)            # Huber（SDQN 惯例）
    opt.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(qnet.parameters(), GRAD_CLIP)
    opt.step()
    return float(loss.detach()), float(q.detach().mean())


def dqnm_reward(env) -> float:
    """Rizvi TMLCN 2025 eq.26：R_u = R(t) / 2^λ。

    原文 §V「8) REWARD」：奖励基于**全网总数据率**而不是单机速率；
    λ 为 QoS 惩罚系数，默认为 0，UAV 选择了违反 QoS 的动作时递增
    （"λ increases if a UAV chooses an action that denies QoS"）。
    本环境无显式 QoS 门限动作，取"该机负载超过可用容量"（u>1）为
    否定 QoS 的等价判据，λ 记违反机数。
    """
    R = float(env.T.sum())
    lam = float((env.u > 1.0).sum())
    return R / (2.0 ** lam)


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
    assert mode == "dqnm", "本训练器只服务 dqnm 臂"
    early_stop: bool = True                    # 同 PPO 裁定：平缓即停
    task = task or run_name(mode, tau, seed, scene, n_uav=n_uav, chan=chan)
    tdir = CKPT_ROOT / task
    tdir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)          # ε 探索专用流（与 env 解耦）

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
    qnet, target = QNet(), QNet()
    target.load_state_dict(qnet.state_dict())  # 同步起点
    opt = torch.optim.Adam(qnet.parameters(), lr=LR)
    buf = Replay(BUF_ROLLOUTS * ROLLOUT * n_uav)
    start = 0

    ckpt = tdir / "latest.pt"
    if resume and ckpt.exists():
        sd = torch.load(ckpt, map_location="cpu")
        qnet.load_state_dict(sd["policy"])
        target.load_state_dict(sd["policy"])   # target 从 policy 重建
        opt.load_state_dict(sd["opt"])
        start = sd["rollout"] + 1
        print(f"[{task}] 断点恢复 → rollout {start}")
    if start >= total_rollouts:
        # 已完成：给调度器留下完成标记，避免空转重派
        print(f"[{task}] {total_rollouts}/{total_rollouts} "
              f"已完成（断点无剩余轮次）")
        return tdir

    dqnw = DQNWrap(qnet)
    log_f = open(tdir / "train_log.csv", "a", newline="", encoding="utf-8")
    w = csv.writer(log_f)
    if start == 0:
        w.writerow(["rollout", "env_steps", "mean_ret20", "eps", "loss",
                    "q_mean", "seconds", "var_u", "u_mean", "sum_t", "n_over"])
        log_f.flush()
    from train_probe import make_probe
    _drain, _reset_len, _win = make_probe(env)

    total_steps = total_rollouts * ROLLOUT
    eps_of = lambda gs: max(EPS_MIN, 1.0 - gs / (EPS_FRAC * total_steps))
    obs = norm_obs(env.reset())
    rets: list[float] = []
    ret_curve: list[float] = []
    gstep = start * ROLLOUT                     # 全局环境步（ε 调度绝对进度）
    loss_v = qm_v = 0.0
    t0 = time.time()
    for ro in range(start, total_rollouts):
        for _ in range(ROLLOUT):
            # 与 Controller.dqnm 分支同路径：DQNWrap.q + select81（单点定义）。
            # 入网观测必须是归一化后的 obs——replay 存的就是 norm_obs，评估端
            # evaluate.py 也走 norm_obs；此前进网的是 env.obs() 原始量，
            # 贪心分支见到的是训练从未出现过的分布。
            q = dqnw.q(obs)
            a_idx, d = select81(q, env, eps_of(gstep), rng)
            nobs, _, done, _ = env.step(d)
            nobs = norm_obs(nobs)
            r = dqnm_reward(env)                 # eq.26（论文奖励本体）
            buf.push(obs, a_idx, r, nobs, float(done),
                     invalid_mask81(env))        # 下一态掩码（Bellman 同掩码）
            obs = nobs
            rets.append(r)
            gstep += 1
            if buf.n >= WARMUP:
                loss_v, qm_v = learn(qnet, target, opt, buf, rng)
            if gstep % TARGET_SYNC == 0:
                target.load_state_dict(qnet.state_dict())
            if done:
                _drain()                         # reset 前结算，之后 EnvStats 重建
                obs = norm_obs(env.reset())
                _reset_len()
        _drain()
        if (ro + 1) % CKPT_EVERY == 0 or ro + 1 == total_rollouts:
            torch.save({"policy": qnet.state_dict(), "opt": opt.state_dict(),
                        "rollout": ro, "mode": mode, "tau": tau},
                       ckpt)                    # eval 只读 policy
            (tdir / "progress.json").write_text(json.dumps({
                "task": task, "rollout": ro + 1, "total": total_rollouts,
                "env_steps": gstep,
                "mean_ret_recent": float(np.mean(rets[-EPISODE_LEN:])),
                "eps": round(eps_of(gstep), 4), "loss": round(loss_v, 5),
                "q_mean": round(qm_v, 3),
                "elapsed_min": round((time.time() - t0) / 60, 1),
            }, ensure_ascii=False), encoding="utf-8")
        w.writerow([ro + 1, gstep,
                    round(float(np.mean(rets[-20 * ROLLOUT:])), 4),
                    round(eps_of(gstep), 4), round(loss_v, 5),
                    round(qm_v, 3), round(time.time() - t0, 1),
                    round(_win("var_u", ROLLOUT), 6),
                    round(_win("u_mean", ROLLOUT), 4),
                    round(_win("sum_t", ROLLOUT), 1),
                    round(_win("n_over", ROLLOUT), 2)])
        log_f.flush()
        if True:  # 逐步打印：每轮一行，不静默
            print(f"[{task}] {ro+1}/{total_rollouts} "
                  f"ret={np.mean(rets[-20*ROLLOUT:]):.3f} eps={eps_of(gstep):.3f} "
                  f"loss={loss_v:.4f} q={qm_v:.2f} "
                  f"{(time.time()-t0)/60:.1f}min", flush=True)
        # 早停同 PPO 判据（结构直承）：近20轮均值曲线连续两窗改善 <0.5%
        ret_curve.append(float(np.mean(rets[-20 * ROLLOUT:])))
        # 早停：滑动判定窗 200 回合；只解除硬回合数下限，判据不变。
        if ((ro + 1) % 25 == 0 and len(ret_curve) >= 2 * EARLY_WIN):
            m2 = float(np.mean(ret_curve[-EARLY_WIN:]))
            m1 = float(np.mean(ret_curve[-2 * EARLY_WIN:-EARLY_WIN]))
            best_w = max(float(np.mean(ret_curve[i:i + EARLY_WIN]))
                         for i in range(0, len(ret_curve) - EARLY_WIN + 1))
            if (abs(m2 - m1) <= EARLY_THR * max(abs(m1), 1e-6)
                    and m2 >= best_w - EARLY_THR * max(abs(best_w), 1e-6)):
                print(f"[{task}] 早停@{ro+1}轮 窗口改善|{m2:.4f}-{m1:.4f}|"
                      f"<{EARLY_THR:.1%} 且达历史最佳窗口 {best_w:.4f}",
                      flush=True)
                torch.save({"policy": qnet.state_dict(),
                            "opt": opt.state_dict(),
                            "rollout": ro, "mode": mode, "tau": tau}, ckpt)
                (tdir / "progress.json").write_text(json.dumps({
                    "task": task, "rollout": ro + 1, "total": total_rollouts,
                    "env_steps": gstep, "early_stop": True,
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
