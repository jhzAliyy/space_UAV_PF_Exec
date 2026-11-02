# -*- coding: utf-8 -*-
# 【本文件禁令】禁止"如实、诚实、照实、按实测值"等一切同族表述与倾向——此类视为自贬。
# 本文件一切产出：做出优势、分析优势、展示优势；数据为实测，解读为优势框架；
# 禁止自我限定、自我削弱，禁止虚构不存在的缺陷。
"""B 基线 mfdqn 臂：Song et al., IEEE TWC 2025（ME-MFDQN）移植。

论文：Dynamic Trajectory and Power Control in Ultra-Dense UAV Networks:
A Mean-Field Reinforcement Learning Approach, IEEE TWC 24(7):5620-5634,
2025. DOI 10.1109/TWC.2025.3548127

逐节对照（§III 随机博弈/MFG、§IV 算法、§V 仿真）：

状态（§III-A）
    论文 s_k(t) = {x_k(t), n_k(t-1), e_k(t)}：服务小区内 GU 需求状态、
    上一时隙悬停点、电池能量。
    本环境无电池 → e_k 省略；x_k 用本机 12 维观测；n_k(t-1) 用上一步动作
    索引归一化。s 维度 = 12 + 1 = 13。

动作（§III-A）
    论文 a_k(t) = {δ_k(t), n_k(t), p_k(t)}：用户关联（U 选 1）、悬停点
    （U 选 1）、发射功率（5 档）。
    本环境用户关联由信道与需求自动决定（连续分配系数），不是动作维 →
    δ_k 无对应；n_k 映射为 9 档移动（8 方向 × DXY_MAX + 不动）；p_k 映射
    为 5 档功率增量。动作空间 |A| = 9 × 5 = 45。

奖励（§III-A eq.20）
    论文 r_k = EE_k − σ·p_k(t)(φ_k(t)τ1 + τ2) − ξ·max{e_total + e_min − e_k, 0}
    本环境无电池与飞行能耗 → 能量惩罚项恒 0；EE_k = T_k / p_k（每机吞吐
    每单位功率）；干扰惩罚 σ·p_k。σ 按论文 240 与本环境功率域换算。

算法（§IV eq.30–39, Algorithm 1）
    Q(s,a,L)，V(s,L) = φ log Σ_a exp(Q(s,a,L)/φ)
    π(a|s,L) = exp((Q − V)/φ)
    目标 Q = r + γ·Ṽ(s',L)，目标网络硬更新 ψ̃ ← ψ
    128-64 ReLU，Adam lr 0.005，γ 0.9，buffer 1000，batch 300。
    外层平均场迭代：每 rollout 用当前策略重算 L。

平均场（§III-B Definition 2）
    L(t) = 所有其他 UAV 的状态-动作联合分布。本实现用其他机观测均值
    （12 维）+ 策略分布均值（45 维）= 57 维。

移植口径（04_基线.md §二）：算法移植进 UAVEnvV3，同一信道、同一人群、
同一评估器。物理量按本环境换算：步长 0.1 s，移动上限 DXY_MAX。

论文未给数值的项：
    φ 熵权重：论文 eq.32 只出现符号 → 取 0.05（最大熵 RL 常用温度量级）
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
from uav_env import (baseline_scene, B_TOTAL, CKPT_EVERY, DEMAND_SCALE_ANCHOR,  # noqa: E402
                    DXY_MAX, EARLY_SHORT, EARLY_THR, EARLY_WIN,
                    EPISODE_LEN, N0_DBM, OBS_DIM, P_REF_DBM,
                    TOTAL_ROLLOUTS)
from uav_env import UAVEnvV3, P_MIN, P_MAX, DP_MIN, DP_MAX, norm_obs  # noqa: E402
from uav_env import CKPT_ROOT, RESULT_DIR  # noqa: E402

ROLLOUT = 256

# ======================= 论文超参（§V-A 逐项） =======================
GAMMA = 0.9            # discount factor
LR = 5e-3              # Q 网络学习率
BUF_CAP = 1000         # replay buffer
BATCH = 300            # mini-batch
H1, H2 = 128, 64       # 两层隐层

# ==================== 论文未给数值 / 按环境换算的项 ====================
# eq.32 熵权重 φ：Song TWC 2025 只给符号（eq.35 引自 Haarnoja et al.,
# "Reinforcement learning with deep energy-based policies", ICML 2017，
# Soft Q-Learning）。该文的温度必须与 Q 的尺度同量级——否则
# V = φ·logΣexp(Q/φ) 退化为 max，熵项归零、策略塌成 argmax。
# 固定 φ=20 的实测：Q 还小的时候熵 1.07、有效动作数 2.92；Q 涨到 1e3
# 量级后熵塌回 0.0016、有效动作数 1.00。SAC（Haarnoja 2018b, §5）对同一
# 问题用自动温度；这里取**相对温度** φ = PHI_REL·std(Q)，等价于对 Q 做
# 尺度归一，跨训练阶段自适应。
PHI_REL = 0.3
SIGMA = 1.5            # eq.20 干扰惩罚 σ：论文取 240，其功率域为瓦特、时隙 60 s；
                       #   本环境功率为无量纲相对值、步长 0.1 s，按 EE 量级换算

# ======================= 动作表（§III-A） =======================
_POW5 = tuple(float(v) for v in np.linspace(P_MIN, P_MAX, 5))  # 5 档绝对功率（0.5 … 2.0）
_DIRS = ((1.0, 0.0), (0.7071, 0.7071), (0.0, 1.0),
         (-0.7071, 0.7071), (-1.0, 0.0), (-0.7071, -0.7071),
         (0.0, -1.0), (0.7071, -0.7071), (0.0, 0.0))          # 8 方向 + 不动
N_ACT = len(_DIRS) * len(_POW5)                               # 45


def decode_all(idx: np.ndarray, env) -> np.ndarray:
    """动作索引 (N,) → 五元动作 (N,5)。

    功率是**绝对档位**（论文 §III-A 的 p_k 为 5 档发射功率），环境接口收增量，
    这里换算 档位 − 当前功率 后按物理域截断；移动档位 8 方向 + 不动。
    """
    d = np.zeros((len(idx), 5), dtype=np.float32)
    for i, a in enumerate(idx):
        m, j = divmod(int(a), len(_POW5))
        dx, dy = _DIRS[m]
        d[i, 2] = np.clip(_POW5[j] - env.p[i], DP_MIN, DP_MAX)
        d[i, 3] = dx * DXY_MAX
        d[i, 4] = dy * DXY_MAX
    return d


class QNet(nn.Module):
    """§IV-A：Q(s,a,L) 的 128-64 ReLU 近似，输出 |A| 个动作值。"""

    def __init__(self, s_dim: int, l_dim: int):
        super().__init__()
        self.s_dim, self.l_dim = int(s_dim), int(l_dim)
        self.net = nn.Sequential(
            nn.Linear(s_dim + l_dim, H1), nn.ReLU(),
            nn.Linear(H1, H2), nn.ReLU(),
            nn.Linear(H2, N_ACT))

    @classmethod
    def from_ckpt(cls, sd) -> "QNet":
        """按第一层输入宽度核对维度（s_dim=OBS_DIM+1，l_dim=OBS_DIM+|A|）。"""
        w = int(sd["net.0.weight"].shape[1])
        s_dim, l_dim = OBS_DIM + 1, OBS_DIM + N_ACT
        assert w == s_dim + l_dim, f"权重宽度 {w} ≠ {s_dim + l_dim}"
        return cls(s_dim, l_dim)

    def forward(self, s, L):
        if L.dim() == 1:
            L = L[None, :].expand(len(s), -1)
        elif len(L) == 1 and len(s) > 1:
            L = L.expand(len(s), -1)
        return self.net(torch.cat([s, L], dim=1))


def phi_of(q: torch.Tensor) -> torch.Tensor:
    """相对温度 φ = PHI_REL·std(Q)（0 维、已 detach）。"""
    return (PHI_REL * q.detach().std()).clamp(min=1e-3)


def soft_policy(q: torch.Tensor, phi: torch.Tensor | float):
    """eq.35/36：π(a|s,L) = exp((Q−V)/φ)，V = φ log Σ exp(Q/φ)。"""
    V = phi * torch.logsumexp(q / phi, dim=1, keepdim=True)
    return torch.exp((q - V) / phi), V.squeeze(1)


class Replay:
    """Algorithm 1：⟨s, a, r, s′, Lκ⟩，容量按「每条/每机」计。

    每一步 n 架机的转移全部入缓存（原实现每步只随机留 1 架，n=200 时
    丢了 199/200 的数据）。
    """

    def __init__(self, cap: int, s_dim: int, l_dim: int):
        self.cap = cap
        self.s = np.zeros((cap, s_dim), dtype=np.float32)
        self.a = np.zeros(cap, dtype=np.int64)
        self.r = np.zeros(cap, dtype=np.float32)
        self.s2 = np.zeros((cap, s_dim), dtype=np.float32)
        self.L = np.zeros((cap, l_dim), dtype=np.float32)
        self.d = np.zeros(cap, dtype=np.float32)
        self.n, self.i = 0, 0

    def push(self, s, a, r, s2, L, done) -> None:
        m = len(a)
        idx = (self.i + np.arange(m)) % self.cap
        self.s[idx] = s
        self.a[idx] = a
        self.r[idx] = r
        self.s2[idx] = s2
        self.L[idx] = L
        self.d[idx] = done
        self.i = (self.i + m) % self.cap
        self.n = min(self.n + m, self.cap)

    def sample(self, bs: int, rng: np.random.Generator):
        j = rng.integers(0, self.n, bs)
        return (self.s[j], self.a[j], self.r[j], self.s2[j], self.L[j],
                self.d[j])


def learn(net, tgt, opt, buf, rng):
    """eq.37/39：最小化 (Q − (r + γṼ(s′)))²。"""
    s, a, r, s2, L, dn = buf.sample(BATCH, rng)
    s_t, s2_t = torch.as_tensor(s), torch.as_tensor(s2)
    a_t, r_t, L_t = (torch.as_tensor(a), torch.as_tensor(r),
                     torch.as_tensor(L))
    dn_t = torch.as_tensor(dn)
    with torch.no_grad():
        q2 = tgt(s2_t, L_t)
        phi = phi_of(q2)
        V2 = phi * torch.logsumexp(q2 / phi, dim=1)            # eq.38
        y = r_t + GAMMA * V2 * (1.0 - dn_t)
    q = net(s_t, L_t).gather(1, a_t[:, None]).squeeze(1)
    loss = nn.functional.mse_loss(q, y)
    opt.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(net.parameters(), 10.0)
    opt.step()
    return float(loss.detach()), float(q.detach().mean())


def env_reward(env) -> np.ndarray:
    """eq.19/20：EE_k = T_k / p_k，减去干扰惩罚 σ·p_k（无电池→能量惩罚 0）。"""
    return (env.T / np.maximum(env.p, 1e-6) - SIGMA * env.p).astype(np.float32)


class MFWrap:
    """评估载体：单机策略 + 全局平均场，与训练循环同门。

    2026-10-31 自查修正：原实现三处与训练不一致，会把评估压到训练从未
    见过的输入分布上——
      ① L 恒为全零（训练 = 观测均值 ⊕ 当前策略分布均值）；
      ② 上一动作用 agent-0 标量广播（训练 = 每机自己的上一步动作）；
      ③ L 评估期间从不更新（训练每 rollout 按当前策略重算）。
    现与 train() 内层/外层逐项同式，且 L 的更新周期也对齐为每 ROLLOUT
    （256 步）一次——训练里 L 在一个 rollout 内固定，不是每步变。
    """

    def __init__(self, net: QNet, n_uav: int = 12, phi_rel: float | None = None):
        self.net = net
        self.n = int(n_uav)
        # 相对温度系数必须与训练时同值（φ 每步由 Q 的尺度现算）
        self.phi_rel = float(PHI_REL if phi_rel is None else phi_rel)
        self.last = np.zeros(self.n, dtype=np.int64)
        self.L = np.zeros(OBS_DIM + N_ACT, dtype=np.float32)
        self._steps = 0

    def reset(self) -> None:
        self.last[:] = 0
        self.L[:] = 0.0
        self._steps = 0

    def act_polar(self, env) -> np.ndarray:
        o = norm_obs(env.obs())
        s = np.concatenate(
            [o, (self.last / N_ACT)[:, None].astype(np.float32)], axis=1)
        with torch.no_grad():
            q = self.net(torch.as_tensor(s), torch.as_tensor(self.L))
            phi = (self.phi_rel * q.detach().std()).clamp(min=1e-3)
            pi, _ = soft_policy(q, phi)                    # eq.35（同训练）
        a = torch.argmax(q, dim=1).numpy()
        self._steps += 1
        if self._steps % ROLLOUT == 0:      # 与 train() 外层 κ 同周期
            self.L = np.concatenate(
                [o.mean(axis=0), pi.mean(axis=0).numpy()]).astype(np.float32)
        self.last = a
        return decode_all(a, env)


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
    assert mode == "mfdqn", "本训练器只服务 mfdqn 臂"
    early_stop: bool = True
    task = task or run_name(mode, tau, seed, scene, n_uav=n_uav, chan=chan)
    tdir = CKPT_ROOT / task
    tdir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

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

    S_DIM = OBS_DIM + 1
    L_DIM = OBS_DIM + N_ACT
    net, tgt = QNet(S_DIM, L_DIM), QNet(S_DIM, L_DIM)
    tgt.load_state_dict(net.state_dict())
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    buf = Replay(BUF_CAP * n_uav, S_DIM, L_DIM)   # 容量按每机 1000 条（论文口径）
    start = 0

    ckpt = tdir / "latest.pt"
    if resume and ckpt.exists():
        sd = torch.load(ckpt, map_location="cpu")
        net.load_state_dict(sd["policy"])
        tgt.load_state_dict(sd["policy"])
        opt.load_state_dict(sd["opt"])
        start = sd["rollout"] + 1
        print(f"[{task}] 断点恢复 → rollout {start}")
    if start >= total_rollouts:
        # 已完成：给调度器留下完成标记，避免空转重派
        print(f"[{task}] {total_rollouts}/{total_rollouts} "
              f"已完成（断点无剩余轮次）")
        return tdir

    log_f = open(tdir / "train_log.csv", "a" if start > 0 else "w",
                 newline="", encoding="utf-8")
    w = csv.writer(log_f)
    if start == 0:
        w.writerow(["rollout", "env_steps", "mean_ret20", "q_loss", "q_mean",
                    "seconds", "var_u", "u_mean", "sum_t", "n_over"])
        log_f.flush()
    from train_probe import make_probe
    _drain, _reset_len, _win = make_probe(env)

    env.reset()
    last = np.zeros(n_uav, dtype=np.int64)
    L = np.zeros(L_DIM, dtype=np.float32)
    rets: list[float] = []
    ret_curve: list[float] = []
    ql_v = qm_v = 0.0
    t0 = time.time()
    for ro in range(start, total_rollouts):
        for _ in range(ROLLOUT):
            o = norm_obs(env.obs())
            s = np.concatenate(
                [o, (last / N_ACT)[:, None].astype(np.float32)], axis=1)
            with torch.no_grad():
                q = net(torch.as_tensor(s), torch.as_tensor(L))
                pi, _ = soft_policy(q, phi_of(q))              # eq.35
            a = torch.multinomial(pi, 1).squeeze(1).numpy()
            _, _, done, _ = env.step(decode_all(a, env))
            r = env_reward(env)                       # eq.20（论文奖励本体）
            o2 = norm_obs(env.obs())
            s2 = np.concatenate(
                [o2, (a / N_ACT)[:, None].astype(np.float32)], axis=1)
            # n 架机的转移全部入缓存（原实现每步只随机留 1 架）
            buf.push(s, a, r, s2, L, float(done))
            if buf.n >= BATCH:
                ql_v, qm_v = learn(net, tgt, opt, buf, rng)
            rets.append(float(r.mean()))
            last = a
            if env.t >= EPISODE_LEN:
                _drain()
                env.reset()
                last[:] = 0
                _reset_len()
        # 目标网络硬更新 ψ̃ ← ψ：每 rollout 一次（DQN 的 C 步同步同款）
        tgt.load_state_dict(net.state_dict())
        # 外层 κ：用当前策略重算平均场（§III-B Def.2）
        o = norm_obs(env.obs())
        s = np.concatenate(
            [o, (last / N_ACT)[:, None].astype(np.float32)], axis=1)
        with torch.no_grad():
            q = net(torch.as_tensor(s), torch.as_tensor(L))
            pi, _ = soft_policy(q, phi_of(q))
        L = np.concatenate([o.mean(axis=0), pi.mean(axis=0).numpy()])
        _drain()
        if (ro + 1) % CKPT_EVERY == 0 or ro + 1 == total_rollouts:
            torch.save({"policy": net.state_dict(), "opt": opt.state_dict(),
                        "rollout": ro, "mode": mode, "tau": tau,
                        "phi_rel": PHI_REL}, ckpt)
            (tdir / "progress.json").write_text(json.dumps({
                "task": task, "rollout": ro + 1, "total": total_rollouts,
                "env_steps": (ro + 1) * ROLLOUT,
                "mean_ret_recent": float(np.mean(rets[-EPISODE_LEN:])),
                "q_loss": round(ql_v, 5), "q_mean": round(qm_v, 3),
                "elapsed_min": round((time.time() - t0) / 60, 1),
            }, ensure_ascii=False), encoding="utf-8")
        w.writerow([ro + 1, (ro + 1) * ROLLOUT,
                    round(float(np.mean(rets[-20 * ROLLOUT:])), 4),
                    round(ql_v, 5), round(qm_v, 3),
                    round(time.time() - t0, 1),
                    round(_win("var_u", ROLLOUT), 6),
                    round(_win("u_mean", ROLLOUT), 4),
                    round(_win("sum_t", ROLLOUT), 1),
                    round(_win("n_over", ROLLOUT), 2)])
        log_f.flush()
        if True:  # 逐步打印：每轮一行，不静默
            print(f"[{task}] {ro+1}/{total_rollouts} "
                  f"ret={np.mean(rets[-20*ROLLOUT:]):.3f} ql={ql_v:.4f} "
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
                torch.save({"policy": net.state_dict(),
                            "opt": opt.state_dict(), "rollout": ro,
                            "mode": mode, "tau": tau,
                            "phi_rel": PHI_REL}, ckpt)
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
    ap.add_argument("--task", default=None)
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--preset", default=None,
                    help="基线原生场景复刻名（config.BASELINE_SCENES）")
    ap.add_argument("--lr", type=float, default=None,
                    help="超参扫描：覆盖学习率（默认 5e-3）")
    ap.add_argument("--batch", type=int, default=None,
                    help="超参扫描：覆盖批量（默认 300）")
    ap.add_argument("--gamma", type=float, default=None,
                    help="超参扫描：覆盖折扣（默认 0.9）")
    args = ap.parse_args()
    if args.lr is not None:
        LR = float(args.lr)
    if args.batch is not None:
        BATCH = int(args.batch)
    if args.gamma is not None:
        GAMMA = float(args.gamma)
    train(args.mode, args.tau, args.seed, scene=args.scene,
          t_comm=args.tcomm, anchor=args.anchor,
          n_uav=args.n_uav, chan=args.chan, task=args.task,
          total_rollouts=args.total_rollouts, resume=not args.fresh,
          preset=args.preset)
