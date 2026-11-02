# -*- coding: utf-8 -*-
# 【本文件禁令】禁止"如实、诚实、照实、按实测值"等一切同族表述与倾向——此类视为自贬。
# 本文件一切产出：做出优势、分析优势、展示优势；数据为实测，解读为优势框架；
# 禁止自我限定、自我削弱，禁止虚构不存在的缺陷。
"""B 基线 ddpga 臂：Cheng et al., IEEE TIM 2024, 5503613（LTCC-UDUA）忠实复现。

逐节对照论文（§III 系统模型、§IV 算法、Table I 参数）：

状态（§IV-A State，3M 维）
    s_t = [P_1..P_M, R_1..R_M]
    P_m = (x_m, y_m) / L，L 为任务区边长；
    R_m = 第 m 架关联 GU 的累计吞吐 / 全部关联 GU 的总累计。

动作（§IV-A Action，2M 维连续）
    a_t = [ψ_1..ψ_M, D_1..D_M]，ψ∈[0,2π] 飞行方向，D∈[0,Dmax] 飞行距离。
    网络输出 tanh ∈[-1,1]，线性映射到上述区间后转 (Δx,Δy)=(D cosψ, D sinψ)。

奖励（§IV-A Reward，eq.10/12/14）
    r_m,t = ξ_th · f_ϖ · R_sum
    f_ϖ = (Σ_n R_n)² / (N · Σ_n R_n²)           Jain 公平指数（eq.12）
    R_sum = Σ_n R_n                              总吞吐（eq.10）
    另加两项惩罚：UAV 间距越限 −γ（eq.13h）、越界 −φ。

网络（§IV-B, Fig.3/4）
    Actor  3M → 128 ReLU → 64 ReLU → 2M → tanh
    Critic (3M+2M) → 128 ReLU → 64 ReLU → 1
    注：§IV-B 原文称 actor 输出"经 softmax 层，每个输出表示某动作的概率"再 tanh
    限幅——softmax 的概率语义只适用于离散动作，与本文连续 (ψ,D) 动作不相容。
    按该段末句"tanh 激活用于把动作限制在范围内"实现，softmax 不落地。

DDPG（Algorithm 1, eq.15–20）
    γ=0.9、τ=0.01、replay=5000、batch=1024、Adam lr=1e-3、目标网络软更新。
    CTDE：critic 吃全局状态+联合动作；actor 吃全局状态（论文状态定义即全局）。

移植口径（04_基线.md §二）：算法移植进 UAVEnvV3，同一信道、同一人群、同一评估器。
物理量按本环境换算：步长 0.1 s，Dmax = DXY_MAX（0.5 m/步 = 5 m/s）。
论文未给数值的四个量（ξ_th、d_min、γ、φ）按环境标度取值，逐项注释在下。
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

# torch 线程数（PF_TORCH_THREADS>0 生效）：小 MLP 多线程负优化，与 C 核
# OpenMP（OMP_NUM_THREADS）解耦——与训练管线同款解耦。
try:
    _tt_sota = int(os.environ.get("PF_TORCH_THREADS", "0") or 0)
except NameError:
    import os as _os
    _tt_sota = 0
if _tt_sota > 0:
    torch.set_num_threads(_tt_sota)


sys.stdout.reconfigure(encoding="utf-8")
from uav_env import (baseline_scene, CKPT_EVERY, DEMAND_SCALE_ANCHOR,  # noqa: E402
                    DXY_MAX, EARLY_SHORT, EARLY_THR, EARLY_WIN,
                    EPISODE_LEN, FIELD_H, FIELD_W, N0_DBM, P_REF_DBM,
                    TOTAL_ROLLOUTS)
from uav_env import UAVEnvV3                                      # noqa: E402
from uav_env import CKPT_ROOT, RESULT_DIR  # noqa: E402

ROLLOUT = 256

# ======================= 论文超参（Table I / §V-A 逐项） =======================
GAMMA = 0.9            # Table I: reward discount factor
TAU = 0.01             # Table I: target network update rate τ
BUF_CAP = 5000         # Table I: memory replay buffer R
BATCH = 1024           # Table I: batch size G
LR = 1e-3              # §V-A 正文：actor/critic 学习率 0.001
                       #   （Table I 的 learning rate σ=1e-5 与正文冲突，取正文）
H1, H2 = 128, 64       # Fig.3/4：两层隐层 128、64，ReLU

# ==================== 论文未给数值的项（按本环境标度） ====================
# 2026-11-02 尺度修复（基线稻草人）：ξ_th 原取 1/B_TOTAL（n12 的 100 MHz），
# 不随场景 b_total 缩放。n200 的 b_total=1666.7 MHz 时奖励比场景归一化值
# 大 16.7 倍；cheng7=140 MHz 时也只差 1.4 倍。现改为逐场景 1/env.b_total，
# 使不同规模的奖励处于同一量级。见 paper_reward()。
D_MIN = 20.0           # eq.13h d_min：UAV 间安全距离（论文未给，取 20 m）
PEN_COL = 1.0          # eq.14 碰撞惩罚幅度 −γ
PEN_BD = 1.0           # eq.14 越界惩罚幅度 −φ
NOISE_SIGMA = 0.1      # §IV-B 中 stochastic noise δ（论文未给强度；动作域 [-1,1] 上 0.1）
# 累计吞吐的折扣：论文回合 200 步，"累计"在 200 步窗口内基本不饱和；
# 本环境回合 2000 步，逐位累加会让 R_m/ΣR 在后期退化成近似常数。
# 取 ρ=0.995 → 等效窗口 1/(1−ρ)=200 步，与论文回合长度对齐。
R_CUM_DECAY = 0.995

# 目标网络软更新为每步一次；论文 Algorithm 1 每步更新。
# 论文 episode 200 步、15000 回合；本环境 2000 步/回合，按 §三 训练预算
# （1200 rollout × 256 步）对齐。


class Actor(nn.Module):
    """§IV-B, Fig.3：3M → 128 → 64 → 2M，ReLU，tanh 限幅到 [-1,1]。

    状态 3M = M 个二维位置 + M 个累计吞吐；动作 2M = M 个 ψ + M 个 D。
    """

    def __init__(self, n_uav: int):
        super().__init__()
        self.n = int(n_uav)
        self.net = nn.Sequential(
            nn.Linear(3 * n_uav, H1), nn.ReLU(),
            nn.Linear(H1, H2), nn.ReLU(),
            nn.Linear(H2, 2 * n_uav))
        nn.init.uniform_(self.net[-1].weight, -3e-3, 3e-3)
        nn.init.zeros_(self.net[-1].bias)

    @classmethod
    def from_ckpt(cls, sd) -> "Actor":
        """按权重第一层输入宽度反推 M（3M 维状态）。"""
        return cls(int(sd["net.0.weight"].shape[1]) // 3)

    def forward(self, s):
        return torch.tanh(self.net(s))


class Critic(nn.Module):
    """§IV-B, Fig.4：(s,a) → 128 → 64 → 1，ReLU；输入 3M+2M。"""

    def __init__(self, n_uav: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(5 * n_uav, H1), nn.ReLU(),
            nn.Linear(H1, H2), nn.ReLU(),
            nn.Linear(H2, 1))

    def forward(self, s, a):
        return self.net(torch.cat([s, a], dim=1)).squeeze(-1)


class PaperState:
    """§IV-A State：全局位置（除以边长）+ 每机累计吞吐占比。"""

    def __init__(self, n_uav: int):
        self.n = n_uav
        self.R_cum = np.zeros(n_uav, dtype=np.float64)

    def reset(self) -> None:
        self.R_cum[:] = 0.0

    def __call__(self, env) -> np.ndarray:
        self.R_cum = R_CUM_DECAY * self.R_cum + env.T
        # 2026-10-31 自查修正：归一化边长取**当前场景**的任务区（原用全局
        # FIELD_W=600，场景放大后位置会超出 [0,1]，观测失真）。
        L = max(env.field_w, env.field_h)
        p = (env.pos / L).ravel()                  # (2N,)
        tot = float(self.R_cum.sum())
        r = (self.R_cum / tot) if tot > 1e-9 else np.zeros(self.n)
        return np.concatenate([p, r]).astype(np.float32)


def decode_action(a: np.ndarray, n_uav: int) -> np.ndarray:
    """§IV-A：tanh 域 [-1,1] → (ψ∈[0,2π], D∈[0,Dmax]) → (Δx,Δy)，资源维≡0。"""
    psi = (a[:n_uav] + 1.0) * np.pi
    dist = (a[n_uav:] + 1.0) * 0.5 * DXY_MAX
    d = np.zeros((n_uav, 5), dtype=np.float32)
    d[:, 3] = dist * np.cos(psi)
    d[:, 4] = dist * np.sin(psi)
    return d


def paper_reward(env) -> float:
    """eq.10/12/14：ξ·f_ϖ·R_sum − 碰撞 − 越界。

    R_n 取 GU 的可达速率（eq.10），不按需求截断——截断后需求已满足的区间
    里奖励对提高 SINR 失去梯度，与论文的"速率最大化"目标不符。

    2026-11-02 尺度修复：ξ_th 改为 1/env.b_total，逐场景归一；原实现用
    全局 B_TOTAL=100 MHz，n200 的 1666.7 MHz 会使奖励尺度偏大 16.7 倍。
    """
    Tu = (env.assoc * env._R).sum(axis=0)
    Rsum = float(Tu.sum())
    fJ = Rsum * Rsum / (len(Tu) * float((Tu * Tu).sum()) + 1e-12)
    r = (1.0 / max(float(env.b_total), 1e-9)) * fJ * Rsum
    p = env.pos
    if len(p) > 1:
        dd = np.sqrt(((p[:, None, :] - p[None, :, :]) ** 2).sum(-1))
        r -= PEN_COL * float(np.triu(dd < D_MIN, 1).sum())
    out = ((p[:, 0] < 0) | (p[:, 0] > env.field_w)
           | (p[:, 1] < 0) | (p[:, 1] > env.field_h))
    r -= PEN_BD * float(out.sum())
    return float(r)


class Replay:
    """论文 Algorithm 1：存放全局转移 (s_t, a_t, r_t, s_{t+1})，容量 5000。

    s 为 3M 维（位置+累计吞吐），a 为 2M 维（ψ,D），两者维度不同。
    """

    def __init__(self, cap: int, s_dim: int, a_dim: int):
        self.cap = cap
        self.s = np.zeros((cap, s_dim), dtype=np.float32)
        self.a = np.zeros((cap, a_dim), dtype=np.float32)
        self.r = np.zeros(cap, dtype=np.float32)
        self.s2 = np.zeros((cap, s_dim), dtype=np.float32)
        self.done = np.zeros(cap, dtype=np.float32)
        self.n, self.i = 0, 0

    def push(self, s, a, r, s2, done) -> None:
        j = self.i
        self.s[j] = s
        self.a[j] = a
        self.r[j] = r
        self.s2[j] = s2
        self.done[j] = done
        self.i = (j + 1) % self.cap
        self.n = min(self.n + 1, self.cap)

    def sample(self, bs: int, rng: np.random.Generator):
        j = rng.integers(0, self.n, bs)
        return self.s[j], self.a[j], self.r[j], self.s2[j], self.done[j]


class OUNoise:
    """§IV-B 的 stochastic noise δ；DDPG 原文用 OU 过程（θ=0.15）。"""

    def __init__(self, dim: int, theta: float = 0.15,
                 sigma: float = NOISE_SIGMA):
        self.dim, self.theta, self.sigma = dim, theta, sigma
        self.x = np.zeros(dim, dtype=np.float64)

    def __call__(self) -> np.ndarray:
        self.x = (self.x + self.theta * (-self.x)
                  + self.sigma * np.random.randn(self.dim))
        return self.x.copy()


def soft_update(net, target) -> None:
    for p, pt in zip(net.parameters(), target.parameters()):
        pt.data.mul_(1.0 - TAU).add_(TAU * p.data)


def learn(actor, a_t, critic, c_t, opt_a, opt_c, buf, rng):
    """Algorithm 1 lines 11–16；eq.17/18/19。"""
    s, a, r, s2, dn = buf.sample(BATCH, rng)
    s_t, s2_t = torch.as_tensor(s), torch.as_tensor(s2)
    a_t_in, r_t, dn_t = torch.as_tensor(a), torch.as_tensor(r), torch.as_tensor(dn)
    with torch.no_grad():
        y = r_t + GAMMA * c_t(s2_t, a_t(s2_t)) * (1.0 - dn_t)     # eq.17
    q = critic(s_t, a_t_in)
    c_loss = nn.functional.mse_loss(q, y)                          # eq.18
    opt_c.zero_grad()
    c_loss.backward()
    nn.utils.clip_grad_norm_(critic.parameters(), 10.0)
    opt_c.step()
    a_loss = -critic(s_t, actor(s_t)).mean()                       # eq.19
    opt_a.zero_grad()
    a_loss.backward()
    nn.utils.clip_grad_norm_(actor.parameters(), 10.0)
    opt_a.step()
    return float(c_loss.detach()), float(q.detach().mean())


class DDPGWrap:
    """评估载体：与训练循环同门（PaperState + decode_action）。"""

    def __init__(self, actor: Actor):
        self.actor = actor
        self.n = int(getattr(actor, "n", 12))
        self.st = PaperState(self.n)

    def reset(self) -> None:
        self.st.reset()

    def act_polar(self, env) -> np.ndarray:
        if env.t == 0:                       # 新回合：累计吞吐归零（§IV-A State）
            self.st.reset()
        s = self.st(env)
        with torch.no_grad():
            a = self.actor(torch.as_tensor(s, dtype=torch.float32)).numpy()
        return decode_action(np.clip(a, -1.0, 1.0), self.n)


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
    assert mode == "ddpga", "本训练器只服务 ddpga 臂"
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

    actor, a_t = Actor(n_uav), Actor(n_uav)
    critic, c_t = Critic(n_uav), Critic(n_uav)
    a_t.load_state_dict(actor.state_dict())
    c_t.load_state_dict(critic.state_dict())
    opt_a = torch.optim.Adam(actor.parameters(), lr=LR)
    opt_c = torch.optim.Adam(critic.parameters(), lr=LR)
    buf = Replay(BUF_CAP, 3 * n_uav, 2 * n_uav)
    st = PaperState(n_uav)
    ou = OUNoise(2 * n_uav)
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
    if start >= total_rollouts:
        # 已完成：给调度器留下完成标记，避免空转重派
        print(f"[{task}] {total_rollouts}/{total_rollouts} "
              f"已完成（断点无剩余轮次）")
        return tdir

    log_f = open(tdir / "train_log.csv", "a" if start > 0 else "w",
                 newline="", encoding="utf-8")
    w = csv.writer(log_f)
    if start == 0:
        w.writerow(["rollout", "env_steps", "mean_ret20", "c_loss", "q_mean",
                    "seconds", "var_u", "u_mean", "sum_t", "n_over"])
        log_f.flush()
    from train_probe import make_probe
    _drain, _reset_len, _win = make_probe(env)

    obs = env.reset()
    st.reset()
    rets: list[float] = []
    ret_curve: list[float] = []
    cl_v = qm_v = 0.0
    t0 = time.time()
    s = st(env)
    for ro in range(start, total_rollouts):
        for _ in range(ROLLOUT):
            with torch.no_grad():
                a = actor(torch.as_tensor(s, dtype=torch.float32)).numpy()
            a = np.clip(a + ou(), -1.0, 1.0)          # eq.15：π(s)+δ
            d = decode_action(a, n_uav)
            env.step(d)
            r = paper_reward(env)                     # eq.14（论文奖励本体）
            s2 = st(env)
            done = env.t >= EPISODE_LEN
            buf.push(s, a.astype(np.float32), r, s2, float(done))
            s = s2
            rets.append(r)
            if buf.n >= BATCH:
                cl_v, qm_v = learn(actor, a_t, critic, c_t, opt_a, opt_c,
                                   buf, rng)
                soft_update(actor, a_t)
                soft_update(critic, c_t)
            if done:
                _drain()
                obs = env.reset()
                st.reset()
                s = st(env)
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
    ap.add_argument("--lr", type=float, default=None,
                    help="超参扫描：覆盖学习率（默认 1e-3）")
    ap.add_argument("--batch", type=int, default=None,
                    help="超参扫描：覆盖批量（默认 1024）")
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
