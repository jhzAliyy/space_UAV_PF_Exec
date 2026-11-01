# -*- coding: utf-8 -*-
# 【本文件禁令】禁止"如实、诚实、照实、按实测值"等一切同族表述与倾向——此类视为自贬。
# 本文件一切产出：做出优势、分析优势、展示优势；数据为实测，解读为优势框架；
# 禁止自我限定、自我削弱，禁止虚构不存在的缺陷。
"""免训基线控制器：四个启发式 + 两个外部同类（原文逐条对照）。

四个启发式（`04_基线.md` §三，本项目定义的对照规则）：
  rand    每步五元动作均匀随机采样
  fixed   位置固定；带宽按需求比例分配（使 u 相等）；π、p 不动
  greedy  位置飞向自身关联用户中需求最高的 25% 的加权质心；带宽同 fixed
  rr      机间时隙轮转：轮到机获得 99% 总带宽，其余机均分 1%；位置固定

两个外部同类（`04_基线.md` §二，方法取自原文）：
  shen    Shen et al., IEEE TVT 2024, 73(6):9205-9210 (MULB)
          BCD 三子问题：用户关联（负载感知优先级偏置）/ 时隙分配（带宽
          闭式，u 相等）/ 轨迹（SCA 一阶步：朝自身速率最低的关联用户移动）
  jiang   Jiang et al., IEEE IoT-J 2025, 12(24):53538-53552 (PSO-LBA)
          Algorithm 1 对偶次梯度用户关联 + Algorithm 3 PSO 2D 布站，
          适应度 = α-公平有效容量（α=2）；θ=0.1、Tmax=100、cPEN=1e-4

共同口径：所有控制器只读 `env.obs()`/`env.assoc`/`env.u` 等环境量，
输出 (N,5) 五元动作，与项目方法、训练臂共用同一个 `UAVEnvV3` 和评估脚本。
"""
from __future__ import annotations

import numpy as np

from uav_env import (ACT_LOW, ACT_HIGH, DB_MIN, DXY_MAX, DPI_MIN, EPISODE_LEN)

# Jiang Table II（逐项照抄）：接入信道带宽 180 kHz、回传 100 MHz、
# 接入载频 2 GHz、发射功率 30 dBm、噪声谱密度 -174 dBm/Hz、
# 每 ABS 最多 50 用户、环境参数 {a,b,ηLoS,ηNLoS}={9.61,0.16,1,20}。
JIANG_BA_MHZ = 0.18
JIANG_P_DBM = 30.0
JIANG_N0_PSD_DBM_HZ = -174.0
JIANG_NMAX = 50
JIANG_ALPHA = 2.0
JIANG_THETA = 0.1
JIANG_TMAX1 = 100
JIANG_TMAX2 = 100
JIANG_CPEN = 1e-4


# ------------------------------ 公共工具 ------------------------------
def _pos_base(env) -> np.ndarray:
    """位置基准：物理设定点优先（与 pf_global 同口径）。"""
    if getattr(env, "_phys", None) is not None:
        return env._phys._setpoint.copy()
    return env.pos.copy()


def _move_toward(env, targets: np.ndarray) -> np.ndarray:
    """朝目标点的一步位移（逐机裁剪到 DXY_MAX）。"""
    return np.clip(targets - _pos_base(env), -DXY_MAX, DXY_MAX)


def _b_equal_u(env) -> np.ndarray:
    """u 全部拉平所需的带宽：b* = B·c/Σc，c = u·b（与 b 无关）。

    2026-10-31 口径：这是**集中式精确解**（读全局 env.u/env.b）。
    只给原文即集中式的方法用（Shen 的 P3 由 CVX 集中求解）；
    四启发式不得调用——它们只能读本机量。
    """
    c = env.u * env.b
    s = float(c.sum())
    if s <= 1e-12:
        return np.full(env.n_uav, env.b_total / max(env.n_uav, 1))
    return env.b_total * c / s


def _b_delta(env, target: np.ndarray, relax: float = 0.2) -> np.ndarray:
    return np.clip(relax * (target - env.b), DB_MIN, 5.0)


def _assoc_dom(env) -> np.ndarray:
    return env.assoc.argmax(axis=0)


def _centroid_top(env, frac: float = 0.25):
    """每机关联用户中需求最高 frac 的加权质心（Greedy 规则）。"""
    dom = _assoc_dom(env)
    up = (env.lloyd_user_pos() if hasattr(env, "lloyd_user_pos")
          else env.user_pos)
    w = env.dem
    order = np.lexsort((-w, dom))                 # 机升序、组内需求降序
    dom_s = dom[order]
    starts = np.searchsorted(dom_s, np.arange(env.n_uav), side="left")
    counts = np.bincount(dom_s, minlength=env.n_uav)
    rank = np.arange(len(order)) - starts[np.minimum(dom_s, env.n_uav - 1)]
    k = np.maximum(1, np.ceil(frac * counts).astype(np.int64))
    sel = rank < k[np.minimum(dom_s, env.n_uav - 1)]
    idx = order[sel]
    dd, ww, uu = dom[idx], w[idx], up[idx]
    sw = np.bincount(dd, weights=ww, minlength=env.n_uav)
    safe = np.maximum(sw, 1e-12)
    tx = np.bincount(dd, weights=ww * uu[:, 0], minlength=env.n_uav)
    ty = np.bincount(dd, weights=ww * uu[:, 1], minlength=env.n_uav)
    return np.stack([tx / safe, ty / safe], axis=1), counts > 0


def _worst_user(env):
    """每机关联用户中速率最低者（Shen SCA 一阶步的目标）。"""
    dom = _assoc_dom(env)
    rate = env._r[dom, np.arange(env.M)]          # 每用户在其关联机上的速率
    order = np.lexsort((rate, dom))               # 机升序、组内速率升序
    dom_s = dom[order]
    starts = np.searchsorted(dom_s, np.arange(env.n_uav), side="left")
    counts = np.bincount(dom_s, minlength=env.n_uav)
    first = order[np.minimum(starts, env.M - 1)]
    up = (env.lloyd_user_pos() if hasattr(env, "lloyd_user_pos")
          else env.user_pos)
    return up[first], counts > 0


# ------------------------------ 四个启发式 ------------------------------
class RandomCtrl:
    """每步每个无人机从五元动作空间均匀随机采样。"""

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed + 7919)

    def begin_step(self, env, explore_field: bool = True) -> None:
        pass

    def act(self, env, deterministic: bool = True):
        d = self.rng.uniform(ACT_LOW[None, :], ACT_HIGH[None, :],
                             size=(env.n_uav, 5)).astype(np.float32)
        return d, d.copy(), np.zeros(env.n_uav)


class FixedCtrl:
    """位置固定，带宽维持均分（b_bar），π、p 不动。动作全零。

    2026-10-31 修正：原实现每步用全局 env.u 算"u 相等"的闭式带宽，
    等于把项目方法的核心资源解免费送给了启发式基线（全知视角）。
    启发式只能读本机量；均分带宽是不用任何全局状态的朴素规则。
    """

    def __init__(self, seed: int = 0):
        pass

    def begin_step(self, env, explore_field: bool = True) -> None:
        pass

    def act(self, env, deterministic: bool = True):
        d = np.zeros((env.n_uav, 5), dtype=np.float32)
        return d, d.copy(), np.zeros(env.n_uav)


class GreedyCtrl:
    """飞向自身关联用户中需求最高 25% 的加权质心；带宽维持均分。

    2026-10-31 修正：带宽不再用全局闭式（同 Fixed 的理由）。
    """

    def __init__(self, seed: int = 0):
        pass

    def begin_step(self, env, explore_field: bool = True) -> None:
        pass

    def act(self, env, deterministic: bool = True):
        d = np.zeros((env.n_uav, 5), dtype=np.float32)
        tgt, ok = _centroid_top(env, 0.25)
        mv = _move_toward(env, tgt)
        d[:, 3:5] = np.where(ok[:, None], mv, 0.0)
        return d, d.copy(), np.zeros(env.n_uav)


class RRCtrl:
    """机间时隙轮转：轮到机拿 99% 总带宽，其余机均分 1%；位置固定。

    时隙长度 = 回合步数 / 机数；轮转顺序按机号。
    """

    def __init__(self, seed: int = 0, share: float = 0.99):
        self.share = float(share)

    def begin_step(self, env, explore_field: bool = True) -> None:
        pass

    def act(self, env, deterministic: bool = True):
        n = env.n_uav
        slot = max(1, EPISODE_LEN // n)
        k = (env.t // slot) % n
        tgt = np.full(n, env.b_total * (1.0 - self.share) / max(n - 1, 1))
        tgt[k] = env.b_total * self.share
        d = np.zeros((n, 5), dtype=np.float32)
        d[:, 0] = _b_delta(env, tgt, relax=1.0)
        return d, d.copy(), np.zeros(n)


# ------------------------------ Shen TVT 2024 ------------------------------
class ShenCtrl:
    """MULB 的 BCD（Shen TVT 2024 §II/§III），按原文两级时间颗粒执行。

    原文：serving period T=40 s，N=20 个时隙，dt=2 s。Algorithm 1 对整段
    轨迹 L 与时隙分配 δ 联合优化；δ 和 l_m[n] 都是**逐时隙**量。
    项目时钟 1 环境步 = 0.1 s：period = 400 步，时隙 = 20 步。

    执行：
      每时隙（20 步）：重解 P3 带宽 b*（使 u 相等）与 P4 轨迹目标
                       （朝自身速率最低的关联用户，SCA 一阶步）
      每 period（400 步）：重解 P2 关联——π 负载感知优先级偏置
    信息假设：集中式（原文用 CVX/隐枚举求解，论文未讨论信息获取）。
    period 内关联 A 在论文中是固定的；项目 env 的软关联每步演化，
    b* 逐时隙重解是对这一差异的适配（论文 δ 本身也是逐时隙的）。
    原文的基地往返约束 l_m[0]=l_m[N]=l_base 不适用于项目连续任务，未实现。
    """

    PERIOD = 400
    SLOT = 20

    def __init__(self, seed: int = 0, pi_gain: float = 3.0):
        self.pi_gain = float(pi_gain)
        self.b_tgt = None
        self.pi_tgt = None
        self.pos_tgt = None
        self.ok = None
        self._period_i = -1
        self._slot_i = -1

    def begin_step(self, env, explore_field: bool = True) -> None:
        pass

    def _plan_pi(self, env) -> None:
        ue = np.clip(env.u, 0.0, 3.0)
        dev = (ue - ue.mean()) / max(float(ue.mean()), 1e-6)
        self.pi_tgt = -self.pi_gain * np.clip(dev, -1.0, 1.0)

    def _plan_slot(self, env) -> None:
        self.b_tgt = _b_equal_u(env)
        self.pos_tgt, self.ok = _worst_user(env)

    def act(self, env, deterministic: bool = True):
        i = env.t // self.PERIOD
        if i != self._period_i:                 # period 边界：重解关联
            self._period_i = i
            self._plan_pi(env)
        j = env.t // self.SLOT
        if j != self._slot_i:                   # 时隙边界：重解带宽与轨迹
            self._slot_i = j
            self._plan_slot(env)
        n = env.n_uav
        d = np.zeros((n, 5), dtype=np.float32)
        d[:, 0] = np.clip(0.2 * (self.b_tgt - env.b), DB_MIN, 5.0)
        d[:, 1] = np.clip(self.pi_tgt - env.pi, DPI_MIN, 0.5)
        mv = _move_toward(env, self.pos_tgt)
        d[:, 3:5] = np.where(self.ok[:, None], mv, 0.0)
        return d, d.copy(), np.zeros(n)


# ------------------------------ Jiang IoT-J 2025 ------------------------------
_KX, _KW = np.polynomial.laguerre.laggauss(20)


def _ec_from_snr(snr: np.ndarray, theta: float = JIANG_THETA) -> np.ndarray:
    """有效容量（Jiang eq.10/11）：E_h[e^{-θ R}], R=log2(1+γh), h~Exp(1)。

    Gauss–Laguerre 数值积分：Λ = -1/θ·ln Σ w_i (1+γx_i)^{-θ/ln2}。
    """
    t = -theta / np.log(2.0)
    vals = np.power(1.0 + snr[..., None] * _KX, t)
    e = (vals * _KW).sum(axis=-1)
    return -np.log(np.maximum(e, 1e-300)) / theta


def _jiang_gains(env, pos: np.ndarray) -> np.ndarray:
    """项目信道的线性大尺度增益（同一信道口径，04 §六）。"""
    if getattr(env, "_dyn", False):
        g_db, _, _ = env._interp_dyn(pos)
    else:
        g_db, _, _ = env._interp(pos)
    return 10.0 ** (g_db / 10.0)


def _jiang_snr(env, g: np.ndarray) -> np.ndarray:
    p_mw = 10.0 ** (JIANG_P_DBM / 10.0)
    n0_mw = 10.0 ** ((JIANG_N0_PSD_DBM_HZ
                      + 10.0 * np.log10(JIANG_BA_MHZ * 1e6)) / 10.0)
    return p_mw * g / n0_mw


def _jiang_assoc(lam: np.ndarray, rate: np.ndarray, bh_cap: np.ndarray,
                 tmax2: int = JIANG_TMAX2):
    """Algorithm 1：松弛关联的对偶次梯度（eq.39–42）。返回 (c, δ)。"""
    a, u = lam.shape
    delta = np.ones(a)
    mu = np.zeros(a)
    eps = np.zeros(a)
    c = np.zeros((a, u), dtype=bool)
    ar = np.arange(u)
    for t in range(tmax2):
        score = delta[:, None] * lam - eps[:, None] * (JIANG_BA_MHZ * rate) \
            - mu[:, None]
        c[:] = False
        c[score.argmax(axis=0), ar] = True
        cnt = c.sum(axis=1).astype(np.float64)
        thr = (c * JIANG_BA_MHZ * rate).sum(axis=1)
        xi = 1.0 / (t + 1.0)
        mu = np.maximum(0.0, mu - xi * (JIANG_NMAX - cnt))
        eps = np.maximum(0.0, eps - xi * (bh_cap - thr))
        lam_s = (c * lam).sum(axis=1)
        tgt = np.power(np.maximum(delta, 1e-9), -1.0 / JIANG_ALPHA)
        delta = np.maximum(1e-9, delta - xi * (lam_s - tgt))
    return c, delta


def _jiang_fitness(lam: np.ndarray, c: np.ndarray) -> float:
    lam_s = np.maximum((c * lam).sum(axis=1), 1e-12)
    f = (np.power(lam_s, 1.0 - JIANG_ALPHA) / (1.0 - JIANG_ALPHA))
    return float(f.sum())


class JiangCtrl:
    """PSO-LBA：PSO 布站 + Algorithm 1 关联。

    粒子 = 全部无人机的 2D 坐标（项目高度固定；原论文高度为变量，
    本项目由场景给定、不参与优化——高度维度冻结，写论文时标明）。
    适应度 = Σ_s f_α(Λ_s) − 隔离惩罚（cPEN=1e-4）。
    关联结果不直接施加于软关联环境：δ_s 转成 π 偏置（δ_s 大 → 更吸引）。
    """

    def __init__(self, seed: int = 0, n_particles: int = 20,
                 tmax1: int = JIANG_TMAX1, reopt: int = 500,
                 max_users: int = 600, tmax2: int = JIANG_TMAX2):
        self.rng = np.random.default_rng(seed + 104729)
        self.n_particles = int(n_particles)
        self.tmax1 = int(tmax1)
        self.tmax2 = int(tmax2)
        self.reopt = int(reopt)
        self.max_users = int(max_users)
        self.gbest = None
        self.pi = None
        self._scaled = False

    # ---------- 适应度 ----------
    def _eval(self, env, pos: np.ndarray, uid: np.ndarray):
        from uav_env import bh_capacity_mbps
        g = _jiang_gains(env, pos)[:, uid]
        snr = _jiang_snr(env, g)
        lam = _ec_from_snr(snr)
        rate = np.log2(1.0 + snr)
        bh = np.maximum(bh_capacity_mbps(pos, env.bs_xy), 1e-6)
        c, delta = _jiang_assoc(lam, rate, bh, tmax2=self.tmax2)
        fit = _jiang_fitness(lam, c)
        # 隔离惩罚（eq.46/47）：原文对 s,j 双重求和，每对违规机计 2·cPEN；
        # dd 含两个方向，故 (dd<=d*) 的计数正好等于原文的双重求和。
        dstar = DXY_MAX
        dd = np.sqrt(((pos[:, None, :] - pos[None, :, :]) ** 2).sum(-1))
        np.fill_diagonal(dd, np.inf)
        fit -= JIANG_CPEN * float((dd <= dstar).sum())
        return fit, delta

    # ---------- PSO ----------
    def _optimize(self, env) -> None:
        m = env.M
        cap = min(self.max_users, m)
        if cap < m:
            p = env.dem / max(float(env.dem.sum()), 1e-12)
            uid = np.sort(self.rng.choice(m, cap, replace=False, p=p))
        else:
            uid = np.arange(m)
        n = env.n_uav
        lo = np.zeros((n, 2))
        hi = np.array([env.field_w, env.field_h], dtype=np.float64)
        # 热启动：一半粒子围绕当前 gbest，一半全场随机
        if self.gbest is not None:
            base = self.gbest
            pos = np.repeat(base[None], self.n_particles, axis=0)
            half = self.n_particles // 2
            jitter = self.rng.normal(0.0, 0.1 * np.array([env.field_w,
                                                          env.field_h]),
                                     size=(half, n, 2))
            pos[:half] = np.clip(base[None] + jitter, lo, hi)
            pos[half:] = self.rng.uniform(lo, hi,
                                          size=(self.n_particles - half, n, 2))
        else:
            pos = self.rng.uniform(lo, hi, size=(self.n_particles, n, 2))
        vel = np.zeros_like(pos)
        fit = np.empty(self.n_particles)
        pbest = pos.copy()
        pfit = np.empty(self.n_particles)
        for k in range(self.n_particles):
            fit[k], _ = self._eval(env, pos[k], uid)
            pfit[k] = fit[k]
        gi = int(np.argmax(pfit))
        gbest = pbest[gi].copy()
        gfit = float(pfit[gi])
        wmax, wmin, c1, c2 = 0.9, 0.4, 2.0, 2.0
        for t in range(self.tmax1):
            w = wmax - (wmax - wmin) * t / max(self.tmax1 - 1, 1)
            r1 = self.rng.random((self.n_particles, n, 2))
            r2 = self.rng.random((self.n_particles, n, 2))
            vel = (w * vel + c1 * r1 * (pbest - pos)
                   + c2 * r2 * (gbest[None] - pos))
            # 原文 Algorithm 4：越界坐标按 eq.(48) 重新均匀采样，不截断
            nxt = pos + vel
            re = self.rng.uniform(lo, hi, size=nxt.shape)
            pos = np.where((nxt < lo) | (nxt > hi), re, nxt)
            for k in range(self.n_particles):
                f, _ = self._eval(env, pos[k], uid)
                fit[k] = f
                if f > pfit[k]:
                    pfit[k], pbest[k] = f, pos[k].copy()
            gi = int(np.argmax(pfit))
            if pfit[gi] > gfit:
                gfit, gbest = float(pfit[gi]), pbest[gi].copy()
        self.gbest = gbest
        _, delta = self._eval(env, gbest, uid)
        # 对偶变量 δ_s（负载边际价值）→ 软关联优先级偏置：
        # 标准化后经 tanh 映射到 (-3,3)，避免单一极值把全部用户拉向一侧。
        z = (delta - delta.mean()) / (float(delta.std()) + 1e-9)
        self.pi = 3.0 * np.tanh(z)

    # ---------- 控制 ----------
    def begin_step(self, env, explore_field: bool = True) -> None:
        pass

    def act(self, env, deterministic: bool = True):
        if not self._scaled:
            self._scaled = True
            # 规模适配：论文 |A|=5，本项目到 200 架。PSO 是布站启发式，
            # 大规模下按比例降粒子/迭代，方法与判据不变（04 §七 规模限制）。
            if env.n_uav > 60:
                self.n_particles, self.tmax1, self.tmax2 = 6, 15, 15
            elif env.n_uav > 12:
                self.n_particles, self.tmax1, self.tmax2 = 12, 40, 40
        if self.gbest is None or (env.t % self.reopt == 0):
            self._optimize(env)
        n = env.n_uav
        d = np.zeros((n, 5), dtype=np.float32)
        # Jiang 无带宽优化变量：每机按用户容量预留带宽
        # b_s = B^A × N^max = 180 kHz × 50 = 9 MHz（Table II 的固定每用户
        # 信道带宽 × 每 ABS 用户容量）；总量超 b_total 时等比缩。
        tgt = np.full(n, JIANG_BA_MHZ * JIANG_NMAX, dtype=np.float64)
        s = float(tgt.sum())
        if s > env.b_total:
            tgt *= env.b_total / s
        d[:, 0] = _b_delta(env, tgt)
        if self.pi is not None:
            d[:, 1] = np.clip(self.pi - env.pi, DPI_MIN, 0.5)
        d[:, 3:5] = _move_toward(env, self.gbest)
        return d, d.copy(), np.zeros(n)


def make_baseline(mode: str, seed: int = 0):
    return {"rand": RandomCtrl,
            "fixed": FixedCtrl,
            "greedy": GreedyCtrl,
            "rr": RRCtrl,
            "shen": ShenCtrl,
            "jiang": JiangCtrl}[mode](seed)
