# -*- coding: utf-8 -*-
"""UAV 环境本体（2026-10-31 结构重置）：场景、人群、信道、关联、负载、
动作解码与奖励。共享超参数在 config.py，本文件只保留物理实现。"""
from __future__ import annotations
from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np

from config import (CKPT_EVERY, EARLY_SHORT, EARLY_THR, EARLY_WIN,
                    EPISODE_LEN, LOAD_FLOOR, LOAD_SLOPE, OBS_NORM, PF_RELAX,
                    PF_U_MAP, PI_DUAL_STEP, PI_MAX, PI_MIN, R_MIN, SE_REF,
                    SIGMA, TOTAL_ROLLOUTS, U_OVER, U_STAR, W_BAL, W_EFF,
                    W_LOAD)

U_STAR_DEFAULT = U_STAR          # 兼容旧导入名；值来自 config

BASE = Path(__file__).resolve().parent.parent          # 项目根（相对锚）
TABLE_DIR = BASE / "env_data" / "channel"               # 信道表
RESULT_DIR = BASE / "results"                           # 结果根
CKPT_ROOT = BASE / "code" / "ckpt"                      # 训练存档根

# ----------------------------- 场地与机队（12 机基准，密度不变口径） -----------------------------
FIELD_W, FIELD_H = 600.0, 450.0
N_UAV = 12
GRID_COLS, GRID_ROWS = 4, 3
GRID_SPACING = 150.0
UAV_ALT = 50.0
NEIGH_DIST = 250.0

# ----------------------------- 用户与需求 -----------------------------
N_USER = 600
HOTSPOTS = ((150.0, 112.0), (450.0, 150.0), (300.0, 340.0))
HOTSPOT_FRAC = 0.25
HOTSPOT_SIGMA = 60.0
LAYOUT_SEED = 20261024             # 用户布点固定种子（与训练种子解耦）
DEMAND_MEAN, DEMAND_STD = 0.5, 0.15

# ----------------------------- 信道表 -----------------------------
CHAN_RES = 10.0
CHAN_SCENARIO = "uma"
CHAN_FC = 2.0e9
CHAN_SHADOW_SEED = 777
N0_DBM = -104.0                    # 热噪底 dBm（物理标准值）
P_REF_DBM = 24.0                   # 参考发射功率 dBm（物理标准值）
# 需求强度锚定（2026-10-27 用户裁定：原始锚定方法，均匀基准一次推导）：
# 无人机网格均匀 + 人群均匀分布（密度一致）时，平均负载恰好锚在
# u_star=0.85 健康区间——密度不变口径下该锚与规模无关，全规模统一常数；
# 场景（热点/移动/聚集）的 u 偏离=场景张力，自然涌现，不人为拟合。
DEMAND_SCALE_ANCHOR = 3.7648       # ops/anchor_demand.py 均匀基准推导
#                                  （n12 均匀人群 u_mean=0.2125@scale1 线性
#                                   锚至平台带中心 0.8；密度不变=>全规模通用）
P_MIN, P_MAX = 0.5, 2.0
B_TOTAL = 100.0

# ----------------------------- 回传链路（转发中继） -----------------------------
# 场景定性（用户裁定）：无人机是**转发中继**，不是独立基站。链路两段：
#   用户 --(接入)--> 无人机 --(回传)--> 地面宿主基站 TBS
# 端到端容量 = min(接入段, 回传段)，约束形式同 Jiang 2025 式 (19)。
#
# 全部参数逐项照抄 Jiang et al., "Joint Aerial Base Station Placement and User
# Association for Latency-Driven Load Balancing in Aerial Mobile Networks",
# IEEE Internet of Things Journal, Vol.12, No.24, pp.53538-53552, Dec. 2025,
# DOI 10.1109/JIOT.2025.3617469 —— §II-A 回传链路模型 + Table II 仿真参数。
# 一个数都不自造。
#
# 回传路损：L^B = 61.4 + 20·log10(d)（28 GHz 毫米波，d 为机-TBS 三维距离）
# 回传容量：C^B = B^B · log2(1 + p^B·g^B/I^B)，I^B = N0·B^B
BH_FC_HZ = 28e9                  # 回传载频 28 GHz（Table II）
BH_BW_HZ = 100e6                 # 每条回传信道带宽 100 MHz（Table II，各机正交独立）
P_BS_DBM = 43.0                  # TBS 回传发射功率 43 dBm（Table II）
N0_PSD_DBM_HZ = -174.0           # 噪声功率谱密度 -174 dBm/Hz（Table II）
BS_POS = (0.0, 0.0)              # TBS 位于场地原点地面（§II 的 {0,0,0}）
RELAY_ENABLE = os.environ.get("PF_RELAY", "1") == "1"   # 中继开关（消融用）

# ----------------------------- 阴影（σ 随机化，2026-10-30 用户裁定） -----------------------------
# 用户裁定：阴影标准差随机，不写死一档。
# 语义：
#   PF_SHADOW_DB=random（默认）→ 每回合从 [MIN, MAX] 均匀抽一个 σ，
#       训练时策略见过从无阴影到最差阴影的全谱；
#   PF_SHADOW_DB=<数值>        → 固定该值，评估/复现时用，保证可比。
# 区间依据：0 = 无阴影（理想），13.2 = 项目信道表拟合后的实测残差
# （23 号文档"表内残差"）；6 = 3GPP 38.901 UMa NLoS 标准值，落在区间内。
# 阴影本身仍然每 SHADOW_T=100 步（50 米相干距离）刷新一次——σ 定"波动多大"，
# 刷新定"什么时候变"，两者独立。
_sd_env = os.environ.get("PF_SHADOW_DB", "random")
SHADOW_RANDOM = _sd_env.strip().lower() in ("random", "rand", "r")
SHADOW_SIGMA_FIXED = 0.0 if SHADOW_RANDOM else float(_sd_env)
SHADOW_SIGMA_MIN, SHADOW_SIGMA_MAX = 0.0, 13.2
SHADOW_EPISODE_SEED = 90210      # 每回合抽 σ 的种子偏移（与训练种子组合）

# ----------------------------- Lloyd 质心的信息获取（2026-10-30 用户裁定） -----------------------------
# 质心用的用户位置**不再是精确全局量**：加 10 米测量误差、延迟 0.5 秒。
#
# 依据：历史实测显示，这个区间内近似无损，超出才开始劣化——所以这两个数是
# **无损上界**，不是随手定的。固化为一档默认口径，不扫参。
#
# 6G 语境：6G 定位预期厘米级、端到端时延亚毫秒，10 米 / 0.5 秒比那宽松两三个
# 数量级。用这档等于在一组对方法更不利的条件下测。
LLOYD_POS_ERR_M = 10.0           # **每维**位置测量误差标准差（米）；
#                                  二维合成径向误差均值 = σ·√(π/2) ≈ 12.5 米，属十米级
LLOYD_DELAY_S = 0.5              # 位置信息延迟（秒）
LLOYD_DELAY_STEPS = 5            # = LLOYD_DELAY_S / 0.1 s 每环境步

# ----------------------------- 五元动作 -----------------------------
DB_MIN, DB_MAX = -5.0, 5.0
DPI_MIN, DPI_MAX = -0.5, 0.5
DP_MIN, DP_MAX = -0.1, 0.1
# 位置维上限：2026-10-28 时钟统一后 = 每环境步 0.5 m ÷ 0.1 s = 5 m/s。
# 实测该指令下 MuJoCo CF2X（27 g / 推重比 2.25）达成 3.6~4.4 m/s，
# 与真实 Crazyflie 2.1 的巡航档一致。原值 5.0 m/步 是把它当 50 m/s 用。
DXY_MAX = float(os.environ.get("PF_DXY_MAX", "0.5"))
#   环境变量可覆盖：基线论文的 UAV 速度是 20~25 m/s（Cheng TIM 2024 Vmax=20、
#   Chen T-ITS 2024 Vmax=25），本项目默认 0.5 m/步 = 5 m/s 属保守真实值。
#   用 PF_DXY_MAX=2.0 可复现 20 m/s 条件，专供"移动能力敏感性"对照。
ACT_DIM = 5
ACT_LOW = np.array([-5.0, -0.5, -0.1, -DXY_MAX, -DXY_MAX])
ACT_HIGH = np.array([5.0, 0.5, 0.1, DXY_MAX, DXY_MAX])
ACT_SCALE = np.array([5.0, 0.5, 0.1, DXY_MAX, DXY_MAX])
# 执行器输出维（用户裁定）：σ = [σ_b, σ_π, σ_p, σ_xy]，四个归一化速率比例。
# 三个通信量各一个倍率，位置 x/y 共用一个。只有这一种形状。
SIG_DIM = 4



# 线程修复（imprv1，2026-10-28）：小图梯度链上多线程"集合-同步"开销
# 远超并行收益——受控实测老云 32 线程 38.0s vs 4 线程 0.19s（慢 200 倍）、
# 新云 128 线程 5.6s vs 4 线程 0.33s（慢 17 倍）；两机单核持平、软件栈相同。
# 单值锚定 4（受控基准两机实测值，不扫参）。运行时 torch.set_num_threads 生效。
TORCH_THREADS = 4
# 任务名尾部自动追加 _imprv1，与旧口径（开关=0、无限程修复）目录隔离。

# ----------------------------- 势场 -----------------------------
# 定义见 field.py。两个参数，每个都有实测依据：
#   PF_U_MAP  C_logc：u≤3 严格恒等（不破坏健康带），u>3 严格单调无封顶。
#             A_clip3 在 u>3 段导数恒 0，过载方向全灭（实测 7/7 状态）；
#             B_clip30 同一缺陷，阈值抬高。
# 信道标定来源（2026-10-29 用户令：切换开关统一放 config）
#   table = 从射线追踪表拟合 C0/γ（现行主口径）
#   umi   = 解析构造，3GPP TR 36.777 表 B-2 的 UMi-AV 两行（微站）
#   uma   = 解析构造，3GPP TR 36.777 表 B-2 的 UMa-AV 两行（宏站）
# 环境变量 PF_CHANNEL 可覆盖；两种解析档内部口径一致，只换公式。
PF_CHANNEL = os.environ.get("PF_CHANNEL", "table")

# ----------------------------- 关联与速率链 -----------------------------
ALPHA_ASSOC = 1.0
KAPPA = 0.5
TAU_U_DEFAULT = 5.0
BW_SHARE = "prop_demand"
EPS_U = 1e-9
EPS_D = 0.1

# ----------------------------- 奖励（现行实现） -----------------------------
# 三个子目标，权重和 = 1，全部输入来自本机观测：
#   均衡度 bal  = 1/(1 + Σ_j(û_i−û_j)²/(deg_i+1))
#   负载度 load = max(R_MIN, 1−((û_i−u*)/SIGMA)²)
#   吞吐度 eff  = clip(T_i/b_bar/SE_REF, 0, 1)
#   û=load_view(u)，与观测第 0 维同一映射；常量全部在 config.py。

# ----------------------------- 物理层 -----------------------------
# 2026-10-28 时钟统一（用户裁定）。三个错位一次性修干净：
#
# 错在哪：
#   1. 两个引擎的时间流速不一致——同一个 step() 调用，人群引擎推进 SFM_DT
#      =0.1 s，无人机推进 PHYS_SUBSTEP/CTRL_FREQ=1/40=0.025 s（差 4 倍）。
#      物理世界里两者共享同一个时钟，不存在「视角换算」。
#   2. 无人机等效速度 = DXY_MAX/(PHYS_HOLD × 环境步长) = 5/(40×0.025)
#      = 1.2 m/s = 4.5 km/h，比人群快组（1.39 m/s）还慢、比冲刺（2.78）慢
#      2.3 倍。连 400 m 跑道都要飞 89 秒，而回合只有 50 秒。
#      而机体本身（MuJoCo CF2X，27 g/推重比 2.25，与真实 Crazyflie 2.1 一致）
#      实测能飞 14.7 m/s = 53 km/h——被算成了 4.5 km/h，差 12 倍。
#   3. 设定点以 5 m/s 前进而机体只能 1.2 m/s 追 → 目标跑飞，实测落后量
#      1 s 时 4.5 m、7 s 时 12.5 m 且仍在增大。飞机在追一个不存在的点。
#
# 修法（内环毫秒级本地、外环秒级决策，各按各的实际）：
#   内环 PID：CTRL_FREQ=40 Hz，每环境步 PHYS_SUBSTEP=4 拍 → 环境步 = 0.1 s
#   外环设定点：PHYS_HOLD=1，每环境步更新一次（=10 Hz 决策，秒级档）
#   速度：DXY_MAX=0.5 m/步 ÷ 0.1 s = 5 m/s（实测达成 3.6~4.4 m/s，留余量）
#   回合 2000 步 = 200 s（与人群引擎原本时间轴一致，SFM_DT 不用动）
#   每机总行程 200 s × 5 m/s = 1000 m（原 250 m）
SIM_FREQ = 120                      # MuJoCo 物理仿真频率 Hz（毫秒级，不动）
PHYS_MODE = "mjc"                   # 物理模式（mjc=MuJoCo 主线）
CTRL_FREQ = 40                      # PID 内环频率 Hz（毫秒级本地，不动）
PHYS_SUBSTEP = 4                    # 每环境步的控制拍数 → 4/40 = 0.1 s
PHYS_HOLD = 1                       # 每环境步更新设定点（外环 10 Hz）
WIND = False

# ----------------------------- 回合与观测 -----------------------------
OBS_DIM = 12

# ----------------------------- 场景（两个真场景） -----------------------------
SCENES = ("flow", "burst")

# 人群移动：社会力模型（lightsfm C++，确定性=共享基底）
SFM_VSLOW = 0.42                    # 慢组期望速度 m/s（1.5 km/h）
SFM_VFAST = 1.39                    # 快组期望速度 m/s（5.0 km/h）
SFM_RADIUS = 0.35                   # 行人体半径 m（lightsfm 默认）
SFM_CELL = 3.0                      # 空间网格 cell = 近邻截断半径 m
SFM_DT = 0.1                        # 人群步长 s（= 环境步长，见 ENV_DT）
#                                   回合 2000 步=200s 人群时间=3.3 分钟，
#                                   快组累计行程 ~278m、慢组 ~84m——
#                                   分钟级空间重构口径，01 文档场景时间轴）
ENV_DT = PHYS_SUBSTEP / CTRL_FREQ   # 每环境步物理时 s（2026-10-28 统一：
#                                   必须等于 SFM_DT，两引擎共享同一时钟）

# 突发聚集（burst 场景）
GATHER_T0 = 600                     # 聚集触发步
GATHER_K = 2                        # 聚集点数（出口，env 由场地推导）
GATHER_R_FRAC = 0.5                 # 聚集驻留半径 = 热点 σ × 此系数
SFM_V_RUSH = 2.78                   # 涌向速度 m/s（10 km/h，3GPP 步行档顶）

# 通讯退化轴（默认 1=每步交换）
T_COMM_DEFAULT = 1

# ----------------------------- 训练超参（统一入 config） -----------------------------
ROLLOUT = 256
EPOCHS = 4
MINIBATCH = 256
CLIP = 0.2
LR = 3e-4
GAMMA, GAE_LAM = 0.99, 0.95
ENT_COEF = 0.01
# 训练预算/早停参数见 config.py。


# ----------------------------- 布局与规模（密度不变解耦） -----------------------------
def grid_layout(n_uav: int) -> tuple[int, int]:
    cols = math.ceil(math.sqrt(n_uav * 4.0 / 3.0))
    rows = math.ceil(n_uav / cols)
    return cols, rows


def layout_custom(n_user: int, fw: float, fh: float,
                  n_hotspots: int = 3, hotspot_sigma: float = 60.0,
                  seed: int = LAYOUT_SEED):
    """自定义场景布局（2026-10-30）：按基线论文的原场景参数布点。

    返回 (user_pos(M,2), hot_users, hotspots)，与 user_layout 同结构。
    固定种子，确定性。
    """
    rs = np.random.default_rng(seed)
    hotspots = [list(c) for c in rs.uniform(
        [0.1 * fw, 0.1 * fh], [0.9 * fw, 0.9 * fh],
        size=(n_hotspots, 2))]
    n_hot = int(n_user * 3 * HOTSPOT_FRAC / len(hotspots))
    pos = [rs.normal(c, hotspot_sigma, size=(n_hot, 2)) for c in hotspots]
    n_tail = n_user - len(hotspots) * n_hot
    pos.append(rs.uniform(0, 1, size=(n_tail, 2)) * [fw, fh])
    user_pos = np.clip(np.concatenate(pos), [0, 0], [fw, fh])
    _, _, inverse, counts = np.unique(
        user_pos, axis=0, return_index=True,
        return_inverse=True, return_counts=True)
    if (counts > 1).any():
        seen = {}
        for j in range(len(user_pos)):
            k = seen.get(tuple(user_pos[j]), 0)
            seen[tuple(user_pos[j])] = k + 1
            if k > 0:
                user_pos[j] = user_pos[j] + k * 0.01 * np.array([1.0, 0.5])
        user_pos = np.clip(user_pos, [0, 0], [fw, fh])
    hot_users = [np.arange(k * n_hot, (k + 1) * n_hot)
                 for k in range(len(hotspots))]
    return user_pos, hot_users, hotspots


def uav_grid_custom(n: int, fw: float, fh: float,
                    spacing: float = GRID_SPACING) -> np.ndarray:
    """自定义场地的初始机位网格（2026-10-30）：矩形排布，居中。"""
    cols = max(1, int(np.ceil(np.sqrt(n * fw / max(fh, 1e-9)))))
    rows = max(1, int(np.ceil(n / cols)))
    x0 = (fw - (cols - 1) * spacing) / 2
    y0 = (fh - (rows - 1) * spacing) / 2
    xs = x0 + np.arange(cols) * spacing
    ys = y0 + np.arange(rows) * spacing
    gx, gy = np.meshgrid(xs, ys)
    g = np.stack([gx.ravel(), gy.ravel()], axis=1)[:n]
    return np.clip(g, [0, 0], [fw, fh])


def layout_uniform(n_user: int, fw: float, fh: float,
                   seed: int = LAYOUT_SEED):
    """均匀随机撒点（Cheng TIM 2024 §V-A：GUs randomly distributed）。"""
    rs = np.random.default_rng(seed)
    pos = rs.uniform(0.0, 1.0, (n_user, 2)) * [fw, fh]
    return pos, [], []


def layout_poi_grid(nx: int, ny: int, fw: float, fh: float):
    """规则网格中心（Chen T-ITS 2024 §V-A：10×10 单元、边长 100 m、
    每个单元中心即一个 PoI）。"""
    xs = (np.arange(nx) + 0.5) * (fw / nx)
    ys = (np.arange(ny) + 0.5) * (fh / ny)
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    return np.column_stack([gx.ravel(), gy.ravel()]), [], []


def layout_sector(ncx: int, ncy: int, per_cell: int, fw: float, fh: float):
    """每小区扇区中心（Song TWC 2025 §II：每个 cell 划分成 U 个扇区，
    GU 位于扇区中心）。per_cell=4 → 每小区 2×2 四个扇区中心。"""
    sx = max(1, int(round(math.sqrt(per_cell))))
    sy = max(1, per_cell // sx)
    cw, ch = fw / ncx, fh / ncy
    pos = []
    for i in range(ncx):
        for j in range(ncy):
            for a in range(sx):
                for b in range(sy):
                    pos.append([(i + (a + 0.5) / sx) * cw,
                                (j + (b + 0.5) / sy) * ch])
    return np.asarray(pos, dtype=np.float64), [], []


def layout_ccc(n_user: int, fw: float, fh: float,
               sigma: float = 30.0, seed: int = LAYOUT_SEED):
    """从控制中心出发（Rizvi TMLCN 2025 §V：Users start from the CCC）。

    论文未给 CCC 坐标，取任务区几何中心；30 m 高斯散布避免重合
    （社会力/直线移动要求互异点）。
    """
    rs = np.random.default_rng(seed)
    c = np.array([fw * 0.5, fh * 0.5])
    pos = c + rs.normal(0.0, sigma, (n_user, 2))
    return np.clip(pos, [0, 0], [fw, fh]), [], []


def scale_spec(n_uav: int) -> dict:
    k = n_uav / N_UAV
    s = math.sqrt(k)
    return {
        "field_w": FIELD_W * s,
        "field_h": FIELD_H * s,
        "n_user": round(N_USER * k),
        "n_hotspots": max(1, round(3 * k)),
        "hotspot_sigma": HOTSPOT_SIGMA * s,
        "b_total": B_TOTAL * k,
    }


def uav_grid(n_uav: int) -> np.ndarray:
    """初始机位网格（z=UAV_ALT 由物理层持有）。"""
    if n_uav == N_UAV:
        cols, rows = GRID_COLS, GRID_ROWS
    else:
        cols, rows = grid_layout(n_uav)
    fw, fh = (FIELD_W, FIELD_H) if n_uav == N_UAV else (
        scale_spec(n_uav)["field_w"], scale_spec(n_uav)["field_h"])
    x0 = (fw - (cols - 1) * GRID_SPACING) / 2
    y0 = (fh - (rows - 1) * GRID_SPACING) / 2
    xs = x0 + np.arange(cols) * GRID_SPACING
    ys = y0 + np.arange(rows) * GRID_SPACING
    gx, gy = np.meshgrid(xs, ys)
    return np.stack([gx.ravel(), gy.ravel()], axis=1)[:n_uav]


def user_layout(n_uav: int) -> tuple[np.ndarray, list[np.ndarray], list[list[float]]]:
    """用户布点（固定 LAYOUT_SEED；标准密度档）。

    返回 (user_pos(M,2), hot_users 列表, 热点中心列表)。
    """
    rs = np.random.default_rng(LAYOUT_SEED)
    if n_uav == N_UAV:
        fw, fh = FIELD_W, FIELD_H
        hotspots = [list(c) for c in HOTSPOTS]
        hs, n_user = HOTSPOT_SIGMA, N_USER
        tot_frac = 3 * HOTSPOT_FRAC
    else:
        spec = scale_spec(n_uav)
        fw, fh = spec["field_w"], spec["field_h"]
        hs, n_user = spec["hotspot_sigma"], spec["n_user"]
        hotspots = [list(c) for c in rs.uniform(
            [0.1 * fw, 0.1 * fh], [0.9 * fw, 0.9 * fh],
            size=(spec["n_hotspots"], 2))]
        tot_frac = 3 * HOTSPOT_FRAC
    n_hot = int(n_user * tot_frac / len(hotspots))
    pos = [rs.normal(c, hs, size=(n_hot, 2)) for c in hotspots]
    pos.append(rs.uniform(0, 1, size=(n_user - len(hotspots) * n_hot, 2))
               * [fw, fh])
    user_pos = np.clip(np.concatenate(pos), [0, 0], [fw, fh])
    # 边界 clip 会把高斯尾压到同一点——重合对使社会力 0/0=nan
    # （n200 实测 3 对）。确定性去重：同位置第 k 个加 k×1cm 偏移。
    _, _, inverse, counts = np.unique(
        user_pos, axis=0, return_index=True,
        return_inverse=True, return_counts=True)
    if (counts > 1).any():
        seen = {}
        for j in range(len(user_pos)):
            key = (round(float(user_pos[j, 0]), 9),
                   round(float(user_pos[j, 1]), 9))
            seen[key] = seen.get(key, 0) + 1
            k = seen[key] - 1
            if k > 0:                     # 同位置第 k 个：加 k×1cm 确定偏移
                user_pos[j] = user_pos[j] + k * 0.01 * np.array([1.0, 0.5])
        user_pos = np.clip(user_pos, [0, 0], [fw, fh])
    hot_users = [np.arange(k * n_hot, (k + 1) * n_hot)
                 for k in range(len(hotspots))]
    return user_pos, hot_users, hotspots


def table_path(n_uav: int, scenario: str, fc_hz: float,
               tag: str | None = None) -> Path:
    """信道表路径（规模_场景_载频）；.d 目录（npy+mmap 共享页）优先。

    tag: 自定义场景的表名标识（2026-10-30，基线原生场景复刻用）。
    """
    stem = TABLE_DIR / f"chan_{tag or n_uav}_{scenario}_{fc_hz/1e9:g}ghz"
    d = Path(f"{stem}.d")
    return d if d.is_dir() else Path(f"{stem}.npz")


def table_meta(n_uav: int, scenario: str, fc_hz: float, field_w: float,
               field_h: float, n_user: int) -> dict:
    return {"n_uav": n_uav, "scenario": scenario, "fc_hz": fc_hz,
            "field_w": field_w, "field_h": field_h, "n_user": n_user,
            "res": CHAN_RES, "alt": UAV_ALT, "shadow_seed": CHAN_SHADOW_SEED,
            "layout_seed": LAYOUT_SEED, "bw_share": BW_SHARE,
            "spec": "gain_db(x,y,m) dB | los(x,y,m) | x_edges | y_edges | meta"}


# ---------------------- 基线原生场景（2026-10-30 用户令，2026-10-30 重订） ------
# 每个基线一条原生场景，参数逐项取自论文表格/正文，一个数不自造。
# 共同口径：信道沿用项目统一的 3GPP 口径（04_基线.md §二"同一信道"）；
#           场地、机数、用户数、带宽、移动能力按各自原文。
# 字段：
#   n_uav        论文的机队规模
#   n_user       论文的地面用户/PoI 数
#   field_w/h    论文的任务区边长（米）
#   b_total      总带宽 MHz = 每信道带宽 × 机数
#   user_motion  static=用户静止（论文无移动模型）；linear=直线移动
#   dxy_max      每环境步位移上限（米/0.1 秒）＝论文最大速度 × 0.1
#   table_tag    信道表标识；analytic=True 表示场地过大不建表，走解析径
BASELINE_SCENES = {
    # Cheng et al., IEEE TIM 2024, 5503613 (LTCC-UDUA)
    # Table I: B=20 MHz, fc=2 GHz；§V-A/Fig.7: 7 架、50 GU 随机分布于 1000×1000 m、
    # 高度 100 m、最大速度 20 m/s；GU 无移动模型（全文仅"randomly distributed"）。
    "cheng7": {"n_uav": 7, "n_user": 50,
               "field_w": 1000.0, "field_h": 1000.0,
               "b_total": 20.0 * 7, "user_motion": "static",
               "dxy_max": 2.0, "table_tag": "c7c50", "layout": "uniform",
               "alt": 100.0, "demand_scale": 48.6968},
    # Song et al., IEEE TWC 2025, 24(7):5620-5634 (ME-MFDQN)
    # §V: 19×19=361 架、密度 1/km²、每机服务 4 GU（共 1444）、B=1 MHz、高度 100 m。
    # 用户位置静态，需求 x_k(t) 时变（论文贡献即"时变需求"）。
    "song361": {"n_uav": 361, "n_user": 1444,
                "field_w": 19000.0, "field_h": 19000.0,
                "b_total": 1.0 * 361, "user_motion": "static",
                "dxy_max": 0.5, "table_tag": None, "analytic": True,
                "layout": "sector", "layout_args": {"ncx": 19, "ncy": 19,
                                                    "per_cell": 4},
                "alt": 100.0, "demand_scale": 0.000220673},
    # Rizvi et al., IEEE TMLCN 2025, vol.3 (masked SDQN)
    # Table 3: U=3 架、K=6 用户、B=15 kHz、Vmax_user=10 m/s、V_uav=5 m/s、
    # 服务区 0~1000 m、Pmax=29 dBm、UAV 高度 20~150 m。
    # 用户从 Crisis Control Center 出发沿最短路移动 → linear 档。
    "rizvi3": {"n_uav": 3, "n_user": 6,
               "field_w": 1000.0, "field_h": 1000.0,
               "b_total": 0.015 * 3, "user_motion": "linear",
               "dxy_max": 0.5, "user_speed": 10.0, "layout": "ccc",
               "alt": 100.0, "table_tag": "rizvi3", "demand_scale": 0.121881},
    # Chen et al., IEEE T-ITS 2024, 25(8) (T-MARL)
    # §V-A: 10×10 单元、单元边长 100 m（1000×1000 m）、每单元中心一个 PoI 共 100 个、
    # 6 架、覆盖半径 250 m、机间通信 500 m、每时隙最大 25 m/s。PoI 固定。
    # 论文未给带宽 → 沿用项目密度口径 100 MHz × 6/12。
    "chen6": {"n_uav": 6, "n_user": 100,
              "field_w": 1000.0, "field_h": 1000.0,
              "b_total": round(100.0 * 6 / 12, 1), "user_motion": "static",
              "dxy_max": 2.5, "table_tag": "chen6", "layout": "grid",
              "layout_args": {"nx": 10, "ny": 10},
              "neigh_dist": 500.0, "demand_scale": 7.77365},
    # Shen et al., IEEE TVT 2024, 73(6):9205-9210 (MULB)
    # §IV: 2×2 km² 开阔区、K=20 UE 随机散布、M=3 UAV、H=100 m、
    # Vmax=50 m/s、Dmin=100 m、Pm=0.1 W、σ²=-110 dBm、β0=-50 dB、
    # B=10 MHz、T=40 s（N=20 时隙 × dt=2 s）。用户静止（论文无移动模型）。
    "shen3": {"n_uav": 3, "n_user": 20,
              "field_w": 2000.0, "field_h": 2000.0,
              "b_total": 10.0, "user_motion": "static",
              "dxy_max": 5.0, "table_tag": None, "analytic": True,
              "layout": "uniform", "alt": 100.0, "demand_scale": 2.12875},
    # Jiang et al., IEEE IoT-J 2025, 12(24):53538-53552 (PSO-LBA)
    # Table II: 水平区 500×500 m、高度 120–300 m、B^A=180 kHz/用户、
    # B^B=100 MHz、N^max=50 用户/ABS、p^A=30 dBm、f^A=2 GHz、
    # N0=-174 dBm/Hz、{a,b,ηLoS,ηNLoS}={9.61,0.16,1 dB,20 dB}。
    # §IV-C 正文 "across the five ABSs" → |A|=5；Fig.3 收敛场景 |U|=250。
    # §II "users are static" → 用户静止。
    # 高度取范围中点 200 m 并冻结（项目 env 高度固定，PSO 高度维冻结）。
    # b_total = 0.18 MHz × 50 用户 × 5 机 = 45 MHz（每 ABS 满用户容量）。
    "jiang5": {"n_uav": 5, "n_user": 250,
               "field_w": 500.0, "field_h": 500.0,
               "b_total": 45.0, "user_motion": "static",
               "dxy_max": 0.5, "table_tag": None, "analytic": True,
               "layout": "uniform", "alt": 200.0, "demand_scale": 3.27812},
}


def baseline_scene(name: str | None) -> dict | None:
    return BASELINE_SCENES.get(name) if name else None


import ctypes
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

logger = logging.getLogger("code.env")

SHADOW_T = 100      # 阴影刷新周期（步）：0.5 米/步 × 100 = 50 米相干距离


LN10_10 = np.log(10.0) / 10.0        # dB→线性比值导数因子：d(10^(x/10))/dx


def bh_capacity_mbps(pos_xy: np.ndarray, bs_xy: np.ndarray) -> np.ndarray:
    """回传容量（Mbps）：Jiang 2025 §II-A，28 GHz 毫米波，TBS → 无人机。

    模型逐式照抄：
        L^B = 61.4 + 20·log10(d)          28 GHz 毫米波路损（dB）
        γ^B = p^B · g^B / I^B             g^B = 10^(−L^B/10)
        C^B = B^B · log2(1 + γ^B)
    参数全部来自 config 的 Table II 值：B^B=100 MHz、p^B=43 dBm、N0=−174 dBm/Hz。
    `pos_xy` 与 `bs_xy` 必须是**同一坐标系**（块拼接场景用全局坐标）。
    小尺度 Rayleigh 衰落取统计均值 1（该文按大时间尺度建模，快衰落被平均掉）。
    """
    dx = pos_xy[:, 0] - bs_xy[0]
    dy = pos_xy[:, 1] - bs_xy[1]
    d = np.sqrt(dx * dx + dy * dy + float(UAV_ALT) ** 2)
    l_db = 61.4 + 20.0 * np.log10(np.maximum(d, 1.0))
    g = 10.0 ** (-l_db / 10.0)
    p_bs = 10.0 ** (P_BS_DBM / 10.0)                       # mW
    i_bh = (10.0 ** (N0_PSD_DBM_HZ / 10.0)) * BH_BW_HZ     # mW（N0 乘带宽）
    return BH_BW_HZ * np.log2(1.0 + p_bs * g / i_bh) / 1e6  # bps → Mbps


def _sub_grid(lay: int, n: int) -> np.ndarray:
    """从 lay 机标准网格里取 n 个机位：离场地中心最近的 n 个（确定性）。

    场地与用户密度按 lay 定（信道表是纯查询表，与机数无关），机位从同一
    网格里取——于是「同一片天，少放几架」。

    取中心簇而不是最远点：网格间距 150 m < NEIGH_DIST 250 m，中心簇天然
    连通（n=2 得 1 条边，n=4 得 2×2 块 4 条边）。势场只在有边的图上存在，
    零边集无从验证——最远点采样会把机位摊到四角、一条边都不剩。
    """
    g = uav_grid(lay)
    if n >= len(g):
        return uav_grid(n) if n > len(g) else g.copy()
    fw, fh = 600.0, 450.0
    if lay != N_UAV:
        s = scale_spec(lay)
        fw, fh = s["field_w"], s["field_h"]
    c = np.array([fw / 2.0, fh / 2.0])
    d = np.linalg.norm(g - c[None, :], axis=1)
    pick = np.sort(np.argsort(d, kind="mergesort")[:n])
    return g[pick].copy()


@dataclass
class EnvStats:
    var_u: list = field(default_factory=list)
    u_mean: list = field(default_factory=list)
    sum_t: list = field(default_factory=list)
    n_over: list = field(default_factory=list)
    e_fly: list = field(default_factory=list)
    # 陷阱 76：深坑机 R_avail≈0 → u 爆（var_u=e12+ 单机淹没全队）。
    # 截尾方差 + 死机数独立口径：var_u 原值保留对照，汇总主指标用 clip 版。
    var_u_clip: list = field(default_factory=list)
    n_dead: list = field(default_factory=list)

    def windows(self) -> dict:
        a = lambda xs, lo, hi: float(np.mean(xs[lo:hi]))     # noqa: E731
        return {"var_u_pre": a(self.var_u, 700, 1000),
                "var_u_post": a(self.var_u, 1700, 2000),
                "u_mean_pre": a(self.u_mean, 700, 1000),
                "u_mean_post": a(self.u_mean, 1700, 2000),
                "sum_t_pre": a(self.sum_t, 700, 1000),
                "sum_t_post": a(self.sum_t, 1700, 2000),
                "n_over_pre": a(self.n_over, 700, 1000),
                "n_over_post": a(self.n_over, 1700, 2000),
                "var_u_clip_pre": a(self.var_u_clip, 700, 1000),
                "var_u_clip_post": a(self.var_u_clip, 1700, 2000),
                "n_dead_pre": a(self.n_dead, 700, 1000),
                "n_dead_post": a(self.n_dead, 1700, 2000)}


class UAVEnvV3:
    """N 机（MuJoCo 真值位置）×M 用户（固定布局种子）×五元动作统一场。"""

    def __init__(self, tau_u: float = TAU_U_DEFAULT, seed: int = 0,
                 scene: str = "flow",
                 n_uav: int = N_UAV, t_comm: int = T_COMM_DEFAULT,
                 bw_share: str = BW_SHARE, use_physics: bool = True,
                 p_ref_dbm: float = P_REF_DBM, n0_dbm: float = N0_DBM,
                 demand_scale: float = 1.0,
                 table=None, scenario: str | None = None,
                 fc_hz: float | None = None,
                 layout_n: int | None = None,
                 override_spec: dict | None = None):
        """scene: flow | burst
        table: 信道表 npz 路径或预载 dict（gain_db/x_edges/y_edges[/meta]）
        layout_n: 场地/用户布局所按的机数规模（默认=n_uav）。信道表是
                  (格点 × 用户) 的纯查询表，机数与它无关，故可用 n12 的
                  场地与 600 用户配 1/2/4 架无人机——密度口径不变。
        """
        self.tau_u, self.seed = float(tau_u), int(seed)
        self.scene = str(scene)
        self.n_uav = int(n_uav)
        self.t_comm = max(1, int(t_comm))
        self.bw_share = str(bw_share)
        self.p_ref_dbm, self.n0_dbm = float(p_ref_dbm), float(n0_dbm)
        self.demand_scale = float(demand_scale)

        # ---- 场地/机位/用户（布局种子固定；训练种子只驱动策略随机性） ----
        if self.scene not in ("flow", "burst", "static"):
            raise ValueError(
                f"未知场景 {self.scene}（flow/burst/static 三档）")
        lay = int(layout_n) if layout_n else self.n_uav
        lay_spec = None
        # 用户运动档（2026-10-30 重订）：flow/burst=社会力人群；static=静止；
        # linear=直线匀速（Rizvi 原场景"用户 10 m/s 从控制中心出发"）
        self.user_motion = self.scene
        self._user_speed = 0.0
        self.alt = float(UAV_ALT)              # 飞行高度（基线场景按论文覆盖）
        self.neigh_dist = float(NEIGH_DIST)    # 机间邻接半径
        if override_spec:                     # 基线原生场景（2026-10-30）
            # 2026-10-31 自查：config 的 dxy_max 必须与 PF_DXY_MAX 一致。
            # 动作解码（decode81/decode_ddp/decode_all）读模块级 DXY_MAX，
            # 两处不一致时基线移动能力会被静默改写（启动脚本 5.0 vs 配置 0.5）。
            _dxy_spec = float(override_spec.get("dxy_max", DXY_MAX))
            assert abs(_dxy_spec - DXY_MAX) < 1e-9, (
                f"场景 dxy_max={_dxy_spec} 与 PF_DXY_MAX={DXY_MAX} 不一致"
                f"（{override_spec.get('table_tag') or override_spec}）")
            self.user_motion = str(override_spec.get("user_motion", "static"))
            self._user_speed = float(override_spec.get("user_speed", 0.0))
            self.alt = float(override_spec.get("alt", UAV_ALT))
            self.neigh_dist = float(override_spec.get("neigh_dist", NEIGH_DIST))
            self.field_w = float(override_spec["field_w"])
            self.field_h = float(override_spec["field_h"])
            self.b_total = float(override_spec["b_total"])
            self.b_bar = self.b_total / self.n_uav
            self.bs_xy = np.array([float(BS_POS[0]), float(BS_POS[1])])
            self.relay_enable = bool(RELAY_ENABLE)
            self.init_xy = uav_grid_custom(self.n_uav, self.field_w,
                                           self.field_h)
            # 用户布局按各论文原文（2026-10-31 核对）：
            #   uniform = 均匀随机（Cheng TIM 2024）
            #   grid    = 规则网格中心（Chen T-ITS 2024，10×10 单元中心即 PoI）
            #   sector  = 每小区扇区中心（Song TWC 2025，每 cell 4 个扇区）
            #   ccc     = 控制中心出发（Rizvi TMLCN 2025）
            #   custom  = 项目自用的热点模板（仅非基线场景）
            _lay = str(override_spec.get("layout", "custom"))
            _la = override_spec.get("layout_args") or {}
            _sd = int(override_spec.get("layout_seed", LAYOUT_SEED))
            if _lay == "uniform":
                self.user_pos, self.hot_users, self.hotspots = layout_uniform(
                    int(override_spec["n_user"]), self.field_w, self.field_h,
                    seed=_sd)
            elif _lay == "grid":
                self.user_pos, self.hot_users, self.hotspots = layout_poi_grid(
                    int(_la.get("nx", 10)), int(_la.get("ny", 10)),
                    self.field_w, self.field_h)
            elif _lay == "sector":
                self.user_pos, self.hot_users, self.hotspots = layout_sector(
                    int(_la.get("ncx", 19)), int(_la.get("ncy", 19)),
                    int(_la.get("per_cell", 4)), self.field_w, self.field_h)
            elif _lay == "ccc":
                self.user_pos, self.hot_users, self.hotspots = layout_ccc(
                    int(override_spec["n_user"]), self.field_w, self.field_h,
                    seed=_sd)
            else:
                self.user_pos, self.hot_users, self.hotspots = layout_custom(
                    int(override_spec["n_user"]), self.field_w, self.field_h,
                    int(override_spec.get("n_hotspots", 3)),
                    float(override_spec.get("hotspot_sigma", 60.0)),
                    seed=_sd)
            self.M = len(self.user_pos)
        else:
            lay_spec = scale_spec(lay) if lay != N_UAV else None
            self.field_w = lay_spec["field_w"] if lay_spec else 600.0
            self.field_h = lay_spec["field_h"] if lay_spec else 450.0
            self.b_total = lay_spec["b_total"] if lay_spec else B_TOTAL
            self.b_bar = self.b_total / self.n_uav
            # 宿主基站 TBS（转发中继）：场地原点地面，参数全部来自 Jiang 2025 Table II。
            # 接入段沿用项目自己的 3GPP 信道，不动；回传段是新增的第二段链路。
            self.bs_xy = np.array([float(BS_POS[0]), float(BS_POS[1])])
            self.relay_enable = bool(RELAY_ENABLE)
            self.init_xy = _sub_grid(lay, self.n_uav)
            self.user_pos, self.hot_users, self.hotspots = user_layout(lay)
            self.M = len(self.user_pos)
        self.user_pos0 = self.user_pos.copy()      # 初始布局（回合边界重置用）
        self._hot_sigma = lay_spec["hotspot_sigma"] if lay_spec else 60.0

        rs = np.random.default_rng(self.seed + 7000)          # 需求流（训练种子）
        self.dem_base = np.clip(np.abs(rs.normal(
            DEMAND_MEAN, DEMAND_STD, self.M)), 0.1, 1.0) * self.demand_scale

        # ---- 人群引擎（flow/burst 用社会力；static/linear 不建引擎） ----
        self._crowd_lib = None
        self._crowd_pos = np.zeros((self.M, 2), dtype=np.float64)
        self._crowd_h = None
        self._user_dir = None
        if self.user_motion in ("flow", "burst"):
            self._crowd_lib = self._load_crowd()
            self._crowd_setup()

        # ---- 信道表（analytic 场景不建表：场地过大，走解析径） ----
        self._analytic = (bool(override_spec.get("analytic"))
                          if override_spec else False)
        if self._analytic:
            self.tab = None
            self.tab_meta = {"analytic": True}
            self._gx0 = self._gy0 = 0.0
            self._nx = self._ny = 0
        else:
            if table is None:             # 表按布局规模取（lay），不是按机数：
                _tag = override_spec.get("table_tag") if override_spec else None
                table = table_path(lay, scenario or CHAN_SCENARIO,
                                   fc_hz or CHAN_FC, tag=_tag)  # 表是纯查询表，与机数无关
            if isinstance(table, (str, Path)):
                p = Path(table)
                if p.is_dir():      # SSD 卸载模式：npz 预解压 npy + mmap 按需页加载
                    xe = np.load(p / "x_edges.npy")
                    ye = np.load(p / "y_edges.npy")
                    gm = np.load(p / "gain_db.npy", mmap_mode="r")
                    self.tab = {"gain_db": gm.reshape(
                        len(xe), len(ye), -1),             # (nx,ny,M) mmap view
                        "x_edges": xe, "y_edges": ye}
                    mj = p / "meta.json"
                    self.tab_meta = json.loads(
                        mj.read_text("utf-8")) if mj.exists() else {}
                else:
                    z = np.load(table, allow_pickle=False)
                    xe, ye = z["x_edges"], z["y_edges"]
                    self.tab = {"gain_db": z["gain_db"].reshape(
                        len(xe), len(ye), -1),             # (nx,ny,M)
                        "x_edges": xe, "y_edges": ye}
                    self.tab_meta = json.loads(str(z["meta"][0])) if "meta" in z else {}
            else:
                self.tab = {k: np.asarray(table[k]) for k in
                            ("gain_db", "x_edges", "y_edges")}
                self.tab_meta = {}
            self._gx0 = float(self.tab["x_edges"][0])
            self._gy0 = float(self.tab["y_edges"][0])
            self._nx, self._ny = len(self.tab["x_edges"]), len(self.tab["y_edges"])

        # ---- 物理层 ----
        self.use_physics = bool(use_physics)
        self._phys = None
        if self.use_physics:
            from mujoco_adapter import PhysicsAdapter
            self._phys = PhysicsAdapter(self.n_uav, self.init_xy,
                                        self.field_w, self.field_h,
                                        seed=self.seed)
            self.pos = self._phys.positions()
        else:
            self.pos = self.init_xy.copy()
        d_uu = np.linalg.norm(self.init_xy[:, None] - self.init_xy[None], axis=2)
        self._adj0 = (d_uu <= self.neigh_dist) & (d_uu > 0)
        self.adj = self._adj0.copy()
        self.neighbors = [np.where(self.adj[i])[0] for i in range(self.n_uav)]
        self.edges = np.array([(i, j) for i in range(self.n_uav)
                               for j in self.neighbors[i] if i < j])

        # ---- 动态状态 ----
        self.b = np.full(self.n_uav, self.b_bar)
        self.pi = np.zeros(self.n_uav)
        self.p = np.ones(self.n_uav)
        self.dem = self.dem_base.copy()
        self.assoc = None
        self.R_avail = np.full(self.n_uav, 1.0)   # attn 首步占位（κ 项→0）
        self.rho = np.zeros(self.n_uav)
        self.t = 0
        self.stats = EnvStats()
        self._obs_cache = None
        self._edge_cache = None
        self._u_stale = None
        # 动态场景走 numpy 解析链路（C 表核只支持静态用户烘焙表）；
        # _dyn 判定提前（scene 在构造参数即已知）。
        # 2026-10-30 重订：analytic 场景（场地过大不建表）同样走解析径。
        self._dyn = (self.scene in ("flow", "burst")) or self._analytic
        # 2026-10-29 用户令：解析径补上 3GPP 第三项——链路级随机阴影。
        # 2026-10-30 用户裁定：σ **随机化**——每回合从 [0, 13.2] 均匀抽一个，
        # 训练见过全谱；评估用 PF_SHADOW_DB=<数值> 固定住以便比对。
        # 阴影本身每 SHADOW_T 步刷新（50 米相干距离／0.5 米每步）不变。
        self._shadow_sigma = SHADOW_SIGMA_FIXED
        self._shadow_rt = None
        self._shadow_t = -1
        self._ep_n = 0
        if self._analytic:
            self._fit_pl_analytic()   # 3GPP 解析式直接回归 C0/γ（无表）
        elif self._dyn:
            self._fit_pl_dyn()        # 解析径参数拟合（权威表 → γ/逐用户截距）
        if self._dyn:
            self._fast_init_dyn()     # 唯一 C 核：fastlink3（仅动态/解析径场景）
        else:
            self._fast3 = None        # 静态场景：numpy 表链路（v1 表核已废弃）
        self.reset()

    # ============================ 唯一 C 加速核（fastlink3） ============================

    # ============================ 动态信道解析径（移动人群，2026-10-27） ============================
    def _fast_init_dyn(self):
        """动态解析径 C 核（fastlink3.so；缺库大声告警回退 numpy，约慢 10 倍）。"""
        self._fast3 = None
        so = Path(__file__).parent / (
            "fastlink3.dll" if os.name == "nt" else "fastlink3.so")
        if not so.exists():
            sys.stderr.write(
                "!!! FASTLINK3 MISSING —— 动态场景回退 numpy 解析链路（慢约 10 倍）\n")
            sys.stderr.flush()
            return
        lib = ctypes.CDLL(str(so))
        dp = ctypes.POINTER(ctypes.c_double)
        i_, d_ = ctypes.c_int, ctypes.c_double
        lib.fast_link_dyn.restype = None
        lib.fast_link_dyn.argtypes = [
            dp, dp, dp, dp, dp, dp, dp, dp, dp,          # 1-9  pos..c0,shadow
            d_, i_, i_,                                   # 10-12 gamma,M,N
            d_, d_, d_, d_, d_, i_,                       # 13-18 物理常数
            dp, dp, dp, dp, dp, dp, dp, dp, dp, dp,       # 19-28 输出1
            dp, dp, dp, dp, dp, dp, dp, dp, dp, dp]       # 29-38 输出2
        N, M = self.n_uav, self.M
        nm = N * M

        def buf():
            a = np.ctypeslib.as_array((ctypes.c_double * nm)())
            a.fill(0.0)
            return a

        self._fb = {"G_db": buf(), "sx": buf(), "sy": buf(), "g": buf(),
                    "sig": buf(), "denom": buf(), "sinr": buf(), "r": buf(),
                    "bsh": buf(), "R": buf(), "attn": buf(), "x_star": buf(),
                    "P": np.zeros(N), "agg": np.zeros(M),
                    "R_avail": np.zeros(N), "rho": np.zeros(N),
                    "u": np.zeros(N), "T": np.zeros(N)}
        self._fast3 = lib

    def _refresh_link_fast_dyn(self):
        lib = self._fast3
        f = self._fb
        N, M = self.n_uav, self.M
        dp = ctypes.POINTER(ctypes.c_double)
        p_ = lambda a: np.ascontiguousarray(a).ctypes.data_as(dp)
        sh = self._shadow_matrix()
        sh_ptr = p_(sh) if sh is not None else None
        lib.fast_link_dyn(
            p_(self.pos), p_(self.b), p_(self.pi), p_(self.p),
            p_(self.dem), p_(self.assoc),
            p_(self.user_pos), p_(self._pl_c0), sh_ptr,
            float(self._pl_gamma), M, N,
            self._n0_lin, 10.0 ** (self.p_ref_dbm / 10.0),
            float(ALPHA_ASSOC), float(KAPPA), float(self.tau_u),
            int(self.bw_share == "assoc_equal"),
            p_(f["G_db"]), p_(f["sx"]), p_(f["sy"]), p_(f["g"]),
            p_(f["P"]), p_(f["sig"]), p_(f["denom"]),
            p_(f["sinr"]), p_(f["r"]), p_(f["bsh"]), p_(f["R"]),
            p_(f["agg"]), p_(f["attn"]), p_(f["x_star"]),
            p_(f["R_avail"]), p_(f["rho"]), p_(f["u"]),
            p_(f["T"]),
            p_(self.R_avail), p_(self.rho))
        shp = (N, M)
        self._G_db = f["G_db"].reshape(shp)
        self._sx, self._sy = f["sx"].reshape(shp), f["sy"].reshape(shp)
        self._g, self._P = f["g"].reshape(shp), f["P"]
        self._sig, self._denom = f["sig"].reshape(shp), f["denom"].reshape(shp)
        self._sinr, self._r = f["sinr"].reshape(shp), f["r"].reshape(shp)
        self._bsh, self._R = f["bsh"].reshape(shp), f["R"].reshape(shp)
        self._x_star, self._attn = f["x_star"].reshape(shp), f["attn"].reshape(shp)
        self.R_avail, self.rho = f["R_avail"], f["rho"]
        self.u, self.T = f["u"], f["T"]

    def _fit_pl_dyn(self) -> None:
        """解析增益参数自权威表拟合（n800 _fit_pl 迁移+逐用户截距扩展）：
        G_db(i,u) = C0[u] - 10γ·log10(d_iu + 50)。γ 全局（逐用户 OLS 取
        中位），C0[u] 逐用户截距（保留用户环境特异性）后插值到全体 M。"""
        g = np.asarray(self.tab["gain_db"][::4, ::4, ::102], dtype=np.float64)
        xs = (np.arange(g.shape[0]) * 4 + 2) * 10.0
        ys = (np.arange(g.shape[1]) * 4 + 2) * 10.0
        X, Y = np.meshgrid(xs, ys, indexing="ij")
        up = self.user_pos0[:: max(1, self.M // g.shape[2])][: g.shape[2]]
        d = np.sqrt((X.ravel()[:, None] - up[None, :, 0]) ** 2
                    + (Y.ravel()[:, None] - up[None, :, 1]) ** 2) + 50.0
        Ym = g.reshape(-1, g.shape[2])
        x = -10.0 * np.log10(d)
        xc, yc = x - x.mean(0), Ym - Ym.mean(0)
        gamma_u = (xc * yc).sum(0) / np.maximum((xc * xc).sum(0), 1e-9)
        self._pl_gamma = float(np.median(gamma_u))
        c0 = Ym.mean(0) - self._pl_gamma * x.mean(0)
        if len(c0) < self.M:
            c0 = np.interp(np.linspace(0.0, 1.0, self.M),
                           np.linspace(0.0, 1.0, len(c0)), c0)
        self._pl_c0 = c0
        # 宏站/微站切换（2026-10-29 用户令，开关在 config.PF_CHANNEL）：
        #   table / umi = 现行表档（微站）
        #   uma         = 表档 C0 + 解析斜率比（宏站），在 d_ref=100 m 处锚定，
        #                 保证两组只差斜率、不差工作点
        if PF_CHANNEL == "uma":
            import pf_channel
            g_umi = pf_channel.fit_c0_gamma(
                self.user_pos0, self.field_w, self.field_h,
                model="umi", fc_ghz=CHAN_FC / 1e9, seed=self.seed)[1]
            g_uma = pf_channel.fit_c0_gamma(
                self.user_pos0, self.field_w, self.field_h,
                model="uma", fc_ghz=CHAN_FC / 1e9, seed=self.seed)[1]
            new_gamma = float(self._pl_gamma * (g_uma / g_umi))
            self._pl_c0 = (self._pl_c0 + 10.0 * (new_gamma - self._pl_gamma)
                           * np.log10(150.0))
            self._pl_gamma = new_gamma

    def _fit_pl_analytic(self) -> None:
        """解析径参数（analytic 场景：场地过大不建表，直接回归 3GPP 36.777）。

        与 _fit_pl_dyn 同构（逐用户 OLS：C0[u] 截距 + γ 中位斜率），
        参考机位由 pf_channel.fit_c0_gamma 在场地内按 step 铺开。
        19 km 场地按 10 m 分辨率建表是 1901×1901×1444 项，不建表。
        """
        import pf_channel
        span = max(self.field_w, self.field_h)
        c0, gamma = pf_channel.fit_c0_gamma(
            self.user_pos0, self.field_w, self.field_h,
            model=("uma" if span > 5000.0 else "umi"),
            fc_ghz=CHAN_FC / 1e9, z=self.alt, step=span / 40.0,
            seed=self.seed)
        self._pl_c0 = np.asarray(c0, dtype=np.float64)
        self._pl_gamma = float(gamma)

    def _shadow_matrix(self):
        """链路级随机阴影 (N,M)，每 SHADOW_T 步刷新；σ=0 时返回 None。

        与 fastlink3 的 shadow 参数同源：numpy 解析径与 C 核读同一张矩阵，
        两条路径的阴影逐位一致。
        """
        if self._shadow_sigma <= 0.0:
            return None
        shp = (self.n_uav, self.M)
        tkey = self.t // SHADOW_T
        if (self._shadow_rt is None or self._shadow_rt.shape != shp
                or self._shadow_t != tkey):
            rng = np.random.default_rng(int(self.seed) * 100003 + tkey)
            self._shadow_rt = rng.normal(0.0, self._shadow_sigma, shp)
            self._shadow_t = tkey
        return self._shadow_rt

    def _interp_dyn(self, pos: np.ndarray):
        """解析增益（用户位置实时参与；对机位置解析斜率同出）。"""
        dx = pos[:, 0:1] - self.user_pos[None, :, 0]
        dy = pos[:, 1:2] - self.user_pos[None, :, 1]
        D = np.sqrt(dx * dx + dy * dy) + 50.0
        G_db = self._pl_c0[None, :] - 10.0 * self._pl_gamma * np.log10(D)
        sh = self._shadow_matrix()
        if sh is not None:
            G_db = G_db + sh
        k = -10.0 * self._pl_gamma / (np.log(10.0) * D * D)
        return G_db, k * dx, k * dy

    # ============================ 信道表插值 ============================
    def _interp(self, pos: np.ndarray):
        """dB 域双线性插值（分片线性→位置解析斜率天然可得）。

        返回 (gain_db (N,M), slope_x (N,M), slope_y (N,M))，dB/m。
        """
        g = self.tab["gain_db"]
        x = np.clip(pos[:, 0], self._gx0,
                    self._gx0 + (self._nx - 1) * CHAN_RES)
        y = np.clip(pos[:, 1], self._gy0,
                    self._gy0 + (self._ny - 1) * CHAN_RES)
        ix = np.clip((x - self._gx0) / CHAN_RES, 0, self._nx - 2).astype(int)
        iy = np.clip((y - self._gy0) / CHAN_RES, 0, self._ny - 2).astype(int)
        fx = (x - self._gx0) / CHAN_RES - ix
        fy = (y - self._gy0) / CHAN_RES - iy
        ixp, iyp = ix + 1, iy + 1
        c00 = g[ix, iy]                                # (N,M)
        c10 = g[ixp, iy]
        c01 = g[ix, iyp]
        c11 = g[ixp, iyp]
        w00 = (1 - fx) * (1 - fy)
        w10 = fx * (1 - fy)
        w01 = (1 - fx) * fy
        w11 = fx * fy
        gain = w00[:, None] * c00 + w10[:, None] * c10 \
            + w01[:, None] * c01 + w11[:, None] * c11
        sx = ((1 - fy[:, None]) * (c10 - c00)
              + fy[:, None] * (c11 - c01)) / CHAN_RES
        sy = ((1 - fx[:, None]) * (c01 - c00)
              + fx[:, None] * (c11 - c10)) / CHAN_RES
        return gain, sx, sy

    # ============================ 通信链 ============================
    def _link(self):
        """逐用户 SINR→频谱效率→带宽份额→速率（通道层；assoc 冻结口径）。"""
        if self._dyn:
            G_db, sx, sy = self._interp_dyn(self.pos)
        else:
            G_db, sx, sy = self._interp(self.pos)
        g = 10.0 ** (G_db / 10.0)                      # 线性增益（含表内阴影）
        P = self.p * 10.0 ** (self.p_ref_dbm / 10.0)   # 绝对发射功率 mW
        sig = P[:, None] * g                           # (N,M)
        agg = (self.assoc * sig).sum(axis=0)           # (M,) Σ_j x_jm p_j g_jm
        denom = self._n0_lin + agg[None, :] - self.assoc * sig
        sinr = sig / np.maximum(denom, 1e-30)
        r = np.log1p(sinr) / np.log(2.0)               # bps/Hz
        if self.bw_share == "assoc_equal":             # P1 消融臂
            wgt = self.assoc
        else:                                          # 预注册默认：按需求比例
            wgt = self.assoc * self.dem[None, :]
        bsh = self.b[:, None] * wgt / np.maximum(
            wgt.sum(axis=1, keepdims=True), 1e-9)
        R = bsh * r                                    # Mbps（b 取 MHz 口径）
        self._G_db, self._sx, self._sy = G_db, sx, sy
        self._g, self._P, self._sig, self._denom = g, P, sig, denom
        self._sinr, self._r, self._bsh, self._R = sinr, r, bsh, R

    def _aggregates(self):
        """ρ/R_avail/u/T —— v2 语义：用**更新后**的关联计算（π 同步响应入内）。"""
        self.R_avail = (self.assoc * self._R).sum(axis=1)
        self.rho = self.assoc @ self.dem
        self.u = self.rho / np.maximum(self.R_avail, EPS_U)
        self.T = (self.assoc * np.minimum(self.dem[None, :], self._R)).sum(axis=1)

    # ==================== Lloyd 质心的信息获取（用户裁定） ====================
    def lloyd_user_pos(self) -> np.ndarray:
        """Lloyd 质心用的用户位置：**每 0.5 秒采样一次、每维 10 米测量误差**。

        2026-10-30 用户裁定：质心不准拿高频全局量。全局量每 LLOYD_DELAY_STEPS
        步才读一次，采样之间零阶保持——通信量是逐用户坐标的 1/5，不是全量。

        噪声每次采样加一次（不是每步加），种子由训练种子与步号组合，可复现。
        首步之前还没有采样值时用初始位置补一次采样。
        """
        if self._lloyd_sample is None:
            _r = np.random.default_rng(int(self.seed) * 100003 + 7)
            self._lloyd_sample = self.user_pos + _r.normal(
                0.0, LLOYD_POS_ERR_M, self.user_pos.shape)
        return self._lloyd_sample

    # ==================== 回传链路（转发中继，Jiang 2025 §II-A） ====================
    def backhaul_capacity(self) -> np.ndarray:
        """回传容量（Mbps）。模型见模块级 `bh_capacity_mbps`。"""
        return bh_capacity_mbps(self.pos, self.bs_xy)

    def _apply_backhaul(self):
        """端到端容量 = min(接入, 回传)：Jiang 2025 式 (19) 的等价写法。

        约束原文 `Σ_u c_s,u·R^A_s,u ≤ B^B·R^B_s`，即一架机服务的全部用户
        接入吞吐之和不得超过它的回传容量。超出时按机等比缩放逐用户速率。

        `R_access` 保留缩放**前**的接入量——块拼接场景（env_n1000）要把
        边界用户的解析速率并回来再统一取小。
        """
        A = (self.assoc * self._R).sum(axis=1)          # 接入段总速率 Mbps
        self.R_access = A
        if not self.relay_enable:
            self.bh_cap = np.full_like(A, 1e9)
            return
        cap = self.backhaul_capacity()                  # 回传段容量 Mbps
        self.bh_cap = cap
        if np.all(cap >= A):                            # 回传不紧（常态），不动
            return
        scale = np.minimum(1.0, cap / np.maximum(A, 1e-9))
        self._R *= scale[:, None]
        self.R_avail = A * scale
        self.rho = self.assoc @ self.dem
        self.u = self.rho / np.maximum(self.R_avail, EPS_U)
        self.T = (self.assoc * np.minimum(self.dem[None, :], self._R)).sum(axis=1)

    def _refresh_link(self):
        if getattr(self, "_fast3", None) is not None:
            self._refresh_link_fast_dyn()
            self._apply_backhaul()
            return
        self._link()
        self._apply_backhaul()
        attn = (ALPHA_ASSOC * self._r
                + KAPPA * (self.R_avail / (self.rho + 1e-3))[:, None]
                + self.pi[:, None])
        x_star = np.exp(attn - attn.max(axis=0, keepdims=True))
        x_star /= x_star.sum(axis=0, keepdims=True)
        a = 1.0 / self.tau_u
        self._x_star = x_star
        self._attn = attn
        self.assoc = (1.0 - a) * self.assoc + a * x_star
        self._aggregates()

    # ============================ 场景需求 ============================
    def _demand_step(self):
        """需求跟人走（社会力人群移动）：人均需求恒定，分布随位置重构。"""
        return

    def _load_crowd(self) -> ctypes.CDLL:
        """加载人群社会力库（crowd_sfm；缺库大声报错，禁静默回退）。"""
        for name in ("crowd_sfm.so", "crowd_sfm.dll"):
            p = Path(__file__).parent / name
            if p.exists():
                lib = ctypes.CDLL(str(p))
                dp = ctypes.POINTER(ctypes.c_double)
                vp = ctypes.c_void_p
                lib.crowd_init.argtypes = [
                    ctypes.c_int, dp, dp, ctypes.c_double, ctypes.c_double,
                    ctypes.c_double, ctypes.c_double, ctypes.c_double,
                    ctypes.c_double, ctypes.c_double]
                lib.crowd_init.restype = vp
                lib.crowd_step.argtypes = [vp, ctypes.c_double, dp]
                lib.crowd_step.restype = ctypes.c_int
                lib.crowd_gather.argtypes = [
                    vp, dp, ctypes.c_int, ctypes.c_double,
                    ctypes.c_double]
                lib.crowd_gather.restype = ctypes.c_int
                lib.crowd_free.argtypes = [vp]
                lib.crowd_free.restype = ctypes.c_int
                return lib
        raise RuntimeError(
            "CROWD_SFM MISSING——人群社会力库不存在"
            "（code/crowd_sfm.so|.dll），两场景训练均不可用；"
            "先编译 crowd_sfm.cpp（g++ -O3 -fopenmp -shared -fPIC "
            "-I shim -I ../deps/lightsfm/include）")

    def _crowd_setup(self) -> None:
        """人群引擎初始化（句柄版，环境实例间互不串台）：
        初始布局布点、静止起步、快慢分组巡游目标。

        快慢分组=用户索引偶慢奇快（确定性混编，无随机源）；巡游半径随
        热点 σ 缩放（密度不变口径）：快组 1.0×σ（跨热点穿行）、慢组
        σ/3（热点内踱步）。
        """
        if getattr(self, "_crowd_h", None) is not None:
            self._crowd_lib.crowd_free(self._crowd_h)
        v_des = np.where(np.arange(self.M) % 2 == 0, SFM_VSLOW, SFM_VFAST)
        pos = np.ascontiguousarray(self.user_pos0, dtype=np.float64)
        self._crowd_h = self._crowd_lib.crowd_init(
            ctypes.c_int(self.M),
            pos.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            np.ascontiguousarray(v_des, dtype=np.float64).ctypes.data_as(
                ctypes.POINTER(ctypes.c_double)),
            ctypes.c_double(self.field_w), ctypes.c_double(self.field_h),
            ctypes.c_double(self._hot_sigma),
            ctypes.c_double(self._hot_sigma / 3.0),
            ctypes.c_double(3.0), ctypes.c_double(3.0),
            ctypes.c_double(SFM_CELL))
        if not self._crowd_h:
            raise RuntimeError("crowd_init 失败（句柄为空）")

    def _reset_linear(self) -> None:
        """直线移动场景的回合初始化（Rizvi 原场景：用户匀速移动）。

        方向按固定种子一次性抽定、回合内不变——论文用户沿最短路从
        Crisis Control Center 出发，速度 10 m/s；本环境无路网图，
        用"从初始撒点出发的匀速直线 + 边界反弹"复现同一物理量级。
        """
        rs = np.random.default_rng(self.seed + 31337)
        ang = rs.uniform(0.0, 2.0 * np.pi, self.M)
        self._user_dir = np.column_stack([np.cos(ang), np.sin(ang)])
        self.user_pos = self.user_pos0.copy()

    def _linear_step(self) -> None:
        """直线匀速推进（步长 = 用户速度 × 环境步长）+ 边界弹性折返。"""
        sp = float(self._user_speed) * ENV_DT
        self.user_pos = self.user_pos + sp * self._user_dir
        for k, lim in ((0, self.field_w), (1, self.field_h)):
            lo = self.user_pos[:, k] < 0.0
            hi = self.user_pos[:, k] > lim
            self.user_pos[lo, k] = -self.user_pos[lo, k]
            self.user_pos[hi, k] = 2.0 * lim - self.user_pos[hi, k]
            flip = lo | hi
            self._user_dir[flip, k] *= -1.0
        self.user_pos = np.clip(self.user_pos, [0.0, 0.0],
                                [self.field_w, self.field_h])

    # ============================ 生命周期 ============================
    @property
    def _n0_lin(self) -> float:
        return 10.0 ** (self.n0_dbm / 10.0)            # mW（每资源块口径）

    def reset(self, seed: int | None = None) -> np.ndarray:
        if seed is not None:
            self.seed = int(seed)
        # σ 随机化（用户裁定）：每回合抽一个，抽法与训练种子组合——
        # 同种子同回合号必得同一 σ，换回合必换 σ，可复现且不重复。
        self._ep_n += 1
        if SHADOW_RANDOM:
            _r = np.random.default_rng(
                SHADOW_EPISODE_SEED + self.seed * 1009 + self._ep_n)
            self._shadow_sigma = float(
                _r.uniform(SHADOW_SIGMA_MIN, SHADOW_SIGMA_MAX))
        self._shadow_rt = None          # 强制重建阴影矩阵（旧回合的实现作废）
        self._shadow_t = -1
        self.dem = self.dem_base.copy()
        self.user_pos = self.user_pos0.copy()
        self._lloyd_sample = None       # 低频采样的缓存（首步采样前为空）
        if self.user_motion in ("flow", "burst"):
            self._crowd_setup()                    # 回合边界：人群回初始布局
        elif self.user_motion == "linear":
            self._reset_linear()                   # 直线场景：方向重抽、位置归位
        self.b = np.full(self.n_uav, self.b_bar)
        self.pi = np.zeros(self.n_uav)
        self.p = np.ones(self.n_uav)
        self.t = 0
        self.stats = EnvStats()
        if self._phys is not None:
            self.pos = self._phys.reset(self.init_xy)
        else:
            self.pos = self.init_xy.copy()
        self.adj = self._adj0.copy()
        G0, _, _ = (self._interp_dyn(self.pos) if self._dyn
                    else self._interp(self.pos))
        attn0 = G0 / 10.0                              # 初始关联=纯信道吸引
        x0 = np.exp(attn0 - attn0.max(axis=0, keepdims=True))
        x0 /= x0.sum(axis=0, keepdims=True)
        self.assoc = x0
        self._refresh_link()
        self._obs_cache = None
        self._edge_cache = None
        self._u_stale = self.u.copy()
        return self.obs()

    def step(self, d_act: np.ndarray):
        """d_act: (N,5) [Δb,Δπ,Δp,Δx,Δy] 逐维裁剪 → (obs, r, done, info)。"""
        d_act = np.atleast_2d(d_act)
        self.b = np.clip(
            self.b + np.clip(d_act[:, 0], DB_MIN, DB_MAX),
            1e-3, self.b_total)       # 带宽个体上下界（2026-10-27 钳制令）
        self.pi = np.clip(self.pi + np.clip(d_act[:, 1], DPI_MIN,
                                             DPI_MAX), -3.0, 3.0)   # 优先级域（对齐 obs 使用点既有 clip 语义）
        self.p = np.clip(self.p + np.clip(d_act[:, 2], DP_MIN, DP_MAX),
                         0.2, 5.0)      # 功率物理域（2026-10-27：负功率根因修复）
        dxy = d_act[:, 3:5]
        self._project()
        self.t += 1
        if self.user_motion in ("flow", "burst"):
            rc = self._crowd_lib.crowd_step(
                self._crowd_h,
                ctypes.c_double(SFM_DT),
                self._crowd_pos.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))
            if rc != 0:
                raise RuntimeError("crowd_step 失败")
            self.user_pos = self._crowd_pos        # 人群位置（需求跟人走）
        elif self.user_motion == "linear":
            self._linear_step()                    # 基线原生场景：直线匀速
        # static：用户位置固定（论文无移动模型），user_pos 保持 user_pos0
        # Lloyd 质心的信息获取（2026-10-30 用户裁定）：低频采样，不是每步取全局量。
        # 每 LLOYD_DELAY_STEPS 步采一次，采样之间零阶保持——这才是"近乎免费"的
        # 真实形态（全局量每 0.5 秒才读一次，不是每 0.1 秒读一次）。
        if self.t % LLOYD_DELAY_STEPS == 0:
            _r = np.random.default_rng(
                int(self.seed) * 100003 + self.t * 31 + 7)
            self._lloyd_sample = self.user_pos + _r.normal(
                0.0, LLOYD_POS_ERR_M, self.user_pos.shape)
        if self.scene == "burst" and self.t == GATHER_T0:
            # 散场语义（2026-10-27 用户裁定 A）：聚集点=外生出口
            # （长边两端、内收 10%），不取热点——全场人群真实迁移
            centers = np.ascontiguousarray(np.array(
                [[0.1 * self.field_w, 0.5 * self.field_h],
                 [0.9 * self.field_w, 0.5 * self.field_h]],
                dtype=np.float64))
            rc2 = self._crowd_lib.crowd_gather(
                self._crowd_h,
                centers.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                ctypes.c_int(GATHER_K),
                ctypes.c_double(self._hot_sigma * GATHER_R_FRAC),
                ctypes.c_double(SFM_V_RUSH))
            if rc2 != 0:
                raise RuntimeError("crowd_gather 失败")
        self._demand_step()
        if self._phys is not None:
            self.pos = self._phys.step(dxy)
            self.adj = self._phys.adjacency(self.neigh_dist)
            self.neighbors = [np.where(self.adj[i])[0]
                              for i in range(self.n_uav)]
            self.edges = np.array([(i, j) for i in range(self.n_uav)
                                   for j in self.neighbors[i] if i < j])
        else:
            self.pos = np.clip(self.pos + dxy, [0, 0],
                               [self.field_w, self.field_h])
            # 2026-10-31 自查修正：无物理引擎分支原先不更新邻接，adj 冻结在
            # 初始位置——原生场景无人机移动后邻居集失真（obs 的邻域统计、
            # tmarl 的邻居输入全部受影响）。按当前位置重算，与
            # _phys.adjacency() 同口径（半径 neigh_dist、排除自身）。
            _duu = np.sqrt(((self.pos[:, None, :] - self.pos[None, :, :])
                            ** 2).sum(-1))
            self.adj = (_duu <= self.neigh_dist) & (_duu > 0)
            self.neighbors = [np.where(self.adj[i])[0]
                              for i in range(self.n_uav)]
            self.edges = np.array([(i, j) for i in range(self.n_uav)
                                   for j in self.neighbors[i] if i < j])
        self._refresh_link()
        r = self._reward(d_act)
        if self.t % self.t_comm == 0:                   # T_comm：信息缓存刷新
            self._obs_cache = None
            self._edge_cache = None
            self._u_stale = self.u.copy()
        self.stats.var_u.append(float(np.var(self.u)))
        self.stats.var_u_clip.append(float(np.var(np.clip(self.u, 0.0, 5.0))))
        self.stats.n_dead.append(int((self.u > 5.0).sum()))
        self.stats.u_mean.append(float(np.mean(self.u)))
        self.stats.sum_t.append(float(self.T.sum()))
        self.stats.n_over.append(int((self.u > 1.0).sum()))
        self.stats.e_fly.append(float(self._phys.e_fly.sum())
                                if self._phys is not None else 0.0)
        return self.obs(), r, self.t >= EPISODE_LEN, {}

    def _project(self) -> None:
        self.b = np.maximum(self.b, 1e-6)
        s = self.b.sum()
        if s > self.b_total:
            self.b *= self.b_total / s
        self.pi = np.clip(self.pi, -3.0, 3.0)
        self.p = np.clip(self.p, P_MIN, P_MAX)

    def _reward(self, d_act: np.ndarray):
        """逐机奖励 (N,)：局部二次型，全部输入来自本机观测。

          均衡度 bal  = 1/(1 + Σ_j(û_i−û_j)²/(deg_i+1))
          负载度 load = max(R_MIN, 1 − ((û_i−U_STAR)/SIGMA)²)
          吞吐度 eff  = clip(T_i/b_bar/SE_REF, 0, 1)

        û=load_view(u) 与观测第 0 维同一映射；deg_i=0 时 d2=0、bal=1，
        公式自然成立，无孤立机分支。奖励允许为负；不做 [0,1] 截断。
        """
        ue = load_view(self.u)
        deg = self.adj.sum(axis=1)
        s1 = self.adj @ ue
        s2 = self.adj @ (ue ** 2)
        d2 = deg * ue ** 2 - 2.0 * ue * s1 + s2
        d2 = np.maximum(d2, 0.0)                 # 数值非负
        bal = 1.0 / (1.0 + d2 / (deg + 1.0))
        # 2026-11-02 v2（docs/06）：û>U_OVER 区从平底改线性下坡——过载越深
        # 罚越重（有界到 LOAD_FLOOR），恢复速度由此获得学习梯度；û≤U_OVER 段不变。
        load = 1.0 - ((ue - U_STAR) / SIGMA) ** 2
        load = np.where(ue > U_OVER,
                        np.maximum(LOAD_FLOOR,
                                   R_MIN - LOAD_SLOPE * (ue - U_OVER)),
                        np.maximum(R_MIN, load))
        eff = np.clip(self.T / max(self.b_bar, 1e-9) / SE_REF, 0.0, 1.0)
        return W_BAL * bal + W_LOAD * load + W_EFF * eff

    # ============================ 观测/邻域 ============================
    @property
    def u_eff(self) -> np.ndarray:
        """共享学习通道的唯一负载视图：C_logc（u≤3 恒等，u>3 严格单调）。

        观测第一维、邻居统计、奖励全部用它；不再有 clip 或尾惩罚。
        stats/指标仍保留原始 u。
        """
        return load_view(self.u)

    def obs(self) -> np.ndarray:
        """o_i = [û,ρ,R,b,π,p,T,over | 邻û均值,邻û方差,邻居数/4] (N,12)。
        T_comm>1 时非刷新步返回缓存（邻域信息时滞=通讯退化轴）。"""
        if self._obs_cache is not None:
            return self._obs_cache
        ue = self.u_eff
        over = (self.u > 1.0).astype(float)
        cnt = self.adj.sum(axis=1)
        safe = np.maximum(cnt, 1)
        uu = load_view(self._u_stale if self.t_comm > 1 else self.u)
        nb_mean = (self.adj @ uu) / safe
        nb_var = (self.adj @ (uu ** 2)) / safe - nb_mean ** 2
        rows = [ue, self.rho, self.R_avail, self.b, self.pi, self.p,
                self.T, over, nb_mean, nb_var, cnt / 4.0,
                # 第 12 维：回传余量，对数标度 log10(回传容量/接入需求)/3，
                # 范围 [0,1]。1.0 = 回传宽出接入 1000 倍，0.4 ≈ 25 倍，
                # 0 = 回传与接入持平或更紧。用对数是因为常态下 cap/A 落在
                # 15~226（线性标度会全部饱和成常数，这一维就没有信息了）。
                np.clip(np.log10(np.maximum(
                    self.bh_cap / np.maximum(self.R_access, 1e-9), 1e-9)),
                    0.0, 3.0) / 3.0]
        o = np.stack(rows, axis=1).astype(np.float32)
        self._obs_cache = o
        return o

    def init_report(self) -> dict:
        return {"u_mean": float(self.u.mean()), "u_std": float(self.u.std()),
                "u_min": float(self.u.min()), "u_max": float(self.u.max()),
                "R_min": float(self.R_avail.min()),
                "R_max": float(self.R_avail.max()),
                "rho_total": float(self.rho.sum()),
                "sinr_db_med": float(np.median(10 * np.log10(np.maximum(
                    self._sinr[self.assoc > 0.01], 1e-30))))}

    def close(self) -> None:
        if self._phys is not None:
            try:
                self._phys.aviary.close()
            except Exception:                              # noqa: BLE001
                pass


# ===== 动作解码与邻域收集（原 controllers 的纯函数，随环境一并提供） =====
# ---- 外部基线 dqnm 臂动作格（Rizvi TMLCN 2025 掩码 SDQN 移植）----
# 8 向移动+悬停=9 移动格 ×3 功率档 ×3 带宽档 = 81 格；Δπ≡0（无波束维）。
MOVES9 = np.array([[0.0, 0.0]] + [
    [np.cos(k * np.pi / 4), np.sin(k * np.pi / 4)] for k in range(8)],
    dtype=np.float32) * DXY_MAX
# 三档功率：绝对档位，取环境物理域的端点与标称点（P 域 [P_MIN, P_MAX]）
DP_LEVELS = np.array([P_MIN, 1.0, P_MAX])
# 三档带宽：绝对档位，每机均份额 b_bar 的 0.5 / 1.0 / 1.5 倍
DB_LEVELS = np.array([0.5, 1.0, 1.5])
N_ACT81 = len(MOVES9) * len(DP_LEVELS) * len(DB_LEVELS)     # = 81


def load_view(u: np.ndarray) -> np.ndarray:
    """共享学习通道唯一负载视图：C_logc（与势场一致）。"""
    from pfmath import MAPS          # 延迟导入，避免与 pfmath 的循环依赖
    return np.asarray(MAPS["C_logc"](np.asarray(u, dtype=np.float64))[0],
                      dtype=np.float64)


def norm_obs(x: np.ndarray) -> np.ndarray:
    """12 维观测逐维除以 OBS_NORM，压到同一量级（所有智能体统一口径）。"""
    return (np.asarray(x, dtype=np.float32) / OBS_NORM[None, :]).astype(np.float32)


def decode81(a_idx: np.ndarray, env) -> np.ndarray:
    """81 格 → (N,5) 五元动作。功率与带宽是**绝对档位**，解码为到档的增量。

    论文的动作是「3 功率档 × 3 带宽档」，档位是绝对量；环境接口收的是增量，
    因此这里换算 target − 当前，再按环境物理域截断。
    """
    mv, dp, db = a_idx // 9, (a_idx // 3) % 3, a_idx % 3
    d = np.zeros((len(a_idx), 5), dtype=np.float32)
    d[:, 0] = np.clip(DB_LEVELS[db] * env.b_bar - env.b, DB_MIN, DB_MAX)
    d[:, 2] = np.clip(DP_LEVELS[dp] - env.p, DP_MIN, DP_MAX)
    d[:, 3:5] = MOVES9[mv]
    return d


def invalid_mask81(env) -> np.ndarray:
    """Rizvi 无效动作掩码：(N,81) True=有效。位置出场地界 → 无效；
    功率与带宽是绝对档位，三档都落在物理域内，恒有效。"""
    nxt = env.pos[:, None, :] + MOVES9[None, :, :]          # (N,9,2)
    bad_mv = ((nxt[..., 0] < 0) | (nxt[..., 0] > env.field_w) |
              (nxt[..., 1] < 0) | (nxt[..., 1] > env.field_h))
    m = np.ones((env.n_uav, N_ACT81), dtype=bool)
    for mv in range(len(MOVES9)):
        m[:, mv * 9:(mv + 1) * 9] = ~bad_mv[:, mv][:, None]
    return m


def select81(q: np.ndarray, env, eps: float,
             rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """ε-greedy+无效掩码：Q[无效]=-1e9 → argmax；探索从有效格均匀采。
    返回 (a_idx(N,), d(N,5))——dqn_arm 训练循环与 Controller.dqnm 分支同走此门。

    全无效机保护（2026-10-29 崩溃修复）：边界态下可能出现某架 81 格全被判无效，
    `rng.integers(0)` 直接抛 `ValueError: high <= 0`，整条训练在 1119 回合炸掉。
    这里强制开放中央格（index 4 = 悬停 + 标称功率档 + 标称带宽档）——
    悬停不动，永远合法；同格若本身有效则保持原样。
    """
    m = invalid_mask81(env)
    dead = ~m.any(axis=1)
    if dead.any():
        m[dead, 4] = True                      # 中央保持格兜底：物理上恒可行
    a = np.where(m, q, -1e9).argmax(axis=1)
    ex = rng.random(env.n_uav) < eps
    for i in np.where(ex)[0]:                              # 掩码内探索（Rizvi 同款）
        valid = np.where(m[i])[0]
        if len(valid) == 0:                    # 兜底后不应发生；防御性跳过
            continue
        a[i] = valid[rng.integers(len(valid))]
    return a, decode81(a, env)
def decode_ddp(polar: np.ndarray) -> np.ndarray:
    """B 批 ddpga 臂（Cheng TIM 2024 移植）：(N,2)[ψ,D] → (N,5)。
    资源维 (Δb,Δπ,Δp)≡0——Cheng 部署-only 口径（无资源维，§六推测载体）。
    ddp_arm 训练循环与 Controller.ddpga 分支同走此门。"""
    d = np.zeros((len(polar), 5), dtype=np.float32)
    d[:, 3] = polar[:, 1] * np.cos(polar[:, 0])
    d[:, 4] = polar[:, 1] * np.sin(polar[:, 0])
    return d
def nb_index(adj: np.ndarray, k_fix: int | None = None,
             pos: np.ndarray | None = None):
    """邻居集**下标**与掩码——collect_nb 的排序单点。

    给定 pos 时按距离最近取前 k 个；否则按布尔邻接矩阵的索引序。
    返回 (idx(N,k) int16, mask(N,k) True=真邻居)。
    下标形式供 tmarl 的 replay 用：存下标而不是邻居观测，内存与 N 成正比。
    """
    deg = adj.sum(axis=1).astype(int)
    k = k_fix or max(int(deg.max()), 1)
    if pos is not None:
        d2 = ((pos[:, None, :] - pos[None, :, :]) ** 2).sum(-1)
        d2 = np.where(adj, d2, np.inf)              # 非邻居置无穷
        order = np.argsort(d2, axis=1)[:, :k]
    else:
        order = np.argsort(-adj.astype(np.int8), axis=1)[:, :k]   # bool adj 归一
    mk = np.zeros((len(adj), k), dtype=bool)
    j = min(order.shape[1], k)
    mk[:, :j] = np.arange(j)[None, :] < deg[:, None]
    return order.astype(np.int16), mk


def collect_nb(obs: np.ndarray, adj: np.ndarray, k_fix: int | None = None,
               pos: np.ndarray | None = None):
    """邻居 obs 集合向量化收集（变长→pad）。B 批 tmarl 臂用（Chen T-MARL
    的变长节点输入机制）。k_fix 给定则 pad/截断到固定宽（Replay 定长需）。
    返回 (nb(N,k,11), mask(N,k) True=真邻居)——nb 由 nb_index 的下标 gather。

    给定 pos 时按**距离最近**排序取前 k 个。注：这是定长 Replay 下的
    工程取舍，论文的 attention 本身不规定取哪 k 个。"""
    idx, mk = nb_index(adj, k_fix, pos)
    return obs[idx], mk

def clip_act(d: np.ndarray) -> np.ndarray:
    return np.clip(d, ACT_LOW[None, :], ACT_HIGH[None, :])


K_NB_FIX = 12    # tmarl 邻居集固定宽（>物理上限 ~8：250m 半径/150m 网格；
#   度数动态涨会炸定长 Replay——固定宽训练/评估同口径）



