# -*- coding: utf-8 -*-
"""全局超参数单一来源。任何奖励/目标/窗口参数不得在其它模块写字面量。"""
from __future__ import annotations

import numpy as np

# ---- 奖励（简单二次型） ----
U_STAR = 0.80          # 锚定目标负载
U_OVER = 1.0           # 过载阈值：SIGMA 由此推导，使 load 在 u=1.0 触底
SIGMA = 0.1414213562373095  # (U_OVER-U_STAR)/sqrt(1-R_MIN)，过载即满罚
R_MIN = -1.0           # 负载项地板：极端死机不再把 critic 目标拉到无穷
W_BAL = 0.45
W_LOAD = 0.35
W_EFF = 0.20
SE_REF = 12.0          # 吞吐归一化参考 Mbps/MHz

# ---- 共享环境/训练 ----
EPISODE_LEN = 2000
OBS_NORM = np.array([3.0, 150.0, 200.0, 16.0, 3.0, 5.0, 150.0,
                     1.0, 3.0, 0.1, 2.0, 1.0], dtype=np.float32)
PF_U_MAP = "C_logc"
PF_RELAX = 0.2
PI_DUAL_STEP = 0.05
PI_MIN, PI_MAX = -3.0, 3.0
CKPT_EVERY = 25
TOTAL_ROLLOUTS = 1200
EARLY_WIN = 200
EARLY_SHORT = 10          # 早停：近 10 轮均值 vs 近 200 轮均值
EARLY_THR = 0.005

# ---- 执行器速度下限（2026-11-01 规模负效应修复，实测依据见
#      _tmp_tables/规模负效应分析_2026-11-01.md §四A） ----
# σ = [σ_b, σ_π, σ_p, σ_xy]，网络输出 lo + (1−lo)·sigmoid ∈ [lo, 1)。
# 根因：奖励在均衡附近对 σ 平坦（随机 σ 与学好的一样好），DDPG 把
# "平时无代价"的速度维漂到危险低值（n100 实测 σ_p≈0.04、σ_xy≈0.29、
# 低至 0.000）；突发后人群迁移距离 ∝ √n 而恢复窗恒定 140 s，同一漂移
# n12 容忍（111 s<140 s）、n100 必死（320 s≫140 s）→ 规模负效应。
# σ_xy 下限 = 人群冲刺 2.78 m/s ÷ DXY_MAX 5 m/s ≈ 0.56，取 0.6：
# n100 burst 确定性闭环实测 post u_mean=0.813、var_u=1.3e-4、
# n_over=0（验收线全过）。通信三维给防关死底 0.2（sigmoid 饱和曾把
# 恢复方向完全掐灭）。悬停不受影响：缺口=0 时任意 σ 都乘出 0。
SIG_LO = np.array([0.2, 0.2, 0.2, 0.6], dtype=np.float32)

# ---- 残差执行器 v2（2026-11-02 用户裁定 B 路，docs/06）----
# 动作 = clip(场指令 + Δ)，Δ = RES_FRAC×ACT_SCALE × tanh(logit)，逐维有界。
# Δ≡0 严格退化为纯势场；历史依据：残差形态曾真正超越纯势场，后遭 critic
# 噪声 bug 污染；critic 已改评执行动作（2026-11-02 修复），残差形态复活。
RES_FRAC = 0.4        # 修正项逐维上界 = RES_FRAC × ACT_SCALE
LOAD_SLOPE = 1.0      # û > U_OVER 线性下坡斜率（恢复期给残差学习梯度）
LOAD_FLOOR = -3.0     # 负载项深地板（有界，护 critic 目标）
