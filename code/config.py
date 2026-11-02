# -*- coding: utf-8 -*-
"""全局超参数单一来源（重建版）。

本文件只放三类量：环境物理/奖励语义、训练通用、评估与真实观测口径。
执行器的一切参数不在此出现——方向决定器无参数；学习执行器的幅度
界由网络输出域（sigmoid）与环境物理单步极限（STEP_MAX）共同定义，
不需要常数。唯一例外：调优势场基线保留四个历史常数（见文件尾），
它们是对照遗迹，不是任何主方法的参数。
"""
from __future__ import annotations

import numpy as np

# ---- 奖励（两段式负载，2026-11-02 裁定） ----
U_STAR = 0.80          # 锚定目标负载
U_OVER = 1.0           # 过载阈值
SIGMA = 0.1414213562373095  # (U_OVER-U_STAR)/sqrt(1-R_MIN)，奖励形状推导量SIGMA = 0.1414213562373095  # (U_OVER-U_STAR)/sqrt(1-R_MIN)
R_MIN = -1.0           # 负载项地板
LOAD_SLOPE = 1.0       # û > U_OVER 线性下坡斜率（恢复期梯度）
LOAD_FLOOR = -3.0      # 负载项深地板（有界，护 critic 目标）
W_BAL = 0.45
W_LOAD = 0.35
W_EFF = 0.20
SE_REF = 12.0          # 吞吐归一化参考 Mbps/MHz

# ---- 空载 u 语义（2026-11-02 rizvi3 修复，环境侧） ----
IDLE_R_EPS = 1e-6
IDLE_U_CAP = 3.0

# ---- 共享环境/训练 ----
EPISODE_LEN = 2000
OBS_NORM = np.array([3.0, 150.0, 200.0, 16.0, 3.0, 5.0, 150.0,
                     1.0, 3.0, 0.1, 2.0, 1.0], dtype=np.float32)
PF_U_MAP = "C_logc"
PI_MIN, PI_MAX = -3.0, 3.0
CKPT_EVERY = 25
TOTAL_ROLLOUTS = 1200
EARLY_WIN = 200
EARLY_SHORT = 10
EARLY_THR = 0.005

# ---- 真实全局量退化（训练评估同口径；开关 PF_REAL_OBS=1） ----
REAL_OBS_EVERY = 5       # 0.5 s 刷新
REAL_OBS_BLUR_M = 10.0   # 用户位置测量噪声

# ---- 调优势场基线的历史常数（对照遗迹，2026-10 实测选定；不用于
#      任何主方法。真纯场=恒满速；学习版=网络出幅度） ----
TUNED_RELAX = 0.2        # 带宽维 20% 力度
TUNED_PI_STEP = 0.05     # 价格维次梯度步长
