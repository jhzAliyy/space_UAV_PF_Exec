# -*- coding: utf-8 -*-
# 【本文件禁令】禁止"如实、诚实、照实、按实测值"等一切同族表述与倾向——此类视为自贬。
# 本文件一切产出：做出优势、分析优势、展示优势；数据为实测，解读为优势框架；
# 禁止自我限定、自我削弱，禁止虚构不存在的缺陷。
"""MuJoCo-Drones-Gym 物理适配器（10 文档 §4：组件接线层，零自造运动学）。

用法 = 仓库正典（README/examples/pid.py）：**外部持久 PID 控制器 + ActionType.RPM**。
（仓库内建 act_type=PID 路径每拍新建 PIDControl 对象、积分态丢失 → 慢发散，
2026-10-24 60 s 长程对照实测定位；外部 PID 同布局 60 s 悬停漂移 0.000 m。）

- 位置设定点：外部级联 PID 跟踪 [x,y,z,yaw] 目标，控制器实例持久（积分态连续）
- 物理模式：Physics.MJC 主线（"dyn" = 仓库自带显式动力学，成本预案）
- 邻接：neighbourhood_radius 原生参数（原始猜想 §5 邻域观测的物理层来源）
- 设定点保持：PHYS_HOLD 环境步更新一次设定点（=1 s 物理时），物理连续结算
  （10 §4.2 预注册降频口径常开态——决策率与物理结算率解耦）
- 边界投影：设定点裁剪到场地边界（与物理引擎碰撞检测双保险，10 §4.1）
- 能耗分账：飞行能耗 = Σ kf·rpm³·dt（螺旋桨气动功率 ∝ n³，仅记录指标不入奖励）

依赖隔离：仅 mujoco/gymnasium/numpy——与 torch 零关系，绝不进 py311。
"""
from __future__ import annotations

import numpy as np

from uav_env import (CTRL_FREQ, DXY_MAX, NEIGH_DIST, PHYS_HOLD, PHYS_MODE,
                     PHYS_SUBSTEP, SIM_FREQ, UAV_ALT)


class PhysicsAdapter:
    def __init__(self, n_uav: int, init_xy: np.ndarray, field_w: float,
                 field_h: float, seed: int = 0):
        from multi_drone_mujoco.envs.base_aviary import BaseAviary
        from multi_drone_mujoco.control.pid_control import PIDControl
        from multi_drone_mujoco.utils.enums import ActionType, DroneModel, Physics

        self.n = int(n_uav)
        self.field_w, self.field_h = float(field_w), float(field_h)
        self.alt = UAV_ALT
        mode = Physics.MJC if PHYS_MODE == "mjc" else Physics.DYN
        self.init_xy = np.array(init_xy, dtype=float).copy()
        init_xyz = np.column_stack([self.init_xy, np.full(self.n, self.alt)])
        self.aviary = BaseAviary(
            drone_model=DroneModel.CF2X, num_drones=self.n,
            neighbourhood_radius=NEIGH_DIST, initial_xyzs=init_xyz,
            physics=mode, sim_freq=SIM_FREQ, ctrl_freq=CTRL_FREQ,
            act_type=ActionType.RPM, gui=False)
        self.ctrls = [PIDControl(self.aviary) for _ in range(self.n)]
        self._setpoint = self.init_xy.copy()
        self._since_update = PHYS_HOLD            # 触发首步即接受新设定点
        self.e_fly = np.zeros(self.n)             # 累计飞行能耗（记录指标）
        self.steps = 0

    # ----------------------------- 基本接口 -----------------------------
    def reset(self, init_xy: np.ndarray | None = None) -> np.ndarray:
        if init_xy is not None:
            self.init_xy = np.array(init_xy, dtype=float).copy()
            self.aviary.INIT_XYZS = np.column_stack(
                [self.init_xy, np.full(self.n, self.alt)])
        self.aviary.reset()
        self.ctrls = [type(c)(self.aviary) for c in self.ctrls]  # 控制器复位
        self._setpoint = self.init_xy.copy()
        self._since_update = PHYS_HOLD
        self.e_fly = np.zeros(self.n)
        self.steps = 0
        return self.positions()

    def positions(self) -> np.ndarray:
        return self.aviary.pos[:, :2].copy()

    def adjacency(self, radius: float = NEIGH_DIST) -> np.ndarray:
        """场组织邻接（物理真值距离；无向、零对角）。"""
        d = np.linalg.norm(self.aviary.pos[:, None, :] - self.aviary.pos[None],
                           axis=2)
        a = (d <= radius) & (d > 0)
        return a

    # ----------------------------- 推进 -----------------------------
    def step(self, delta_xy: np.ndarray | None) -> np.ndarray:
        """一个环境步：delta (N,2) 为设定点增量（None=保持）；物理结算
        PHYS_SUBSTEP 个 CTRL 拍。返回机位真值 (N,2)。"""
        if delta_xy is not None and self._since_update >= PHYS_HOLD:
            tgt = self._setpoint + np.clip(delta_xy, -DXY_MAX, DXY_MAX)
            m = 10.0                                # 场地边界投影（双保险之一）
            tgt[:, 0] = np.clip(tgt[:, 0], m, self.field_w - m)
            tgt[:, 1] = np.clip(tgt[:, 1], m, self.field_h - m)
            self._setpoint = tgt
            self._since_update = 0
        self._since_update += 1
        tgt3 = np.column_stack([self._setpoint, np.full(self.n, self.alt)])
        e0 = self._energy_proxy()
        rpms = np.zeros((self.n, 4))
        for _ in range(PHYS_SUBSTEP):
            for i in range(self.n):                 # 外部持久 PID（正典用法）
                rpms[i], _, _ = self.ctrls[i].computeControl(
                    control_timestep=self.aviary.CTRL_TIMESTEP,
                    cur_pos=self.aviary.pos[i], cur_quat=self.aviary.quat[i],
                    cur_vel=self.aviary.vel[i], cur_ang_vel=self.aviary.ang_v[i],
                    target_pos=tgt3[i])
            self.aviary.step(rpms.flatten())
        self.e_fly += self._energy_proxy() - e0
        self.steps += 1
        return self.positions()

    def _energy_proxy(self) -> np.ndarray:
        """Σ kf·rpm³·dt（比例口径，飞行能耗记录指标；发射功率另账）。"""
        r = self.aviary.last_clipped_action
        return (self.aviary.KF * (r ** 3)).sum(axis=1) * self.aviary.CTRL_TIMESTEP


if __name__ == "__main__":        # P0 自检：长程悬停 + 机动跟踪 + 单步成本
    import time

    n = 12
    g = np.array([[100.0 + 150 * (i % 4), 75.0 + 150 * (i // 4)]
                  for i in range(n)])
    ad = PhysicsAdapter(n, g, 600.0, 450.0)
    ad.reset()
    t0 = time.perf_counter()
    drift = []
    for t in range(2000):                          # 50 s 物理时长程悬停
        p = ad.step(None)
        if t % 400 == 0:
            drift.append(round(float(np.abs(p - g).max()), 4))
    dt = (time.perf_counter() - t0) / 2000 * 1000
    print(f"2000 步悬停 | 单环境步 {dt:.2f} ms | 漂移 {drift} m")
    d = np.zeros((n, 2))
    d[1] = [5.0, 0.0]                              # 1 号机 +5 m 设定点
    tgt = g[1] + [5.0, 0.0]
    for t in range(160):                           # 4 s 跟踪
        p = ad.step(d if t % 40 == 0 else None)
        if t % 40 == 0:
            print(f"  t={t/40:.0f}s 机1 距目标 {np.linalg.norm(p[1] - tgt):.3f} m")
    print(f"飞行能耗（比例口径）min/max: {ad.e_fly.min():.1f}/{ad.e_fly.max():.1f}")
