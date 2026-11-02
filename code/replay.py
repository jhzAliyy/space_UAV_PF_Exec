# -*- coding: utf-8 -*-
"""ReplayBuffer：定长环形缓冲，按环境步追加、随机采样。

接口对标 cleanrl_utils.buffers.ReplayBuffer 的 add / sample 语义；
本实现按单环境展开，字段名 obs / next_obs / actions / rewards / dones。
v2（2026-11-02）：增存 a_field / next_a_field——TD3 目标与 actor 损失
都要用当场纯势场动作组合修正项。
"""
from __future__ import annotations

import numpy as np


class ReplayBuffer:
    def __init__(self, capacity: int, obs_dim: int, act_dim: int):
        self.cap = int(capacity)
        self.obs = np.zeros((self.cap, obs_dim), dtype=np.float32)
        self.actions = np.zeros((self.cap, act_dim), dtype=np.float32)
        self.rewards = np.zeros(self.cap, dtype=np.float32)
        self.next_obs = np.zeros((self.cap, obs_dim), dtype=np.float32)
        self.dones = np.zeros(self.cap, dtype=np.float32)
        self.a_field = np.zeros((self.cap, act_dim), dtype=np.float32)
        self.next_a_field = np.zeros((self.cap, act_dim), dtype=np.float32)
        self.n, self.i = 0, 0

    def add(self, obs, actions, rewards, next_obs, dones,
            a_field, next_a_field) -> None:
        m = len(actions)
        idx = (self.i + np.arange(m)) % self.cap
        self.obs[idx] = obs
        self.actions[idx] = actions
        self.rewards[idx] = rewards
        self.next_obs[idx] = next_obs
        self.dones[idx] = dones
        self.a_field[idx] = a_field
        self.next_a_field[idx] = next_a_field
        self.i = (self.i + m) % self.cap
        self.n = min(self.n + m, self.cap)

    def sample(self, batch_size: int, rng: np.random.Generator):
        j = rng.integers(0, self.n, batch_size)
        return (self.obs[j], self.actions[j], self.rewards[j],
                self.next_obs[j], self.dones[j],
                self.a_field[j], self.next_a_field[j])
