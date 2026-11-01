# -*- coding: utf-8 -*-
"""ReplayBuffer：定长环形缓冲，按环境步追加、随机采样。

接口对标 cleanrl_utils.buffers.ReplayBuffer 的 add / sample 语义；
本实现按单环境展开，字段名 obs / next_obs / actions / rewards / dones。
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
        self.n, self.i = 0, 0

    def add(self, obs, actions, rewards, next_obs, dones) -> None:
        m = len(actions)
        idx = (self.i + np.arange(m)) % self.cap
        self.obs[idx] = obs
        self.actions[idx] = actions
        self.rewards[idx] = rewards
        self.next_obs[idx] = next_obs
        self.dones[idx] = dones
        self.i = (self.i + m) % self.cap
        self.n = min(self.n + m, self.cap)

    def sample(self, batch_size: int, rng: np.random.Generator):
        j = rng.integers(0, self.n, batch_size)
        return (self.obs[j], self.actions[j], self.rewards[j],
                self.next_obs[j], self.dones[j])

    def save_npz(self, path, rollout: int) -> None:
        """把整个环形缓冲写入 npz，rollout 用于和 ckpt 对齐。"""
        np.savez_compressed(
            path, obs=self.obs, actions=self.actions, rewards=self.rewards,
            next_obs=self.next_obs, dones=self.dones,
            n=np.int64(self.n), i=np.int64(self.i),
            cap=np.int64(self.cap), rollout=np.int64(rollout))

    def load_npz(self, path) -> int:
        """读回 npz，返回写入时的 rollout；形状/容量不符直接报错。"""
        with np.load(path) as z:
            if int(z["cap"]) != self.cap:
                raise ValueError(f"replay capacity {int(z['cap'])} != {self.cap}")
            if (z["obs"].shape != self.obs.shape
                    or z["actions"].shape != self.actions.shape):
                raise ValueError("replay buffer shape mismatch")
            self.obs[...] = z["obs"]
            self.actions[...] = z["actions"]
            self.rewards[...] = z["rewards"]
            self.next_obs[...] = z["next_obs"]
            self.dones[...] = z["dones"]
            self.n = int(z["n"])
            self.i = int(z["i"])
            return int(z["rollout"])
