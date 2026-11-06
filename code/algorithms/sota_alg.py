# -*- coding: utf-8 -*-
"""SOTA 对照臂（外部基线移植）：训练委托各自算法文件，从零恒定。

dqnm / ddpga / mfdqn / tmarl 四臂的模型与训练循环逐位沿用旧实现
（含各自的动作格与解码）；resume 恒 False——训练只有从零模式。
评估由 evaluate.py 按 ckpt 加载，不走本类的 controller。
"""
from __future__ import annotations

from algorithms.base import BaseAlg


class SotaAlg(BaseAlg):
    """一个超参换算法：module.train(mode, ...) 委托。"""

    needs_training = True

    def __init__(self, args, module_name: str):
        super().__init__(args)
        self.module_name = module_name
        self.name = module_name

    def train(self, env):
        import importlib
        mod = importlib.import_module(self.module_name)
        a = self.args
        mod.train(
            self.module_name, a.tau, a.seed, scene=a.scene,
            t_comm=getattr(a, "t_comm", 1) or 1, anchor=0.0,
            n_uav=a.n, chan=a.chan,
            total_rollouts=a.total_rollouts,
            resume=False,                       # 从零唯一模式
            task=a.task or None, preset=a.preset or None)
        env.close()
