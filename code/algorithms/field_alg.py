# -*- coding: utf-8 -*-
"""场算法：场输出包 → 拔插执行器 → 动作。一条链，训练评估同路。

--algo field/tuned/learned 是三个预绑别名；--executor 可直通覆盖。
"""
from __future__ import annotations

from algorithms.base import BaseAlg
from field import Field
from executors import EXECUTORS


class FieldAlg(BaseAlg):
    """场 + 执行器组合。executor_name 决定一切行为差异。"""

    def __init__(self, args, executor_name: str):
        super().__init__(args)
        self.executor_name = executor_name
        self.name = f"field+{executor_name}"
        self.needs_training = EXECUTORS[executor_name].needs_training
        self._field = Field()
        self._executor = None

    def controller(self, env):
        self._executor = EXECUTORS[self.executor_name](self.args)
        ex, f = self._executor, self._field

        def ctrl(env, deterministic=True, explore=0.0):
            reading = f.reading(env)          # 调用时取势场输出
            return ex.act(env, reading,       # 直接输入执行器模块处理
                          deterministic=deterministic,
                          explore=explore) if ex.needs_training \
                else ex.act(env, reading)
        return ctrl

    def train(self, env):
        from trainer import Trainer           # 学习执行器的训练管线
        Trainer(self.args).run(env)
