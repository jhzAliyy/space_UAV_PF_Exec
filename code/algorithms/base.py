# -*- coding: utf-8 -*-
"""算法基类与注册协议。

每个算法 = BaseAlg 子类 + 一行注册。统一入口 run.py 与评估 evaluate.py
只认注册表，不认算法内部。训练一律从零（无续跑路径）。
"""
from __future__ import annotations

import numpy as np


class BaseAlg:
    """算法协议。

    子类必须：
      - 声明 needs_training（True=训练臂，False=免训臂）
      - 实现 controller(env) -> 可调用 ctrl(env) -> (N,5) 动作
      - 训练臂另实现 train(env)（内部自带日志/ckpt/进度落盘）
    """

    needs_training: bool = False
    name: str = ""

    def __init__(self, args):
        self.args = args

    def controller(self, env):
        raise NotImplementedError

    def train(self, env):
        raise NotImplementedError(f"{self.name} 为免训臂")
