# -*- coding: utf-8 -*-
"""执行器协议：拔插模块。

统一调用链（训练与评估同一条）：
    reading = field.reading(env)        # 场输出包
    action  = executor.act(env, reading)  # 导入的执行器模块处理再输出

执行器各取所需：max/net 吃方向与步幅；tuned 吃目标差。
新增执行器 = 实现 Executor + 在 executors/__init__.py 注册一行。
"""
from __future__ import annotations

import numpy as np

from field import FieldReading


class Executor:
    """可拔插执行器。免训执行器直接 act；学习执行器另持网络并训练。"""
    needs_training: bool = False
    name: str = ""

    def act(self, env, reading: FieldReading) -> np.ndarray:
        raise NotImplementedError
