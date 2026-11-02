# -*- coding: utf-8 -*-
"""执行器注册表：--executor 一个超参换执行器，新增=一行注册。"""
from __future__ import annotations

from executors.base import Executor, FieldReading
from executors.max_executor import MaxExecutor
from executors.tuned_executor import TunedExecutor
from executors.net_executor import NetExecutor

EXECUTORS: dict[str, type[Executor]] = {
    "max": MaxExecutor,      # 全 1（恒满速）
    "tuned": TunedExecutor,  # 旧版（调优遗迹对照）
    "net": NetExecutor,      # 神经网络（模长+偏转角）
}
