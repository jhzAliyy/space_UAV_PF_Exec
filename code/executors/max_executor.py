# -*- coding: utf-8 -*-
"""全 1（最大）执行器：方向 × 恒满速，没有任何其他东西。"""
from __future__ import annotations

import numpy as np

from executors.base import Executor, FieldReading


class MaxExecutor(Executor):
    name = "max"

    def act(self, env, reading: FieldReading) -> np.ndarray:
        return reading.step[None, :] * reading.s
