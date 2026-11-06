# -*- coding: utf-8 -*-
"""算法注册表：新增算法=实现 BaseAlg 子类+一行注册，其余零改动。

场族三别名 = 场 + 拔插执行器；SOTA 对照臂 = 外部基线移植（从零恒定）；
免训基线 = heuristics 六控制器。
"""
from __future__ import annotations

from algorithms.base import BaseAlg
from algorithms.field_alg import FieldAlg
from algorithms.sota_alg import SotaAlg


def _field_with(executor_name):
    def _make(args):
        return FieldAlg(args, executor_name)
    return _make


def _sota(module_name):
    def _make(args):
        return SotaAlg(args, module_name)
    return _make


def _baseline(mode):
    def _make(args):
        return _BaselineAlg(args, mode)
    return _make


class _BaselineAlg(BaseAlg):
    """免训基线臂（heuristics.make_baseline）。"""

    def __init__(self, args, mode):
        super().__init__(args)
        self.mode = mode
        self.name = mode

    def controller(self, env):
        from heuristics import make_baseline
        ctrl = make_baseline(self.mode, self.args.seed)

        def wrap(env):
            return ctrl(env)
        return wrap


ALGOS: dict[str, type[BaseAlg]] = {
    "field": _field_with("max"),      # 真纯场：方向×恒满速
    "tuned": _field_with("tuned"),    # 调优遗迹基线（对照）
    "learned": _field_with("net"),    # 学习执行器（逐维增益 η，方向恒由场给定）
    "ddpga": _sota("ddpga"),          # SOTA 对照（Cheng TIM 2024）
    "dqnm": _sota("dqnm"),            # SOTA 对照（Rizvi TMLCN 2025）
    "mfdqn": _sota("mfdqn"),          # SOTA 对照（Song TWC 2025）
    "tmarl": _sota("tmarl"),          # SOTA 对照（Chen T-ITS 2024）
    "rand": _baseline("rand"),        # 免训基线
    "fixed": _baseline("fixed"),
    "greedy": _baseline("greedy"),
    "rr": _baseline("rr"),
    "shen": _baseline("shen"),
    "jiang": _baseline("jiang"),
}
