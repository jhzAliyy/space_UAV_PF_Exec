# -*- coding: utf-8 -*-
"""统一入口：一个超参换算法。

用法（全部臂同一套命令行）：
  python run.py --algo learned --scene flow --n 12 --total-rollouts 1200
  python run.py --algo field   --scene flow --n 12            # 真纯场（免训）
  python run.py --algo tuned   --scene flow --n 12            # 调优遗迹（免训）
  python run.py --algo ddpga   --scene flow --n 12 --total-rollouts 1200
  python evaluate.py artifacts/lists/<x>.txt out.csv          # 评估同注册表

算法注册表在 algorithms/__init__.py；新增算法=实现 BaseAlg 子类+一行注册，
本文件与其他算法零改动。训练一律从零（无续跑路径）。
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="统一算法入口（注册表驱动）")
    # ---- 算法与任务 ----
    ap.add_argument("--algo", required=True,
                    help="算法注册名：见 algorithms/__init__.py 的 ALGOS")
    ap.add_argument("--task", default="", help="ckpt 任务名（缺省自动生成）")
    # ---- 环境超参 ----
    ap.add_argument("--scene", default="flow", choices=["flow", "burst", "static"])
    ap.add_argument("--n", type=int, default=12, help="无人机数")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tau", type=float, default=5.0)
    ap.add_argument("--tcomm", type=int, default=1, help="信息缓存刷新周期")
    ap.add_argument("--preset", default="", help="原生场景名（cheng7/shen3/jiang5/rizvi3/song361/chen6）")
    ap.add_argument("--chan", default="main", help="信道档（main）")
    # ---- 训练超参（默认值=旧管线口径） ----
    ap.add_argument("--total-rollouts", type=int, default=1200)
    ap.add_argument("--rollout-len", type=int, default=256,
                    help="日志/统计窗步数（旧管线 256；非回合长）")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--learning-starts", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--policy-freq", type=int, default=2)
    ap.add_argument("--target-noise", type=float, default=0.2)
    ap.add_argument("--noise-clip", type=float, default=0.5)
    ap.add_argument("--exploration-noise", type=float, default=0.3,
                    help="采集探索噪声（logit 域）")
    ap.add_argument("--buffer-rollouts", type=int, default=30,
                    help="replay 容量（×rollout_len×n）")
    # ---- 口径开关 ----
    ap.add_argument("--real-obs", action="store_true",
                    help="真实全局量退化（0.5 s 过期 + 10 m 位置噪声）")
    return ap.parse_args()


def main() -> int:
    a = parse_args()
    from algorithms import ALGOS
    if a.algo not in ALGOS:
        print(f"[run] 未知算法 {a.algo}；可用：{sorted(ALGOS)}", flush=True)
        return 2
    alg = ALGOS[a.algo](a)

    if a.real_obs:
        os.environ["PF_REAL_OBS"] = "1"       # 训练评估同口径

    from channel_env import build_env
    env = build_env(a)
    try:
        if alg.needs_training:
            alg.train(env)
        else:
            print(f"[run] {a.algo} 为免训臂，无需训练；评估请用 evaluate.py",
                  flush=True)
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
