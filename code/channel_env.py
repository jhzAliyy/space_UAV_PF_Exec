# -*- coding: utf-8 -*-
"""环境构造的唯一入口：从命令行超参到 UAVEnvV3。

场景/信道映射（CHAN_MAP）：列表文件的 chan 列 → (scenario, fc)。
原生场景（preset）复刻：baseline_scene 提供覆盖规格。
"""
from __future__ import annotations

from uav_env import UAVEnvV3, P_REF_DBM, N0_DBM, baseline_scene

CHAN_MAP = {
    "main": ("uma", 2e9),
}


def build_env(args):
    """run.py / evaluate.py 共用的环境构造。"""
    spec = baseline_scene(args.preset or None)
    scen, fc = CHAN_MAP[args.chan]
    return UAVEnvV3(
        tau_u=args.tau, seed=args.seed, scene=args.scene, n_uav=args.n,
        t_comm=getattr(args, "t_comm", 1) or 1,
        layout_n=max(12, args.n), use_physics=(spec is None),
        override_spec=spec, scenario=scen, fc_hz=fc,
        p_ref_dbm=P_REF_DBM, n0_dbm=N0_DBM,
        demand_scale=(spec or {}).get("demand_scale", 3.7648))
