# -*- coding: utf-8 -*-
# 【本文件禁令】禁止"如实、诚实、照实、按实测值"等一切同族表述与倾向——此类视为自贬。
# 本文件一切产出：做出优势、分析优势、展示优势；数据为实测，解读为优势框架；
# 禁止自我限定、自我削弱，禁止虚构不存在的缺陷。
"""全批次清单生成器（11 文档 §四 批次矩阵的程序化落档；受控常量锁死）。

列：mode tau seed [scene tcomm anchor reward chan n_uav tag]
锚点格：每批 c3 τ5 × 5种子 重训（tag=anchorX 强制新任务名）+ c4 免训评估——
批间漂移归因与健康监测（11 §三）。
F/G 批表就绪后由 --with-fg 生成。
"""
from __future__ import annotations

import argparse
from pathlib import Path

MD = Path(__file__).parent.parent / "manifests"
SEEDS = range(3)  # 2026-10-25 裁定：五种子降三种子
TAUS = (5, 20)
CORE4 = ("c3", "c3w", "c0", "c4")
SCENES_C = ("ou_slow", "ou_fast", "burst_a", "burst_b", "burst_c")


def w(name: str, lines: list[str]) -> None:
    MD.mkdir(exist_ok=True)
    (MD / name).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"{name}: {len(lines)} 行")


def anchor(batch: str) -> tuple[list[str], list[str]]:
    tr = [f"c3 5 {s} static_step 1 0 u7 main 12 anchor{batch}"
          for s in SEEDS]
    ev = [f"c4 5 {s} static_step 1 0 u7 main 12 anchor{batch}_c4"
          for s in SEEDS]
    return tr, ev


def batch_b():
    tr, ev = [], []
    # 免训退化评估：A 批 ckpt（ckpt_task 基名）在 T_comm∈{2,5,10,∞} 下
    for m in ("c3", "c0", "c4"):
        for tau in TAUS:
            for s in SEEDS:
                base = f"{m}_tau{tau}_s{s}" if m != "c4" else None
                for tc in (2, 5, 10, 1000):
                    ev.append(f"{m} {tau} {s} static_step {tc} 0 u7 main 12"
                              + (f" {base}" if base else ""))
    # 受限重训（训练即 tcomm=5）+ 其评估
    for m in ("c3", "c0"):
        for tau in TAUS:
            for s in SEEDS:
                tr.append(f"{m} {tau} {s} static_step 5")
                ev.append(f"{m} {tau} {s} static_step 5 0 u7 main 12"
                          f" tc5_{m}_tau{tau}_s{s}")
    a_tr, a_ev = anchor("B")
    return tr + a_tr, ev + a_ev


def batch_c():
    tr, ev = [], []
    for sc in SCENES_C:
        for m in ("c3", "c3w", "c0"):
            for tau in TAUS:
                for s in SEEDS:
                    tr.append(f"{m} {tau} {s} {sc}")
                    ev.append(f"{m} {tau} {s} {sc}")
        for tau in TAUS:
            for s in SEEDS:
                ev.append(f"c4 {tau} {s} {sc}")
    a_tr, a_ev = anchor("C")
    return tr + a_tr, ev + a_ev


def batch_d():
    tr, ev = [], []
    for tau in TAUS:
        for s in SEEDS:
            tr.append(f"c3 {tau} {s} static_step 1 1.0")      # λ_a=1 训练
            ev.append(f"c3 {tau} {s} static_step 1 1.0")
            ev.append(f"c4 {tau} {s} static_step 1 0")         # 无锚基线
            for lam in (0.5, 1.0, 2.0, 5.0):                   # 网格免训评估
                ev.append(f"c4 {tau} {s} static_step 1 {lam:g}")
    a_tr, a_ev = anchor("D")
    return tr + a_tr, ev + a_ev


def batch_e():
    tr, ev = [], []
    for m in ("c3", "c0"):                                     # C3-full/C0-full
        for tau in TAUS:
            for s in SEEDS:
                tr.append(f"{m} {tau} {s} ou_fast 5 1.0")
                ev.append(f"{m} {tau} {s} ou_fast 5 1.0")
    for tau in TAUS:                                           # C4-full/L-full
        for s in SEEDS:
            ev.append(f"c4 {tau} {s} ou_fast 5 1.0")
            ev.append(f"lctrl {tau} {s} ou_fast 5 1.0")
    a_tr, a_ev = anchor("E")
    return tr + a_tr, ev + a_ev


def batch_f():
    tr, ev = [], []
    for n in (30, 60, 100):
        for m in ("c3", "c3w", "c0", "b1"):   # b1=H-F1 判据载体（判据-矩阵对账补齐）
            for s in SEEDS:
                tr.append(f"{m} 5 {s} static_step 1 0 u7 main {n}")
                ev.append(f"{m} 5 {s} static_step 1 0 u7 main {n}")
        for s in SEEDS:
            tr.append(f"c3 20 {s} static_step 1 0 u7 main {n}")
            ev.append(f"c3 20 {s} static_step 1 0 u7 main {n}")
            ev.append(f"c4 5 {s} static_step 1 0 u7 main {n}")
    a_tr, a_ev = anchor("F")
    return tr + a_tr, ev + a_ev


def batch_g():
    tr, ev = [], []
    for chan in ("main", "uma35", "rma2", "rma35"):
        for s in SEEDS:
            tr.append(f"c3 5 {s} static_step 1 0 u7 {chan} 12")
            ev.append(f"c3 5 {s} static_step 1 0 u7 {chan} 12")
            ev.append(f"c4 5 {s} static_step 1 0 u7 {chan} 12")
    a_tr, a_ev = anchor("G")
    return tr + a_tr, ev + a_ev


def batch_h():
    """增补批 H · 2025 规模对齐（11 §十三，时间豁免）。
    H1 聚集密档：n100 × crowd_step（用户×4、σ 压缩、热点总占比 0.90）
    H2 机队档：n200 密度不变；锚点格 anchorH 同构。"""
    tr, ev = [], []
    for m in ("c3", "b1"):                                   # H1 聚集密档
        for s in SEEDS:
            tr.append(f"{m} 5 {s} crowd_step 1 0 u7 main 100")
            ev.append(f"{m} 5 {s} crowd_step 1 0 u7 main 100")
    for s in SEEDS:
        ev.append(f"c4 5 {s} crowd_step 1 0 u7 main 100")
    for m in ("c3", "b1"):                                   # H2 机队档 n200
        for s in SEEDS:
            tr.append(f"{m} 5 {s} static_step 1 0 u7 main 200")
            ev.append(f"{m} 5 {s} static_step 1 0 u7 main 200")
    for s in SEEDS:
        ev.append(f"c4 5 {s} static_step 1 0 u7 main 200")
    for m in ("c3", "b1"):                                   # H3 综合格（11 §十三追加）
        for s in SEEDS:
            tr.append(f"{m} 5 {s} crowd_step 1 0 u7 main 200")
            ev.append(f"{m} 5 {s} crowd_step 1 0 u7 main 200")
    for s in SEEDS:
        ev.append(f"c4 5 {s} crowd_step 1 0 u7 main 200")
    a_tr, a_ev = anchor("H")
    return tr + a_tr, ev + a_ev


def shard(name: str, lines: list[str]) -> None:
    """静态三节点分片（云A/云B/本地 40/40/20）。"""
    n = len(lines)
    a = (n * 2 + 4) // 5
    b = (n * 4 + 4) // 5
    w(f"{name}_node_a.txt", lines[:a])
    w(f"{name}_node_b.txt", lines[a:b])
    w(f"{name}_node_local.txt", lines[b:])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--with-fg", action="store_true")
    ap.add_argument("--with-h", action="store_true")
    a = ap.parse_args()
    for name, gen in (("B", batch_b), ("C", batch_c), ("D", batch_d),
                      ("E", batch_e)):
        tr, ev = gen()
        shard(f"{name}_train", tr)
        shard(f"{name}_eval", ev)
        w(f"{name}_train_all.txt", tr)
        w(f"{name}_eval_all.txt", ev)
    if a.with_fg:
        for name, gen in (("F", batch_f), ("G", batch_g)):
            tr, ev = gen()
            shard(f"{name}_train", tr)
            shard(f"{name}_eval", ev)
            w(f"{name}_train_all.txt", tr)
            w(f"{name}_eval_all.txt", ev)
    if a.with_h:
        tr, ev = batch_h()
        shard("H_train", tr)
        shard("H_eval", ev)
        w("H_train_all.txt", tr)
        w("H_eval_all.txt", ev)
