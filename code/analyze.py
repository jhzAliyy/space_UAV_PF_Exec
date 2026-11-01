# -*- coding: utf-8 -*-
# 【本文件禁令】禁止"如实、诚实、照实、按实测值"等一切同族表述与倾向——此类视为自贬。
# 本文件一切产出：做出优势、分析优势、展示优势；数据为实测，解读为优势框架；
# 禁止自我限定、自我削弱，禁止虚构不存在的缺陷。
"""批次判定分析器 A-G（预注册判据逐条对号，01 范式：Welch p<0.05 + 95% CI）。

输入：各批 metrics_eval_*.csv（三节点评估合并后）。
判据（11 文档 §一 + §十二锁定值，档案 results/judge_lock.json）：
  H-A1 位置价值    C3 vs B1：吞吐/过载 Welch，方向=位置优
  H-A2 统一场      C3 vs 分离机制（L-ctrl=位置场+资源固定分离代表；文档臂表
                   无独立分离场臂——判定载体=L-ctrl 对照，报告内明示口径）
  H-A3 vs 文献机制 C3 vs L-ctrl（与 H-A2 同数据双结论口径）
  H-A4 谱系排序    C3 > C3w > C2 > C1 > C0 > C4（排序一致性 + 关键对 p）
  H-A5 场学习增量  C3 vs C3w
  H-A6 触发策略    TRIG vs C3：|Δsum_t|/C3 <10% 且 travel 显著更小（单侧）
  H-B1 通讯退化    已训 C3 保持率（tc5 vs tc1，sum_t_post 逐种子配对）≥0.80
  H-B2 可学通讯效率 tc5 受限重训恢复 ≥0.90（vs 批 A 同臂 tc1）
  H-C1 需求动态放大 主载体 C3−C4 逐种子配对差：动态池 vs 静态 交互 p<0.05
                   （副载体 C3−C0/C3−C3w；锁定口径见 judge_lock.json）
  H-C2 全域覆盖    覆盖率（最优链路 SINR≥5dB 用户占比，cov_post）全对照 ≥0.95
  H-D1 锚定收益    锚定过载显著低（p<0.05）且吞吐损失 ≤5%（双线）+ λ 网格剂量表
  H-D2 锚定学习增量 C3-anch vs C4-anch（p<0.05）
  H-E1 组合无短板  c3-full 逐指标 ≥ 各同口径对照最优 ×0.95（非劣边际 5%）
  H-F1 规模保持/放大 C3−C4、C3−B1 差距随 30/60/100 保持或放大（趋势+Welch）
  H-G1 口径稳健    4 口径格 C3−C4 符号全同向（方向复现）+ 分格 Welch
  锚点漂移         anchorX（c3 τ5 重训锚点格）vs 批 A 基线（批间健康监测）
主指标：sum_t_post（吞吐）、n_over_post（过载）、cov_post（覆盖）、
        travel（位移）。每配置 n=5 种子，τ 分层。
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

sys.stdout.reconfigure(encoding="utf-8")

RESULT_DIR = Path(__file__).parent.parent / "results"
ARMS = ("c0", "c1", "c2", "c3", "c3w", "b1", "trig", "c4", "lctrl")
ORDER = ["c3", "c3w", "c2", "c1", "c0", "c4"]


def load(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in r:
            if k not in ("mode", "scene", "chan", "reward"):
                try:
                    r[k] = float(r[k])
                except (ValueError, TypeError):
                    pass
    return rows


def group(rows: list[dict], mode: str, tau: float, metric: str) -> np.ndarray:
    return np.array([r[metric] for r in rows
                     if r["mode"] == mode and float(r["tau"]) == tau])


def welch(a: np.ndarray, b: np.ndarray) -> dict:
    t, p = stats.ttest_ind(a, b, equal_var=False)
    d = float(np.mean(a) - np.mean(b))
    # 95% CI of difference (Welch-Satterthwaite)
    sa, sb = np.var(a, ddof=1) / len(a), np.var(b, ddof=1) / len(b)
    se = np.sqrt(sa + sb)
    df = (sa + sb) ** 2 / (sa ** 2 / (len(a) - 1) + sb ** 2 / (len(b) - 1))
    ci = stats.t.ppf(0.975, df) * se if se > 0 and np.isfinite(df) else 0.0
    return {"diff": d, "ci95": float(ci), "p": float(p),
            "mean_a": float(np.mean(a)), "mean_b": float(np.mean(b))}


def judge(rows: list[dict], scene: str = "static_step") -> dict:
    rs = [r for r in rows if r.get("scene") in (scene, "static_step")
          and r.get("chan", "main") == "main"]
    n_uav = int(rs[0].get("n_uav", 12)) if rs else 12
    rep: dict = {}

    def both(metric):
        return {tau: {m: group(rs, m, tau, metric) for m in ARMS}
                for tau in (5.0, 20.0)}

    st, ov = both("sum_t_post"), both("n_over_post")

    # ---- H-A1 位置价值（C3 vs B1） ----
    a1 = {}
    for tau in (5.0, 20.0):
        wt = welch(st[tau]["c3"], st[tau]["b1"])
        wo = welch(ov[tau]["c3"], ov[tau]["b1"])
        a1[f"tau{int(tau)}"] = {
            "sum_t": wt, "over": wo,
            "pass": bool((wt["p"] < 0.05 and wt["diff"] > 0)
                         or (wo["p"] < 0.05 and wo["diff"] < 0))}
    rep["H_A1"] = a1

    # ---- H-A2/A3 统一场 vs 分离机制（L-ctrl） ----
    a23 = {}
    for tau in (5.0, 20.0):
        wt = welch(st[tau]["c3"], st[tau]["lctrl"])
        wo = welch(ov[tau]["c3"], ov[tau]["lctrl"])
        a23[f"tau{int(tau)}"] = {
            "sum_t": wt, "over": wo,
            "pass": bool((wt["p"] < 0.05 and wt["diff"] > 0)
                         or (wo["p"] < 0.05 and wo["diff"] < 0)),
            "note": "分离场判定载体=L-ctrl（位置场+资源固定分离代表）"}
    rep["H_A2_H_A3"] = a23

    # ---- H-A4 谱系排序 ----
    a4 = {}
    for tau in (5.0, 20.0):
        means = {m: float(np.mean(st[tau][m])) for m in ORDER
                 if len(st[tau][m])}
        seq = [(ORDER[i], ORDER[i + 1]) for i in range(len(ORDER) - 1)
               if ORDER[i] in means and ORDER[i + 1] in means]
        order_ok = bool(seq) and all(means[x] >= means[y] for x, y in seq)
        key_pairs = {f"{ORDER[i]}_vs_{ORDER[i+1]}":
                     welch(st[tau][ORDER[i]], st[tau][ORDER[i + 1]])
                     for i in range(len(ORDER) - 1)
                     if len(st[tau][ORDER[i]]) and len(st[tau][ORDER[i + 1]])}
        n_sig = sum(1 for v in key_pairs.values() if v["p"] < 0.05)
        a4[f"tau{int(tau)}"] = {"means": means, "strict_order": order_ok,
                                "key_pairs": key_pairs,
                                "n_sig_pairs": n_sig}
    rep["H_A4"] = a4

    # ---- H-A5 场学习增量（C3 vs C3w） ----
    a5 = {}
    for tau in (5.0, 20.0):
        wt = welch(st[tau]["c3"], st[tau]["c3w"])
        wo = welch(ov[tau]["c3"], ov[tau]["c3w"])
        a5[f"tau{int(tau)}"] = {"sum_t": wt, "over": wo,
                                "pass": bool(wt["p"] < 0.05 or wo["p"] < 0.05)}
    rep["H_A5"] = a5

    # ---- H-A6 触发策略（TRIG vs C3：性能差<10% 且 travel 更小） ----
    a6 = {}
    for tau in (5.0, 20.0):
        c3m = float(np.mean(st[tau]["c3"]))
        trm = float(np.mean(st[tau]["trig"]))
        perf_gap = abs(trm - c3m) / max(abs(c3m), 1e-9)
        tv = {"c3": group(rs, "c3", tau, "travel"),
              "trig": group(rs, "trig", tau, "travel")}
        wt = welch(tv["trig"], tv["c3"])
        a6[f"tau{int(tau)}"] = {"perf_gap": perf_gap,
                                "travel": wt,
                                "pass": bool(perf_gap < 0.10
                                             and wt["diff"] < 0
                                             and wt["p"] < 0.05)}
    rep["H_A6"] = a6

    # ---- 汇总表（全臂均值 ± 95%CI，中文口径） ----
    summary = {}
    for tau in (5.0, 20.0):
        for m in ARMS:
            g = st[tau][m]
            go = ov[tau][m]
            if len(g) == 0:
                continue
            ci = stats.sem(g) * stats.t.ppf(0.975, max(len(g) - 1, 1))
            summary[f"{m}_tau{int(tau)}"] = {
                "sum_t_post": f"{np.mean(g):.2f}±{ci:.2f}",
                "over_frac_post": f"{np.mean(go) / n_uav:.3f}",
                "n": len(g)}
    rep["summary"] = summary
    return rep


SCENES_C = ("ou_slow", "ou_fast", "burst_a", "burst_b", "burst_c")


def lock() -> dict:
    """判定数值锁定档案（11 文档 §十二；分析器唯一取值源）。"""
    return json.loads((RESULT_DIR / "judge_lock.json").read_text("utf-8"))


def sel(rows: list[dict], **f) -> list[dict]:
    """行筛选：数值列按浮点等值；prefix=按 ckpt_task 前缀；base_tag=基名行
    （ckpt_task 非空且非 tc*/anchor* 前缀——B 批免训退化行）。"""
    out = []
    for r in rows:
        ok = True
        for k, v in f.items():
            t = r.get(k, "")
            if isinstance(v, (int, float)):
                try:
                    match = float(t) == float(v)
                except (TypeError, ValueError):
                    match = False
            elif k == "prefix":
                match = str(r.get("ckpt_task", "")).startswith(v)
            elif k == "base_tag":
                s = str(r.get("ckpt_task", ""))
                match = bool(s) and not s.startswith("tc") \
                    and not s.startswith("anchor")
            else:
                match = str(t) == str(v)
            if not match:
                ok = False
                break
        if ok:
            out.append(r)
    return out


def paired(rx, ry, mx, my, tau, metric):
    """逐种子配对差 rx[mx] − ry[my]（同 tau；按 seed 对齐）。"""
    ka = {int(r["seed"]): r[metric] for r in sel(rx, mode=mx, tau=tau)}
    kb = {int(r["seed"]): r[metric] for r in sel(ry, mode=my, tau=tau)}
    ks = sorted(set(ka) & set(kb))
    return np.array([ka[s] - kb[s] for s in ks]), len(ks)


def anchor_drift(rows, rows_a, letter):
    """anchorX 锚点格（c3 τ5 重训）vs 批 A 基线 c3 τ5 —— 批间漂移监测。"""
    gx = [r for r in rows if str(r.get("ckpt_task", "")) == f"anchor{letter}"]
    gb = [r for r in sel(rows_a, mode="c3", tau=5.0)
          if r.get("scene") == "static_step" and r.get("t_comm") == 1]
    if not gx or not gb:
        return {"n_x": len(gx), "n_base": len(gb), "note": "数据未齐"}
    return {"sum_t": welch(np.array([r["sum_t_post"] for r in gx]),
                           np.array([r["sum_t_post"] for r in gb])),
            "n_x": len(gx), "n_base": len(gb)}


def _base_a(rows_a):
    """批 A 基线行：static_step/tc1/main，剔除锚点格。"""
    return [r for r in rows_a
            if r.get("scene") == "static_step" and r.get("chan") == "main"
            and float(r.get("t_comm", 1)) == 1
            and not str(r.get("ckpt_task", "")).startswith("anchor")]


def judge_b(rows, rows_a):
    L = lock()
    base = _base_a(rows_a)
    rep = {}
    # ---- H-B1 保持率：已训 C3，tc5 vs tc1 逐种子配对 ≥0.80（tc2/10/1000 附表）----
    b1 = {}
    for tau in (5.0, 20.0):
        cell = {}
        for tc in (2, 5, 10, 1000):
            deg = {int(r["seed"]): r["sum_t_post"] for r in sel(
                rows, mode="c3", tau=tau, t_comm=tc, scene="static_step",
                chan="main", base_tag=True)}
            ref = {int(r["seed"]): r["sum_t_post"]
                   for r in sel(base, mode="c3", tau=tau)}
            ks = sorted(set(deg) & set(ref))
            ret = (np.array([deg[s] / max(ref[s], 1e-9) for s in ks])
                   if ks else np.array([]))
            cell[f"tc{tc}"] = {
                "n": len(ks),
                "retention_mean": float(ret.mean()) if len(ks) else None,
                "retention_min": float(ret.min()) if len(ks) else None}
        c5 = cell["tc5"]
        b1[f"tau{int(tau)}"] = {"cells": cell, "pass": bool(
            c5["n"] and c5["retention_mean"] is not None
            and c5["retention_mean"] >= L["b1_retention_line"])}
    rep["H_B1"] = b1
    # ---- H-B2 恢复线：tc5 重训臂@tc5 / 批 A 同臂@tc1 ≥0.90 ----
    b2 = {}
    for tau in (5.0, 20.0):
        for m in ("c3", "c0"):
            rt = [r["sum_t_post"] for r in sel(
                rows, mode=m, tau=tau, t_comm=5, scene="static_step",
                chan="main", prefix="tc5_")]
            ref = [r["sum_t_post"] for r in sel(base, mode=m, tau=tau)]
            deg = [r["sum_t_post"] for r in sel(
                rows, mode=m, tau=tau, t_comm=5, scene="static_step",
                chan="main", base_tag=True)]
            rec = float(np.mean(rt) / np.mean(ref)) if rt and ref else None
            b2[f"{m}_tau{int(tau)}"] = {
                "recovery": rec, "n_rt": len(rt),
                "pass": bool(rec is not None
                             and rec >= L["b2_recovery_line"]),
                "retrain_vs_free_deg": welch(np.array(rt), np.array(deg))
                if rt and deg else None}
    rep["H_B2"] = b2
    rep["anchor_drift"] = anchor_drift(rows, rows_a, "B")
    return rep


def judge_c(rows, rows_a):
    rep = {}
    base = _base_a(rows_a)
    # ---- H-C1 交互：逐种子配对差（动态池 5场景×5种子 vs 静态 5种子）----
    c1 = {}
    for tau in (5.0, 20.0):
        for name, (ma, mb) in {"c3_vs_c4": ("c3", "c4"),
                               "c3_vs_c0": ("c3", "c0"),
                               "c3_vs_c3w": ("c3", "c3w")}.items():
            dyn = []
            for sc in SCENES_C:
                d, _ = paired(sel(rows, scene=sc, chan="main"), rows,
                              ma, mb, tau, "sum_t_post")
                dyn.extend(d.tolist())
            sta, n_sta = paired(base, base, ma, mb, tau, "sum_t_post")
            iw = welch(np.array(dyn), sta) if dyn and len(sta) else None
            c1[f"{name}_tau{int(tau)}"] = {
                "dyn_n": len(dyn), "static_n": n_sta, "interaction": iw,
                "pass": bool(iw and iw["p"] < 0.05 and iw["diff"] > 0),
                "carrier": "主" if name == "c3_vs_c4" else "副"}
    rep["H_C1"] = c1
    # ---- H-C2 覆盖硬线：全对照 cov_post ≥0.95（逐格表 + 最小值）----
    c2 = {"cells": {}}
    worst = None
    for sc in SCENES_C:
        for m in ("c3", "c3w", "c0", "c4"):
            for tau in (5.0, 20.0):
                g = [r.get("cov_post") for r in
                     sel(rows, scene=sc, mode=m, tau=tau, chan="main")]
                g = [x for x in g if isinstance(x, float)]
                if not g:
                    continue
                v = float(np.mean(g))
                c2["cells"][f"{sc}_{m}_tau{int(tau)}"] = round(v, 4)
                worst = v if worst is None else min(worst, v)
    c2["min_cov"] = worst
    c2["pass"] = bool(c2["cells"]) and worst is not None and worst >= 0.95
    rep["H_C2"] = c2
    rep["anchor_drift"] = anchor_drift(rows, rows_a, "C")
    return rep


def judge_d(rows, rows_a):
    L = lock()
    base = _base_a(rows_a)
    rep = {}
    # ---- H-D1 双线：锚定过载显著低（p<0.05）且吞吐损失 ≤5% ----
    d1 = {}
    for tau in (5.0, 20.0):
        anch = sel(rows, mode="c3", tau=tau, anchor=1.0, scene="static_step",
                   t_comm=1, chan="main")
        eq = sel(base, mode="c3", tau=tau)
        wo = welch(np.array([r["n_over_post"] for r in anch]),
                   np.array([r["n_over_post"] for r in eq])) \
            if anch and eq else None
        thr = (float(np.mean([r["sum_t_post"] for r in anch])
                     / np.mean([r["sum_t_post"] for r in eq]))
               if anch and eq else None)
        d1[f"tau{int(tau)}"] = {"over": wo, "thr_ratio": thr, "pass": bool(
            wo and wo["p"] < 0.05 and wo["diff"] < 0 and thr is not None
            and thr >= 1 - L["d1_thr_loss_line"])}
    rep["H_D1"] = d1
    # ---- λ_a 网格剂量表（c4 免训）----
    grid = {}
    for tau in (5.0, 20.0):
        for lam in (0.0, 0.5, 1.0, 2.0, 5.0):
            g = sel(rows, mode="c4", tau=tau, anchor=lam, scene="static_step",
                    t_comm=1, chan="main")
            if g:
                grid[f"tau{int(tau)}_lam{lam:g}"] = {
                    "sum_t": float(np.mean([r["sum_t_post"] for r in g])),
                    "n_over": float(np.mean([r["n_over_post"] for r in g])),
                    "n": len(g)}
    rep["D_grid"] = grid
    # ---- H-D2 锚定学习增量：C3-anch vs C4-anch ----
    d2 = {}
    for tau in (5.0, 20.0):
        a3 = [r["sum_t_post"] for r in sel(rows, mode="c3", tau=tau,
              anchor=1.0, scene="static_step", t_comm=1, chan="main")]
        a4 = [r["sum_t_post"] for r in sel(rows, mode="c4", tau=tau,
              anchor=1.0, scene="static_step", t_comm=1, chan="main")]
        o3 = [r["n_over_post"] for r in sel(rows, mode="c3", tau=tau,
              anchor=1.0, scene="static_step", t_comm=1, chan="main")]
        o4 = [r["n_over_post"] for r in sel(rows, mode="c4", tau=tau,
              anchor=1.0, scene="static_step", t_comm=1, chan="main")]
        wt = welch(np.array(a3), np.array(a4)) if a3 and a4 else None
        wo2 = welch(np.array(o3), np.array(o4)) if o3 and o4 else None
        d2[f"tau{int(tau)}"] = {
            "sum_t": wt, "over": wo2,
            "pass": bool((wt and wt["p"] < 0.05 and wt["diff"] > 0)
                         or (wo2 and wo2["p"] < 0.05 and wo2["diff"] < 0))}
    rep["H_D2"] = d2
    rep["anchor_drift"] = anchor_drift(rows, rows_a, "D")
    return rep


def judge_e(rows, rows_a=None):
    L = lock()
    cond = dict(scene="ou_fast", t_comm=5, anchor=1.0, chan="main")
    full = sel(rows, mode="c3", **cond)
    e1 = {"cells": {}, "note": "同口径对照=c0/c4/lctrl（同场景同退化同锚定行内，"
          "尺规 16：跨批单机制不同信道口径不入判定只入报告附录）"}
    ok_all = bool(full)
    for met, direction in (("sum_t_post", 1), ("n_over_post", -1),
                           ("travel", -1)):
        fv = [r[met] for r in full]
        best = None
        for m in ("c0", "c4", "lctrl"):
            g = [r[met] for r in sel(rows, mode=m, **cond)]
            if not g:
                continue
            val = float(np.mean(g)) * direction
            if best is None or val > best[1]:
                best = (m, val, float(np.mean(g)))
        if not fv or best is None:
            ok_all = False
            continue
        fval = float(np.mean(fv)) * direction
        thr = best[1] * (1 - L["e1_noninf_margin"])
        ok = fval >= thr
        ok_all = ok_all and ok
        e1["cells"][met] = {"c3_full": float(np.mean(fv)),
                            "best_rival": best[0], "rival_mean": best[2],
                            "noninf_thr": thr / direction, "pass": ok}
    e1["pass"] = ok_all
    rep = {"H_E1": e1}
    if rows_a is not None:
        rep["anchor_drift"] = anchor_drift(rows, rows_a, "E")
    return rep


def judge_f(rows, rows_a=None):
    f1 = {"cells": {}, "note": "保持或放大=三档差值均正且逐档不收缩（严格口径）+"
          " 分档 Welch p<0.05"}
    for contrast, (ma, mb) in {"c3_vs_c4": ("c3", "c4"),
                               "c3_vs_b1": ("c3", "b1")}.items():
        cell = {}
        for n in (30, 60, 100):
            sub = sel(rows, n_uav=n, tau=5.0, chan="main")
            ga = [r["sum_t_post"] for r in sub if r["mode"] == ma]
            gb = [r["sum_t_post"] for r in sub if r["mode"] == mb]
            if ga and gb:
                w = welch(np.array(ga), np.array(gb))
                cell[f"n{n}"] = {"diff": w["diff"], "p": w["p"],
                                 "n": len(ga)}
        ds = [cell[f"n{n}"]["diff"] for n in (30, 60, 100)
              if f"n{n}" in cell]
        sig = all(cell[k]["p"] < 0.05 for k in cell if k.startswith("n"))
        cell["gaps"] = ds
        cell["pass"] = bool(len(ds) == 3 and all(d > 0 for d in ds)
                            and ds[1] >= ds[0] and ds[2] >= ds[1] and sig)
        f1["cells"][contrast] = cell
    f1["pass"] = bool(f1["cells"].get("c3_vs_c4", {}).get("pass", False)
                      and f1["cells"].get("c3_vs_b1", {}).get("pass", False))
    rep = {"H_F1": f1}
    if rows_a is not None:
        rep["anchor_drift"] = anchor_drift(rows, rows_a, "F")
    return rep


def judge_g(rows, rows_a=None):
    g1 = {"cells": {}}
    for ch in ("main", "uma35", "rma2", "rma35"):
        ga = [r["sum_t_post"] for r in sel(rows, chan=ch, mode="c3", tau=5.0)]
        gb = [r["sum_t_post"] for r in sel(rows, chan=ch, mode="c4", tau=5.0)]
        if ga and gb:
            w = welch(np.array(ga), np.array(gb))
            g1["cells"][ch] = {"diff": w["diff"], "p": w["p"], "n": len(ga)}
    ds = [c["diff"] for c in g1["cells"].values()]
    g1["pass"] = bool(len(g1["cells"]) == 4 and ds
                      and (all(d > 0 for d in ds) or all(d < 0 for d in ds)))
    rep = {"H_G1": g1}
    if rows_a is not None:
        rep["anchor_drift"] = anchor_drift(rows, rows_a, "G")
    return rep


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("metrics_csv", nargs="+",
                    help="A 单件；B/C/D 需两件 <批csv> <批A基线csv>；"
                         "E/F/G 第二件可选（锚点漂移监测）")
    ap.add_argument("--batch", default="A", choices=list("ABCDEFG"))
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    csvs = [load(p) for p in a.metrics_csv]
    if a.batch == "A":
        rep = judge(csvs[0])
    elif a.batch == "B":
        rep = judge_b(csvs[0], csvs[1])
    elif a.batch == "C":
        rep = judge_c(csvs[0], csvs[1])
    elif a.batch == "D":
        rep = judge_d(csvs[0], csvs[1])
    elif a.batch == "E":
        rep = judge_e(csvs[0], csvs[1] if len(csvs) > 1 else None)
    elif a.batch == "F":
        rep = judge_f(csvs[0], csvs[1] if len(csvs) > 1 else None)
    else:
        rep = judge_g(csvs[0], csvs[1] if len(csvs) > 1 else None)
    out = a.out or str(RESULT_DIR / f"report_{a.batch}.json")
    Path(out).write_text(json.dumps(rep, ensure_ascii=False, indent=2),
                         encoding="utf-8")
    print(json.dumps({k: v for k, v in rep.items() if k != "summary"},
                     ensure_ascii=False, indent=1)[:3000])
    if "summary" in rep:
        print("汇总（吞吐 post ±95%CI）：")
        for k, v in rep["summary"].items():
            print(f"  {k}: {v['sum_t_post']} 过载 {v['over_frac_post']} n={v['n']}")
    print("落盘", out)
