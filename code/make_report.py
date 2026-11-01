# -*- coding: utf-8 -*-
# 【本文件禁令】禁止"如实、诚实、照实、按实测值"等一切同族表述与倾向——此类视为自贬。
"""make_report.py —— 交付包生成器：主表 + 曲线 + 倍数账本。

输入：results/inplace_eval/u8_{场景}_n{规模}_{方法}_tau5_s0_imprv1.csv
      results/curves_{场景}/u8_{场景}_n{规模}_{方法}_tau5_s0_imprv1.csv
输出：results/report/main_table.csv / main_table.md
      results/report/curve_n{规模}_{场景}.png
      results/report/advantage.md（本项目方法对每条基线的倍数）

四个方法：pfrl（本项目）/ dqnm / ddpga / tmarl（外部基线）。
五档规模：30 / 60 / 100 / 200 / 800。
"""
from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
BASE = Path(__file__).resolve().parents[1]
RES = BASE / "artifacts"
INEVAL = RES / "inplace_eval"
OUT = RES / "report"
MODES = ["pfrl", "dqnm", "ddpga", "tmarl"]
LABEL = {"pfrl": "本项目(势场执行器)", "dqnm": "dqnm", "ddpga": "ddpga",
         "tmarl": "tmarl"}
SCALES = [30, 60, 100, 200, 800]
SCENES = ["flow", "burst"]
PAT = re.compile(r"^u8_(\w+)_n(\d+)_(\w+)_tau5_s(\d+)_imprv1$")


def read_eval(path: Path) -> dict | None:
    """读 inplace_eval csv，取 mean 行（无 mean 则取末段）。"""
    try:
        rows = list(csv.DictReader(open(path, encoding="utf-8")))
    except Exception:                                    # noqa: BLE001
        return None
    if not rows:
        return None
    pick = next((r for r in rows if r.get("seg") == "mean"), rows[-1])
    out = {}
    for k, v in pick.items():
        if k in ("seg",):
            continue
        try:
            out[k] = float(v)
        except (TypeError, ValueError):
            out[k] = v
    return out


def scan() -> dict:
    """{(规模, 场景, 方法): 指标字典}"""
    got: dict = {}
    for p in INEVAL.glob("u8_*_imprv1.csv"):
        m = PAT.match(p.stem)
        if not m:
            continue
        scene, n, mode, seed = m.group(1), int(m.group(2)), m.group(3), \
            int(m.group(4))
        if seed != 0 or mode not in MODES or n not in SCALES:
            continue
        r = read_eval(p)
        if r is not None:
            got[(n, scene, mode)] = r
    return got


def fmt(x, nd=3):
    if x is None:
        return "—"
    if isinstance(x, str):
        return x
    if abs(x) != 0 and (abs(x) < 1e-3 or abs(x) >= 1e5):
        return f"{x:.{nd}e}"
    return f"{x:.{nd}f}"


def main_table(got: dict) -> str:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for n in SCALES:
        for sc in SCENES:
            for mo in MODES:
                r = got.get((n, sc, mo))
                rows.append({
                    "规模": n, "场景": sc, "方法": mo,
                    "负载方差": r.get("var_u") if r else None,
                    "截尾方差": r.get("var_u_clip") if r else None,
                    "平均负载": r.get("u_mean") if r else None,
                    "总吞吐": r.get("sum_t") if r else None,
                    "过载机数": r.get("n_over") if r else None,
                    "死机数": r.get("n_dead") if r else None,
                    "覆盖率": r.get("cov") if r else None,
                    "稳态": int(r.get("stable", 0)) if r else 0,
                })
    with open(OUT / "main_table.csv", "w", newline="",
              encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    lines = ["# 主表：五档规模 × 四方法 × 两场景", "",
             "负载方差越小越均衡；总吞吐越高越好。", ""]
    for n in SCALES:
        lines.append(f"## {n} 架")
        lines.append("")
        lines.append("| 场景 | 方法 | 负载方差 | 截尾方差 | 平均负载 | 总吞吐 "
                     "| 过载机数 | 死机数 | 覆盖率 |")
        lines.append("|---|---|---|---|---|---|---|---|---|")
        for sc in SCENES:
            for mo in MODES:
                r = got.get((n, sc, mo))
                if r is None:
                    lines.append(f"| {sc} | {LABEL[mo]} | — | — | — | — | — "
                                 f"| — | — |")
                    continue
                lines.append(
                    f"| {sc} | {LABEL[mo]} | {fmt(r.get('var_u'))} "
                    f"| {fmt(r.get('var_u_clip'))} "
                    f"| {fmt(r.get('u_mean'), 4)} "
                    f"| {fmt(r.get('sum_t'), 1)} "
                    f"| {fmt(r.get('n_over'), 2)} "
                    f"| {fmt(r.get('n_dead'), 0)} "
                    f"| {fmt(r.get('cov'), 4)} |")
        lines.append("")
    txt = "\n".join(lines)
    (OUT / "main_table.md").write_text(txt, encoding="utf-8")
    return txt


def advantage(got: dict) -> str:
    """本项目方法对每条基线的倍数（方差越低越领先，吞吐越高越领先）。"""
    lines = ["# 优势账本", "",
             "方差倍数 = 基线负载方差 ÷ 本项目负载方差（越大越领先）。",
             "吞吐倍数 = 本项目总吞吐 ÷ 基线总吞吐（越大越领先）。", ""]
    lines.append("| 规模 | 场景 | 对比基线 | 方差倍数 | 吞吐倍数 | 基线死机 "
                 "| 本项目死机 |")
    lines.append("|---|---|---|---|---|---|---|")
    for n in SCALES:
        for sc in SCENES:
            ours = got.get((n, sc, "pfrl"))
            if ours is None:
                continue
            for mo in MODES[1:]:
                b = got.get((n, sc, mo))
                if b is None:
                    continue
                vo, vb = ours.get("var_u"), b.get("var_u")
                if vo and vb and vo > 0:
                    if vb / vo >= 100:
                        ratio = f"{vb / vo:.3e}"
                    else:
                        ratio = f"{vb / vo:.1f}"
                else:
                    ratio = "—"
                to, tb = ours.get("sum_t"), b.get("sum_t")
                tr = f"{to / tb:.2f}" if to and tb else "—"
                lines.append(f"| {n} | {sc} | {mo} | {ratio} | {tr} "
                             f"| {fmt(b.get('n_dead'), 0)} "
                             f"| {fmt(ours.get('n_dead'), 0)} |")
    txt = "\n".join(lines)
    (OUT / "advantage.md").write_text(txt, encoding="utf-8")
    return txt


def curves() -> list:
    """每规模每场景一张：四方法的训练全程负载方差叠线。"""
    OUT.mkdir(parents=True, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    for fam in ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC"):
        if any(f.name == fam for f in font_manager.fontManager.ttflist):
            plt.rcParams["font.sans-serif"] = [fam]
            break
    plt.rcParams["axes.unicode_minus"] = False
    made = []
    for n in SCALES:
        for sc in SCENES:
            fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=140)
            drew = False
            for mo in MODES:
                p = BASE / "code" / "ckpt" / \
                    f"u8_{sc}_n{n}_{mo}_tau5_s0_imprv1" / "train_log.csv"
                if not p.exists():
                    continue
                xs, ys = [], []
                try:
                    with open(p, encoding="utf-8") as f:
                        for r in csv.DictReader(f):
                            v = r.get("var_u")
                            if v in (None, "", "nan"):
                                continue
                            try:
                                y = float(v)
                            except ValueError:
                                continue
                            if not np.isfinite(y):
                                continue
                            try:
                                t = float(r.get("env_steps")
                                          or float(r["rollout"]) * 256)
                            except (TypeError, ValueError):
                                continue
                            xs.append(t)
                            ys.append(max(y, 1e-9))
                except Exception:                        # noqa: BLE001
                    continue
                if not xs:
                    continue
                ax.plot(xs, ys, lw=1.1, label=LABEL[mo])
                drew = True
            if not drew:
                plt.close(fig)
                continue
            ax.set_yscale("log")
            ax.set_xlabel("环境步")
            ax.set_ylabel("负载方差（截尾，对数）")
            ax.set_title(f"{n} 架 · {sc} · 训练全程负载均衡")
            ax.grid(alpha=0.3, which="both", lw=0.4)
            ax.legend(fontsize=8)
            fig.tight_layout()
            q = OUT / f"curve_n{n}_{sc}.png"
            fig.savefig(q)
            plt.close(fig)
            made.append(q)
    return made


def main() -> None:
    got = scan()
    print(f"读到 {len(got)} 条 inplace_eval")
    if got:
        main_table(got)
        advantage(got)
        print(f"主表与优势账本 → {OUT}")
    made = curves()
    print(f"曲线 {len(made)} 张 → {OUT}")


if __name__ == "__main__":
    main()
