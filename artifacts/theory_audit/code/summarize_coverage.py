# -*- coding: utf-8 -*-
"""summarize_coverage.py — 汇总 coverage_n3_n100.csv。只保留星号标记。"""
from __future__ import annotations

import csv
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "artifacts" / "theory_audit" / "out"
SRC = OUT / "coverage_n3_n100.csv"
if (OUT / "coverage_n3_n100_with_extrapolated.csv").exists():
    SRC = OUT / "coverage_n3_n100_with_extrapolated.csv"


def f(r, k):
    try:
        return float(r.get(k, "nan"))
    except (TypeError, ValueError):
        return float("nan")


def med(vals):
    vals = [v for v in vals if v == v]
    return st.median(vals) if vals else float("nan")


def main():
    rows = list(csv.DictReader(SRC.open(encoding="utf-8")))
    groups = {}
    for r in rows:
        groups.setdefault((int(r["n"]), r["scene"]), []).append(r)
    out_rows = []
    print("n scene cells amax_med aent_med frac<0.99 rho_a700 rho_a1500 in_region u_mean dead_max")
    for (n, scene), rs in sorted(groups.items()):
        n_star = sum(r.get("mark") == "*" for r in rs)
        row = dict(
            n=n, scene=scene, cells=len(rs), star=n_star,
            cells_mark=f"{len(rs)}*" if n_star else str(len(rs)),
            amax_median=med([f(r, "amax_median") for r in rs]),
            aent_median=med([f(r, "a_entropy_median") for r in rs]),
            frac_amax_lt_099=med([f(r, "frac_amax_lt_099") for r in rs]),
            frac_amax_lt_0999=med([f(r, "frac_amax_lt_0999") for r in rs]),
            rho_a_700=med([f(r, "rho_a_700") for r in rs]),
            rho_a_1500=med([f(r, "rho_a_1500") for r in rs]),
            in_region_cells=sum(
                1 for r in rs
                if str(r.get("in_region_700")).lower() == "true"
                or str(r.get("in_region_1500")).lower() == "true"),
            u_mean=med([f(r, "u_mean") for r in rs]),
            n_dead_max=max(int(float(r.get("n_dead_max", 0) or 0)) for r in rs),
        )
        out_rows.append(row)
        print(f"{n:3d} {scene:5s} {row['cells_mark']:>4s} "
              f"{row['amax_median']:.9f} {row['aent_median']:.4f} "
              f"{row['frac_amax_lt_099']:.4f} {row['rho_a_700']:.1f} "
              f"{row['rho_a_1500']:.1f} {row['in_region_cells']}/{len(rs)} "
              f"{row['u_mean']:.3f} {row['n_dead_max']}")
    dst = OUT / "coverage_summary_n3_n100.csv"
    with dst.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)
    print(f"-> {dst}")

    total = len(rows)
    in_region = sum(
        1 for r in rows
        if str(r.get("in_region_700")).lower() == "true"
        or str(r.get("in_region_1500")).lower() == "true")
    n_star = sum(r.get("mark") == "*" for r in rows)
    print(f"overall: cells={total} star={n_star} in_region={in_region}/{total} "
          f"frac_steps_amax<0.99 median={med([f(r,'frac_amax_lt_099') for r in rows]):.4f} "
          f"(* internal)")


if __name__ == "__main__":
    main()
