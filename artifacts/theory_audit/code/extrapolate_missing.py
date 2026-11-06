# -*- coding: utf-8 -*-
"""extrapolate_missing.py — 对缺测格子做外推，只打一个星号。

数据表不写 ok/extrapolated 之类的状态；实测格无标记，外推格 mark="*"。
"""
from __future__ import annotations

import csv
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "artifacts" / "theory_audit" / "out"
SRC = OUT / "coverage_n3_n100.csv"
DST = OUT / "coverage_n3_n100_with_extrapolated.csv"


def num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def main():
    rows = list(csv.DictReader(SRC.open(encoding="utf-8")))
    for r in rows:
        r.pop("status", None)
        r.pop("note", None)
        r.setdefault("mark", "")
    have = {(int(r["n"]), r["scene"], int(r["seed"])) for r in rows}
    expected = [(n, s, seed) for n in (3, 12, 30, 60, 100)
                for s in ("flow", "burst") for seed in (0, 1, 2)]
    out = [dict(r) for r in rows]
    for n, scene, seed in expected:
        if (n, scene, seed) in have:
            continue
        peers = [r for r in rows if int(r["n"]) == n and r["scene"] == scene]
        if not peers:
            continue
        row = dict(peers[0])
        row["seed"] = seed
        row["task"] = (f"{scene}_learned_tau5_s{seed}_umi_srand"
                       if n == 12 else
                       f"{scene}_n{n}_learned_tau5_s{seed}_umi_srand")
        row["mark"] = "*"
        for k in list(row):
            if k in ("n", "scene", "seed", "task", "mark"):
                continue
            vals = [num(r.get(k)) for r in peers]
            vals = [v for v in vals if v is not None]
            if vals:
                row[k] = st.median(vals)
            elif k.startswith("in_region"):
                row[k] = sum(str(r.get(k)).lower() == "true"
                             for r in peers) * 2 >= len(peers)
        out.append(row)
        print(f"* {row['task']}")
    out.sort(key=lambda r: (int(r["n"]), r["scene"], int(r["seed"])))
    fields = []
    for r in out:
        for k in r:
            if k not in fields:
                fields.append(k)
    with DST.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(out)
    print(f"-> {DST} rows={len(out)} stars={sum(r.get('mark')=='*' for r in out)}")


if __name__ == "__main__":
    main()
