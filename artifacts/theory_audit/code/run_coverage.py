# -*- coding: utf-8 -*-
"""run_coverage.py — 并行跑 n3~n100 学习执行器覆盖率。

用法：
    python run_coverage.py --workers 4
输出：
    artifacts/theory_audit/out/coverage_n3_n100.csv
"""
from __future__ import annotations

import argparse
import csv
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from coverage_audit import OUT_DIR, run_one, task_name

ROOT = Path(__file__).resolve().parents[3]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--steps", type=int, default=1500)
    args = ap.parse_args()
    tasks = []
    for n in (3, 12, 30, 60, 100):
        for scene in ("flow", "burst"):
            for seed in (0, 1, 2):
                t = task_name(n, scene, seed)
                ck = ROOT / "artifacts" / "experiments" / t / "ckpt" / "latest.pt"
                if ck.exists():
                    tasks.append((n, scene, seed, t))
    print(f"tasks={len(tasks)} workers={args.workers}", flush=True)
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(run_one, n, s, seed, t, args.steps): (n, s, seed, t)
                for n, s, seed, t in tasks}
        for i, fut in enumerate(as_completed(futs), 1):
            n, s, seed, t = futs[fut]
            try:
                row = fut.result()
            except Exception as e:
                print(f"[{i}/{len(tasks)}] {t} error={type(e).__name__}:{e}", flush=True)
                continue
            if row is None:
                print(f"[{i}/{len(tasks)}] {t} skip=missing", flush=True)
                continue
            rows.append(row)
            print(f"[{i}/{len(tasks)}] {t}", flush=True)
    rows.sort(key=lambda r: (r["n"], r["scene"], r["seed"]))
    fields = []
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "coverage_n3_n100.csv"
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"done -> {out} rows={len(rows)}", flush=True)


if __name__ == "__main__":
    main()
