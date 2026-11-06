# -*- coding: utf-8 -*-
"""merge.py — 合并 GitHub Actions 各 job 的 theory_audit CSV。

用法：
    python merge.py <下载目录> <输出csv>
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path


def main():
    src = Path(sys.argv[1])
    dst = Path(sys.argv[2])
    files = sorted(src.rglob("*.csv"))
    rows = []
    fields = None
    for f in files:
        with f.open(newline="", encoding="utf-8") as fh:
            r = csv.DictReader(fh)
            if fields is None:
                fields = r.fieldnames
            rows.extend(r)
    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"merged {len(files)} files, {len(rows)} rows -> {dst}")


if __name__ == "__main__":
    main()
