#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path

base = Path("/workspace/differential-casemarking-learning/data/perturbed")

def count_lines(p: Path):
    try:
        with open(p, "r", encoding="utf-8") as f:
            return sum(1 for line in f if line.strip())
    except FileNotFoundError:
        return 0

print(f"{'Dataset':40s}  {'A':>8s}  {'U':>8s}  {'A/U':>8s}  {'A%total':>9s}")
print("-"*80)

for subdir in sorted(base.iterdir()):
    if not subdir.is_dir():
        continue
    a_file = subdir / "train_affected.txt"
    u_file = subdir / "train_unaffected.txt"

    A = count_lines(a_file)
    U = count_lines(u_file)
    total = A + U
    if total == 0:
        continue

    ratio = A / U if U > 0 else float('inf')
    a_pct = 100 * A / total

    print(f"{subdir.name:40s}  {A:8d}  {U:8d}  {ratio:8.3f}  {a_pct:8.3f}%")
