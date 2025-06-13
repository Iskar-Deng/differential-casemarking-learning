#!/usr/bin/env python3
"""
human_spot_check.py
Randomly sample N lines (default: 50) from each cbt_affected.txt file:

  <proj_root>/data/perturbed/heuristic/cbt_affected.txt
  <proj_root>/data/perturbed/rule/cbt_affected.txt

and write:
  <same dir>/cbt_affected_<N>.txt
"""

import argparse
import os
import random

# ── Project-wide configuration -------------------------------------------------
from utils import DATA_PATH, AGENT_MARK, PATIENT_MARK  # marks unused here


# ── I/O helpers ----------------------------------------------------------------
def read_lines(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return f.readlines()


def write_lines(path: str, lines):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(lines)


def sample_file(src: str, k: int, rng: random.Random) -> None:
    lines = read_lines(src)
    if k > len(lines):
        raise ValueError(f"Requested {k} lines, but {src} contains only {len(lines)}.")
    sampled = rng.sample(lines, k)

    root, ext = os.path.splitext(src)              # ".../cbt_affected"
    dst = f"{root}_{k}{ext}"                       # ".../cbt_affected_50.txt"
    write_lines(dst, sampled)

    print(f"{os.path.relpath(src)} → {os.path.relpath(dst)} ({k} lines)")


# ── Main -----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Sample N lines from each CBT-affected file (heuristic & rule)."
    )
    parser.add_argument("--num_lines", type=int, default=50, help="Lines to sample")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    rng = random.Random(args.seed)

    # 1) where this file lives            …/relational-casemarking-learning/data_processing
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # 2) project root = parent directory  …/relational-casemarking-learning
    proj_root = os.path.abspath(os.path.join(script_dir, os.pardir))
    # 3) full data path                   …/relational-casemarking-learning/data
    data_root = os.path.join(proj_root, DATA_PATH)

    targets = [
        os.path.join(data_root, "perturbed", "heuristic", "cbt_affected.txt"),
        os.path.join(data_root, "perturbed", "rule", "cbt_affected.txt"),
    ]

    for src in targets:
        sample_file(src, args.num_lines, rng)


if __name__ == "__main__":
    main()
