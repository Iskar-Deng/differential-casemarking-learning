#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import List

def load_and_prepare(run_id: str, result_dir: Path) -> pd.DataFrame:
    csv_path = result_dir / run_id / f"{run_id}_minipair_results.csv"
    df = pd.read_csv(csv_path)
    df["run_id"] = run_id
    return df

def find_available_runs(result_dir: Path) -> List[str]:
    runs = []
    for d in result_dir.iterdir():
        if d.is_dir():
            csv_file = d / f"{d.name}_minipair_results.csv"
            if csv_file.exists():
                runs.append(d.name)
    return sorted(runs)

def plot_curves(dfs, out_file: Path, y_max: float = 1.0):
    plt.figure(figsize=(10, 6))
    for df in dfs:
        label = df["run_id"].iloc[0]
        x = pd.to_numeric(df["step"], errors="coerce")
        x = x.fillna(pd.Series(range(len(df)), index=df.index))
        y = df["accuracy"]
        plt.plot(x.values, y.values, marker="o", label=label)

    plt.xlabel("Step")
    plt.ylabel("Accuracy")
    plt.title("MiniPair Accuracy vs Training Step")
    plt.legend()
    plt.grid(True)
    plt.ylim(0, y_max)

    plt.tight_layout()
    out_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_file, dpi=200)
    print(f"[Info] Figure saved to: {out_file.resolve()}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="*", default=None,
                        help="Run-ids to plot. If not set, auto-detect from result-dir.")
    parser.add_argument("--result-dir", type=Path, default=Path("results"),
                        help="Base result directory.")
    parser.add_argument("--out", type=Path, required=True,
                        help="Output PNG path for the figure.")
    parser.add_argument("--y-max", type=float, default=1.0,
                        help="Max limit for y-axis (accuracy).")

    args = parser.parse_args()

    if args.runs:
        run_ids = args.runs
    else:
        run_ids = find_available_runs(args.result_dir)
        print(f"[Info] Auto-discovered runs: {run_ids}")

    if not run_ids:
        raise RuntimeError("No run-ids found to plot.")

    dfs = []
    for r in run_ids:
        try:
            df = load_and_prepare(r, args.result_dir)
            dfs.append(df)
        except Exception as e:
            print(f"[Warn] Failed to load run '{r}': {e}")

    if not dfs:
        raise RuntimeError("No result CSVs loaded.")

    plot_curves(dfs, args.out, y_max=args.y_max)

if __name__ == "__main__":
    main()
