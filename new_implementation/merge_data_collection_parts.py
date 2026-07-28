#!/usr/bin/env python3
"""Merge disjoint data-collection shards into the usual single CSV outputs."""

import argparse
from pathlib import Path

import pandas as pd


def merge_parts(folder: Path, stem: str, sort_columns: list[str]) -> None:
    parts = sorted(folder.glob(f"{stem}.part-*.csv"))
    if not parts:
        raise FileNotFoundError(f"No parts found for {folder / stem}")

    frames = [pd.read_csv(part) for part in parts]
    merged = pd.concat(frames, ignore_index=True).sort_values(sort_columns).reset_index(drop=True)
    output = folder / f"{stem}.csv"
    merged.to_csv(output, index=False)
    print(f"Merged {len(parts)} parts ({len(merged)} rows) into {output}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("budget", type=int)
    parser.add_argument("algorithm")
    parser.add_argument("dimension", type=int)
    args = parser.parse_args()

    root = Path("data") / f"dim_{args.dimension}"
    tag = f"{args.algorithm}_B{args.budget}_{args.dimension}D"
    merge_parts(root / "ela_features" / tag, "ELA_features", ["fid", "iid", "rep", "ela_budget"])
    merge_parts(root / "achieved_regrets", f"achieved_regrets_{tag}", ["fid", "iid", "rep"])
    merge_parts(root / "achieved_aucs", f"achieved_aucs_{tag}", ["fid", "iid", "rep"])


if __name__ == "__main__":
    # main()
    df = pd.read_csv("data/dim_40/ela_features/Non-elitist_B1000_40D/ELA_features.csv")
    # drop the column ela_meta.quad_simple.cond if it exists
    df = df.drop(columns=["ela_meta.quad_simple.cond"], errors="ignore")
    df.to_csv("data/dim_40/ela_features/Non-elitist_B1000_40D/ELA_features.csv", index=False)