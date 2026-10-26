#!/usr/bin/env python3
"""Merge disjoint data-collection shards into the usual single CSV outputs."""

import argparse
from pathlib import Path

import pandas as pd


def merge_parts(
    folder: Path,
    stem: str,
    sort_columns: list[str],
    timing_outputs: dict[str, str] | None = None,
) -> None:
    parts = sorted(folder.glob(f"{stem}.part-*.csv"))
    if not parts:
        raise FileNotFoundError(f"No parts found for {folder / stem}")

    frames = [pd.read_csv(part) for part in parts]
    merged = pd.concat(frames, ignore_index=True).sort_values(sort_columns).reset_index(drop=True)

    if timing_outputs is not None:
        missing = set(timing_outputs) - set(merged.columns)
        if missing:
            raise ValueError(
                f"Timing columns {sorted(missing)!r} were not found in the part files. "
                "Run data collection with timing tracking enabled."
            )
        for timing_column, output_name in timing_outputs.items():
            (folder / output_name).write_text(f"{merged[timing_column].sum():.9f}\n")
        merged = merged.drop(columns=list(timing_outputs))

    output = folder / f"{stem}.csv"
    merged.to_csv(output, index=False)
    print(f"Merged {len(parts)} parts ({len(merged)} rows) into {output}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("budget", type=int)
    parser.add_argument("algorithm")
    parser.add_argument("dimension", type=int)
    parser.add_argument(
        "--use-ma",
        action="store_true",
        help="Merge shards produced from ManyAffine functions.",
    )
    args = parser.parse_args()

    root = Path("data") / (
        f"dim_{args.dimension}_ma" if args.use_ma else f"dim_{args.dimension}"
    )
    tag = f"{args.algorithm}_B{args.budget}_{args.dimension}D"
    merge_parts(
        root / "ela_features" / tag,
        "ELA_features",
        ["fid", "iid", "rep", "ela_budget"],
        timing_outputs={
            "ela_cpu_time_seconds": "ELA_cpu_time_seconds.txt",
            "ela_wall_time_seconds": "ELA_wall_time_seconds.txt",
        },
    )
    merge_parts(root / "achieved_regrets", f"achieved_regrets_{tag}", ["fid", "iid", "rep"])
    merge_parts(root / "achieved_aucs", f"achieved_aucs_{tag}", ["fid", "iid", "rep"])
    merge_parts(
        root / "internal_state",
        f"internal_state_{tag}",
        ["fid", "iid", "rep", "evaluations", "iteration"],
    )


if __name__ == "__main__":
    main()
    # df = pd.read_csv("data/dim_40/ela_features/Non-elitist_B1000_40D/ELA_features.csv")
    # # drop the column ela_meta.quad_simple.cond if it exists
    # df = df.drop(columns=["ela_meta.quad_simple.cond"], errors="ignore")
    # df.to_csv("data/dim_40/ela_features/Non-elitist_B1000_40D/ELA_features.csv", index=False)
