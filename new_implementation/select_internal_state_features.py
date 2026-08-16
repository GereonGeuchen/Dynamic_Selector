#!/usr/bin/env python3
"""Select tsfresh features from budget-truncated CMA-ES internal states.

Example
-------
python select_internal_state_features.py \
    data/dim_40_with_internal_state/internal_state/internal_state_Non-elitist_B1000_40D.csv \
    --budget 500

The input must contain one row per CMA-ES iteration and the identifiers
``fid``, ``iid`` and ``rep``.  Each run is truncated to the requested number
of objective-function evaluations before tsfresh feature extraction.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from tsfresh import extract_features
from tsfresh.utilities.dataframe_functions import impute


# The ten state channels described in the experiment: step size, summaries of
# covariance eigenvalues and evolution paths, Mahalanobis distances, and log
# likelihood. ``ps_squared`` is intentionally excluded: it is derived from
# ``ps_norm`` and is not part of that ten-dimensional state vector.
DEFAULT_CHANNELS = (
    "sigma",
    "d_norm",
    "d_mean",
    "ps_norm",
    "ps_mean",
    "pc_norm",
    "pc_mean",
    "mhl_norm",
    "mhl_mean",
    "loglikelihood",
)
IDENTIFIER_COLUMNS = ("fid", "iid", "rep")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_csv", type=Path)
    parser.add_argument("--dimension", type=int, default=40,
                        help="Problem dimension used to choose the default output directory.")
    parser.add_argument("--budget", type=int, required=True,
                        help="Include states through the first observation with evaluations >= this value.")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--threshold", type=float, default=2e-3,
                        help="Keep Random-Forest importances strictly above this value.")
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--random-state", type=int, default=32)
    parser.add_argument("--save-series", action="store_true",
                        help="Save the budget-truncated table passed to tsfresh as truncated_series.csv.")
    parser.add_argument("--channels", nargs="+", default=list(DEFAULT_CHANNELS),
                        help="State columns for tsfresh extraction (default: the 10-vector).")
    return parser.parse_args()


def prepare_time_series(
    states: pd.DataFrame, budget: int, channels: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = set(IDENTIFIER_COLUMNS) | {"evaluations"} | set(channels)
    missing = sorted(required - set(states.columns))
    if missing:
        raise ValueError(f"Input is missing required columns: {', '.join(missing)}")

    # A string ID is robust to tsfresh's grouping and is retained as metadata.
    states["run_id"] = states[list(IDENTIFIER_COLUMNS)].astype(str).agg("_".join, axis=1)
    states = states.sort_values(["run_id", "evaluations", "iteration"], kind="stable")

    # CMA-ES advances by a population at a time.  Keep all prior observations
    # and the first one that reaches or passes the requested evaluation budget.
    # A run that never reaches the budget is retained in full.
    def cutoff_evaluation(run: pd.DataFrame) -> int:
        reached = run.loc[run["evaluations"] >= budget, "evaluations"]
        return reached.iloc[0] if not reached.empty else run["evaluations"].iloc[-1]

    cutoffs = states.groupby("run_id", sort=False, group_keys=False).apply(cutoff_evaluation)
    states = states.loc[states["evaluations"] <= states["run_id"].map(cutoffs)].copy()
    metadata = states.drop_duplicates("run_id").set_index("run_id")[list(IDENTIFIER_COLUMNS)]

    run_lengths = states.groupby("run_id", sort=False).size()
    if (run_lengths < 2).any():
        bad_runs = ", ".join(run_lengths[run_lengths < 2].index[:5])
        raise ValueError(
            "tsfresh needs at least two observations per run; insufficient runs: " + bad_runs
        )

    return states[["run_id", "evaluations", *channels]], metadata


def main() -> None:
    args = parse_args()
    if args.budget <= 0:
        raise ValueError("--budget must be positive")

    states = pd.read_csv(args.input_csv)
    series, metadata = prepare_time_series(states, args.budget, args.channels)
    output_dir = args.output_dir or (
        Path("data") / f"dim_{args.dimension}_with_internal_state" / "time_series_analysis"
    )
    output_suffix = f"_B{args.budget}"
    if args.save_series:
        output_dir.mkdir(parents=True, exist_ok=True)
        series.to_csv(output_dir / f"truncated_series{output_suffix}.csv", index=False)
        metadata.reset_index().to_csv(output_dir / f"series_metadata{output_suffix}.csv", index=False)

    features = extract_features(
        series,
        column_id="run_id",
        column_sort="evaluations",
        n_jobs=args.n_jobs,
    )
    impute(features)
    metadata = metadata.loc[features.index]

    classifier = RandomForestClassifier(
        n_jobs=args.n_jobs,
        class_weight="balanced",
        max_depth=5,
        random_state=args.random_state,
    )
    classifier.fit(features, metadata["fid"])
    importances = pd.Series(classifier.feature_importances_, index=features.columns, name="importance")
    importances = importances.sort_values(ascending=False)
    selected_names = importances.index[importances > args.threshold]

    output_dir.mkdir(parents=True, exist_ok=True)
    importances.rename_axis("feature").reset_index().to_csv(
        output_dir / f"feature_importances{output_suffix}.csv", index=False
    )
    pd.DataFrame({"feature": selected_names, "importance": importances.loc[selected_names].to_numpy()}).to_csv(
        output_dir / f"selected_features{output_suffix}.csv", index=False
    )
    selected = features.loc[:, selected_names].copy()
    selected.insert(0, "rep", metadata["rep"].to_numpy())
    selected.insert(0, "iid", metadata["iid"].to_numpy())
    selected.insert(0, "fid", metadata["fid"].to_numpy())
    selected.to_csv(output_dir / f"selected_feature_matrix{output_suffix}.csv", index=False)
    joblib.dump(classifier, output_dir / f"fid_random_forest{output_suffix}.joblib")

    print(f"Runs: {len(features)}; channels: {len(args.channels)}; tsfresh features: {features.shape[1]}")
    print(f"Selected {len(selected_names)} features with importance > {args.threshold:g}")
    print(f"Wrote results to {output_dir}")


if __name__ == "__main__":
    main()
