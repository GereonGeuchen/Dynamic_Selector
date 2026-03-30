import os

import pandas as pd

# This function marks optimal evaluations as true that fall within a CMA-ES iteration after which we should have switched


def create_is_optimal_trajectory_files(
    trajectory_input_path: str,
    optimal_budget_input_path: str,
    output_lowest_path: str,
    output_highest_path: str,
    output_all_path: str,
):
    """
    Creates three trajectory files with an added boolean column 'is_optimal'.

    Inputs
    ------
    trajectory_input_path:
        CSV with columns at least:
        fid, iid, rep, evaluations, raw_y, current_best

    optimal_budget_input_path:
        CSV with columns at least:
        fid, iid, rep, budget, precision, is_optimal

    Outputs
    -------
    output_lowest_path:
        For each (fid, iid, rep), only the lowest optimal budget is considered.

    output_highest_path:
        For each (fid, iid, rep), only the highest optimal budget is considered.

    output_all_path:
        For each (fid, iid, rep), all optimal budgets are considered.

    Marking rule
    ------------
    For each selected optimal budget b of a given (fid, iid, rep):
    - compute the smallest multiple of 8 that is >= b
    - call this m
    - set end_eval = m - 1
    - set start_eval = end_eval - 7
    - mark is_optimal = True for evaluations start_eval, ..., end_eval

    Example:
    - if budget = 50, then m = 56
    - start_eval = 48, end_eval = 55
    - marked evaluations are 48..55
    """

    # Load data
    df_traj = pd.read_csv(trajectory_input_path)
    df_opt = pd.read_csv(optimal_budget_input_path)

    # Ensure correct dtypes
    for col in ["fid", "iid", "rep"]:
        df_traj[col] = df_traj[col].astype(int)
        df_opt[col] = df_opt[col].astype(int)

    df_traj["evaluations"] = df_traj["evaluations"].astype(int)
    df_traj["raw_y"] = df_traj["raw_y"].astype(float)
    df_traj["current_best"] = df_traj["current_best"].astype(float)

    df_opt["budget"] = df_opt["budget"].astype(int)
    df_opt["precision"] = df_opt["precision"].astype(float)
    df_opt["is_optimal"] = df_opt["is_optimal"].astype(bool)

    # Keep only optimal budgets
    df_opt_true = df_opt[df_opt["is_optimal"]].copy()

    def build_output_df(mode: str) -> pd.DataFrame:
        if mode == "all":
            df_selected = df_opt_true.copy()
        elif mode == "lowest":
            idx = df_opt_true.groupby(["fid", "iid", "rep"])["budget"].idxmin()
            df_selected = df_opt_true.loc[idx].copy()
        elif mode == "highest":
            idx = df_opt_true.groupby(["fid", "iid", "rep"])["budget"].idxmax()
            df_selected = df_opt_true.loc[idx].copy()
        else:
            raise ValueError(f"Unknown mode: {mode}")

        # Compute marking interval
        m = ((df_selected["budget"] + 7) // 8) * 8
        df_selected["end_eval"] = m - 1
        df_selected["start_eval"] = df_selected["end_eval"] - 7

        # Copy trajectories
        df_out = df_traj.copy()
        df_out["is_optimal"] = False

        # Merge intervals onto trajectories
        merged = df_out.merge(
            df_selected[["fid", "iid", "rep", "start_eval", "end_eval"]],
            on=["fid", "iid", "rep"],
            how="left",
        )

        # Mark evaluations inside interval
        in_interval = (
            merged["start_eval"].notna()
            & (merged["evaluations"] >= merged["start_eval"])
            & (merged["evaluations"] <= merged["end_eval"])
        )

        if mode in {"lowest", "highest"}:
            merged["is_optimal"] = in_interval
            result = merged[
                ["fid", "iid", "rep", "evaluations", "raw_y", "current_best", "is_optimal"]
            ].copy()
        else:
            # Multiple optimal budgets may produce duplicate merged rows; collapse with OR
            merged["hit"] = in_interval
            result = (
                merged.groupby(
                    ["fid", "iid", "rep", "evaluations", "raw_y", "current_best"],
                    as_index=False
                )["hit"]
                .any()
                .rename(columns={"hit": "is_optimal"})
            )

        result = result.sort_values(["fid", "iid", "rep", "evaluations"]).reset_index(drop=True)
        return result

    # Build outputs
    df_lowest = build_output_df("lowest")
    df_highest = build_output_df("highest")
    df_all = build_output_df("all")

    # Save outputs
    for path in [output_lowest_path, output_highest_path, output_all_path]:
        os.makedirs(os.path.dirname(path), exist_ok=True)

    df_lowest.to_csv(output_lowest_path, index=False)
    df_highest.to_csv(output_highest_path, index=False)
    df_all.to_csv(output_all_path, index=False)



if __name__ == "__main__":
    # create_is_optimal_trajectory_files(
    #     trajectory_input_path="../data/A1_B1000_5D_with_current_best.csv",
    #     optimal_budget_input_path="../data/A2_best_precisions_with_is_optimal.csv",
    #     output_lowest_path="../data/A1_B1000_5D_with_current_best_with_is_optimal_lowest.csv",
    #     output_highest_path="../data/A1_B1000_5D_with_current_best_with_is_optimal_highest.csv",
    #     output_all_path="../data/A1_B1000_5D_with_current_best_with_is_optimal_all.csv",
    # )

    df_labeled = pd.read_csv("../data/A1_B1000_5D_with_current_best_with_is_optimal_lowest.csv")
    df_x = pd.read_csv("../data/A1_B1000_5D.csv")

    # Ensure consistent types
    for col in ["fid", "iid", "rep", "evaluations"]:
        df_labeled[col] = df_labeled[col].astype(int)
        df_x[col] = df_x[col].astype(int)

    # Detect x-columns automatically
    x_cols = [c for c in df_x.columns if c.startswith("x")]
    if not x_cols:
        raise ValueError("No x-columns found in x_values_input_path.")

    # Keep only keys + x-columns from the original trajectory file
    df_x_small = df_x[["fid", "iid", "rep", "evaluations"] + x_cols].copy()

    # Optional sanity check: there should be at most one matching row per key
    dup_count = df_x_small.duplicated(subset=["fid", "iid", "rep", "evaluations"]).sum()
    if dup_count > 0:
        raise ValueError(
            f"x-values file contains {dup_count} duplicate rows for "
            f"(fid, iid, rep, evaluations)."
        )

    # Merge
    df_out = df_labeled.merge(
        df_x_small,
        on=["fid", "iid", "rep", "evaluations"],
        how="left",
    )

    # Optional sanity check: make sure all rows got x-values
    missing = df_out[x_cols].isna().any(axis=1).sum()
    if missing > 0:
        print(f"Warning: {missing} rows did not find matching x-values.")

    df_out = df_out.sort_values(["fid", "iid", "rep", "evaluations"]).reset_index(drop=True)

    output_path = "../data/A1_B1000_5D_with_current_best_with_is_optimal_lowest_and_x.csv"

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_out.to_csv(output_path, index=False)