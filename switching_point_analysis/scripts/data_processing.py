import os
import pandas as pd

def create_is_optimal_trajectory_files(
    trajectory_input_path: str,
    optimal_budget_input_path: str,
    output_lowest_path: str,
    output_highest_path: str,
    output_all_path: str,
):
    """
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

    This corresponds to the CMA iteration after which we should have switched (8 is the CMA-ES population size)

    We create three files: one for the lowest optimal budget per (fid, iid, rep), 
                           one for the highest, and one for all optimal budgets.
    """

    df_traj = pd.read_csv(trajectory_input_path)
    df_opt = pd.read_csv(optimal_budget_input_path)

    for col in ["fid", "iid", "rep"]:
        df_traj[col] = df_traj[col].astype(int)
        df_opt[col] = df_opt[col].astype(int)

    df_traj["evaluations"] = df_traj["evaluations"].astype(int)
    df_traj["raw_y"] = df_traj["raw_y"].astype(float)
    df_traj["current_best"] = df_traj["current_best"].astype(float)

    df_opt["budget"] = df_opt["budget"].astype(int)
    df_opt["precision"] = df_opt["precision"].astype(float)
    df_opt["is_optimal"] = df_opt["is_optimal"].astype(bool)

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

        df_out = df_traj.copy()
        df_out["is_optimal"] = False

        merged = df_out.merge(
            df_selected[["fid", "iid", "rep", "start_eval", "end_eval"]],
            on=["fid", "iid", "rep"],
            how="left",
        )

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

    df_lowest = build_output_df("lowest")
    df_highest = build_output_df("highest")
    df_all = build_output_df("all")

    for path in [output_lowest_path, output_highest_path, output_all_path]:
        os.makedirs(os.path.dirname(path), exist_ok=True)

    df_lowest.to_csv(output_lowest_path, index=False)
    df_highest.to_csv(output_highest_path, index=False)
    df_all.to_csv(output_all_path, index=False)



if __name__ == "__main__":
    create_is_optimal_trajectory_files(
        trajectory_input_path="../data/A1_B1000_5D_with_current_best.csv",
        optimal_budget_input_path="../data/A2_best_precisions_with_is_optimal.csv",
        output_lowest_path="../data/A1_B1000_5D_with_current_best_with_is_optimal_lowest.csv",
        output_highest_path="../data/A1_B1000_5D_with_current_best_with_is_optimal_highest.csv",
        output_all_path="../data/A1_B1000_5D_with_current_best_with_is_optimal_all.csv",
    )
