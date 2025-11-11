import pandas as pd
import os
import numpy as np

# Given the perforamce data of the selection models, we determine the best selector (and corresponding budget) for each fid
def compute_best_budgets(input_csv):
    df = pd.read_csv(input_csv)

    # Identify the static budget columns
    budget_cols = [col for col in df.columns if col.startswith("static_B")]

    results = []

    # Group by fid
    for fid, group in df.groupby("fid"):
        # Sum precision values for each budget column
        sums = group[budget_cols].sum()
        min_total = sums.min()

        # Get budgets that achieve the minimum total precision
        best_budgets = sums[sums == min_total].index

        for budget_col in best_budgets:
            budget = int(budget_col.split("_B")[-1])
            results.append({
                "fid": fid,
                "best_budget": budget,
                "total_precision": min_total
            })

    return pd.DataFrame(results)

# Creates the data for the switching models, based on features and best budget for each fid

def mark_switch_budget_and_greater_budgets(
    ela_with_state_dir,
    best_budgets_csv,
    output_dir
):
    os.makedirs(output_dir, exist_ok=True)

    best_df = pd.read_csv(best_budgets_csv)

    best_budget_map = best_df.groupby("fid")["best_budget"].min().to_dict()

    # Process each ELA file
    for file in sorted(os.listdir(ela_with_state_dir)):
        if not file.endswith(".csv"):
            continue

        budget_str = file.split("_")[1]  # B50
        budget = int(budget_str[1:])

        ela_path = os.path.join(ela_with_state_dir, file)
        df = pd.read_csv(ela_path)
        df["fid"] = df["fid"].astype(int)

        # For each fid, mark True if this file's budget >= fid's best budget
        df["switch"] = df["fid"].apply(
            lambda fid: budget >= best_budget_map.get(fid, float('inf'))
        )

        out_path = os.path.join(output_dir, file)
        df.to_csv(out_path, index=False)
        print(f"✅ Wrote: {out_path}")

def make_run_specific_best_budgets(input_csv, output_csv, tie_break="lowest"):
    df = pd.read_csv(input_csv)

    # all precision columns
    static_cols = [c for c in df.columns if c.startswith("static_B")]

    # extract numeric budgets from column names, e.g. "static_B8" -> 8
    budgets = np.array([int(c.split("B")[1]) for c in static_cols])

    values = df[static_cols].to_numpy()

    # per-row minimum value across all budgets
    row_min = values.min(axis=1, keepdims=True)

    # True where this entry equals the row minimum (potential ties)
    is_min = values == row_min

    if tie_break == "lowest":
        filler = np.inf
        candidates = np.where(is_min, budgets, filler)
        best_budget = candidates.min(axis=1)
    else:
        filler = -np.inf
        candidates = np.where(is_min, budgets, filler)
        best_budget = candidates.max(axis=1)

    # Build result with same index columns + best_budget
    out = df[["fid", "iid", "rep"]].copy()
    out["best_budget"] = best_budget.astype(int)

    out.to_csv(output_csv, index=False)
    return out

def mark_switch_budget_and_greater_budgets_per_run(
    ela_with_state_dir: str,
    best_budgets_csv: str,
    output_dir: str,
):
    os.makedirs(output_dir, exist_ok=True)

    # Read the per-(fid,iid,rep) best budgets
    best_df = pd.read_csv(best_budgets_csv)
    best_df[["fid", "iid", "rep"]] = best_df[["fid", "iid", "rep"]].astype(int)

    # Map (fid, iid, rep) -> best_budget
    best_budget_map = (
        best_df
        .set_index(["fid", "iid", "rep"])["best_budget"]
        .to_dict()
    )

    # Process each ELA file
    for file in sorted(os.listdir(ela_with_state_dir)):
        if not file.endswith(".csv"):
            continue

        budget_str = file.split("_")[1]  # e.g. "B50"
        budget = int(budget_str[1:])    # 50

        ela_path = os.path.join(ela_with_state_dir, file)
        df = pd.read_csv(ela_path)

        df[["fid", "iid", "rep"]] = df[["fid", "iid", "rep"]].astype(int)

        # For each (fid,iid,rep), mark True if this file's budget >= best one
        df["switch"] = df.apply(
            lambda row: budget >= best_budget_map.get(
                (row["fid"], row["iid"], row["rep"]),
                float("inf")  # if no best is known: never switch
            ),
            axis=1
        )

        out_path = os.path.join(output_dir, file)
        df.to_csv(out_path, index=False)

if __name__ == "__main__":
    # ela_with_state_dir = "../data/ela_normalized_with_precisions/A1_data_5D"
    # best_budgets_csv = "../data/selector_performances/best_budgets.csv"
    # output_dir = "../data/ela_with_switch_budget/A1_data_5D"

    # # compute_best_budgets(
    # #     "../data/selector_performances/predicted_static_precisions_rep_fold_all_sp.csv"
    # # ).to_csv(best_budgets_csv, index=False)

    # mark_switch_budget_and_greater_budgets(
    #     ela_with_state_dir,
    #     best_budgets_csv,
    #     output_dir
    # )
    # make_run_specific_best_budgets("../data/selector_performances/predicted_static_precisions_all_sp.csv",
    #                                "../data/selector_performances/run_specific_best_budgets_all_sp_highest.csv", tie_break="highest")
    mark_switch_budget_and_greater_budgets_per_run(
        ela_with_state_dir="../data/ela_normalized_with_precisions/A1_data_5D",
        best_budgets_csv="../data/selector_performances/run_specific_best_budgets_all_sp_highest.csv",
        output_dir="../data/ela_with_switch_budget/A1_data_5D_per_run_highest"
    )