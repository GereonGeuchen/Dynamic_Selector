import pandas as pd
import os
import numpy as np
import glob
import re

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
    output_dir,
    min=True
):
    os.makedirs(output_dir, exist_ok=True)

    best_df = pd.read_csv(best_budgets_csv)
    if min:
        best_budget_map = best_df.groupby("fid")["best_budget"].min().to_dict()
    else:
        best_budget_map = best_df.groupby("fid")["best_budget"].max().to_dict()

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

def update_algo_columns(predictions_csv: str, a1_folder: str, output_folder: str):

    algo_cols = ["BFGS", "DE", "Elitist", "MLSL", "Non-elitist", "PSO"]
    variance_cols = ["var_" + col for col in algo_cols]

    # Load predictions
    preds = pd.read_csv(predictions_csv)

    # Check required columns
    required_cols = ["fid", "iid", "rep", "budget"] + algo_cols
    missing = [c for c in required_cols if c not in preds.columns]
    if missing:
        raise ValueError(f"Missing columns in predictions file: {missing}")

    # Create output folder if needed
    os.makedirs(output_folder, exist_ok=True)

    # Iterate over all A1 files
    files = glob.glob(os.path.join(a1_folder, "A1_B*_5D_ela.csv"))
    for path in files:
        print(f"Processing {path} ...")

        # Extract budget from filename (e.g., A1_B50_5D.csv → 50)
        m = re.search(r"_B(\d+)_", os.path.basename(path))
        if not m:
            print(f"Could not extract budget from filename {path}, skipping.")
            continue
        budget = int(m.group(1))

        # Load this A1 file
        df = pd.read_csv(path)

        # Filter predictions for this budget
        preds_b = preds[preds["budget"] == budget][["fid", "iid", "rep"] + algo_cols + variance_cols]

        # Remove old algo columns
        df_clean = df.drop(columns=[c for c in algo_cols if c in df.columns])

        # Merge in the new predictions
        merged = df_clean.merge(preds_b, on=["fid", "iid", "rep"], how="left")

        # Warning if something mismatches
        if merged[algo_cols].isna().any().any():
            bad_rows = merged[merged[algo_cols].isna().any(axis=1)][["fid", "iid", "rep"]].drop_duplicates()
            print(f"⚠ WARNING: Missing predictions for some rows:\n{bad_rows}")

        # Save in output folder under same filename
        out_path = os.path.join(output_folder, os.path.basename(path))
        merged.to_csv(out_path, index=False)
        print(f"✔ Wrote updated file to {out_path}")

    print("🎉 All updated files written to:", output_folder)


if __name__ == "__main__":
    # make_run_specific_best_budgets("../data/selector_performances/no_state_variance/predicted_static_precisions.csv",
    #                                "../data/selector_performances/no_state_variance/best_budgets_per_run.csv", tie_break="highest")
    
    mark_switch_budget_and_greater_budgets_per_run(
        ela_with_state_dir="../data/ela/A1_data_ela_with_algo_features",
        best_budgets_csv="../data/selector_performances/algo_features/best_budgets_per_run.csv",
        output_dir="../data/switch_data/A1_data_algo_features_switch_2"
    )

    # update_algo_columns(
    #     predictions_csv="../data/selector_performances/no_state_variance/all_normalized_predictions.csv",
    #     a1_folder="../data/ela/A1_data_ela_normalized_with_precisions",
    #     output_folder="../data/ela/A1_data_ela_with_algo_features_variance"
    # )


    # df1 = pd.read_csv("../data/selector_performances/algo_features_variance/best_budgets_per_run.csv")
    # df2 = pd.read_csv("../data/selector_performances/algo_features/best_budgets_per_run.csv")

    # # Check if best_budgets columns are identical
    # print(f"Are best budgets identical? {df1['best_budget'].equals(df2['best_budget'])}")