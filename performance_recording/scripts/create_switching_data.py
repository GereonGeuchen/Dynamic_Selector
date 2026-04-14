import pandas as pd
import os
import numpy as np
import glob
import re
from pathlib import Path

"""
This file is used to create the data for the switching models based on ELA features, 
algorithm predictions, and predictions of the EPMs.
"""

# For each run (fid,iid,rep), determine the best budget based on static predictions of the selection models
# Used for the run-specific switching data
# If there are multiple best budgets, tie_break determines whether to choose the lowest or highest budget among them
def make_run_specific_best_budgets(input_csv, output_csv, tie_break="lowest"):
    df = pd.read_csv(input_csv)

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

# For each ELA file, mark for each (fid,iid,rep) whether this budget is the best or greater than the best budget for that run
# Used to create the run-specific switching data
def mark_switch_budget_and_greater_budgets_per_run(
    ela_dir: str,
    best_budgets_csv: str,
    output_dir: str,
):
    os.makedirs(output_dir, exist_ok=True)

    best_df = pd.read_csv(best_budgets_csv)
    best_df[["fid", "iid", "rep"]] = best_df[["fid", "iid", "rep"]].astype(int)

    best_budget_map = (
        best_df
        .set_index(["fid", "iid", "rep"])["best_budget"]
        .to_dict()
    )

    # Process each ELA file
    for file in sorted(os.listdir(ela_dir)):
        if not file.endswith(".csv"):
            continue

        budget_str = file.split("_")[1] 
        budget = int(budget_str[1:])   

        ela_path = os.path.join(ela_dir, file)
        df = pd.read_csv(ela_path)

        df[["fid", "iid", "rep"]] = df[["fid", "iid", "rep"]].astype(int)

        df["switch"] = df.apply(
            lambda row: budget >= best_budget_map.get(
                (row["fid"], row["iid"], row["rep"]),
                float("inf")  # if no best is known: never switch
            ),
            axis=1
        )

        out_path = os.path.join(output_dir, file)
        df.to_csv(out_path, index=False)

# This function takes the normalized_precision ELA files used to train the selection models,
# and substitutes the algorithm prediction columns with those from the given predictions CSV file
# Used to create ELA files that also include the algorithm predictions from the selection models

# === This is not used in the final paper ===
def update_algo_columns(predictions_csv: str, a1_folder: str, output_folder: str):

    algo_cols = ["BFGS", "DE", "Elitist", "MLSL", "Non-elitist", "PSO"]
    variance_cols = ["var_" + col for col in algo_cols]

    preds = pd.read_csv(predictions_csv)

    required_cols = ["fid", "iid", "rep", "budget"] + algo_cols
    missing = [c for c in required_cols if c not in preds.columns]
    if missing:
        raise ValueError(f"Missing columns in predictions file: {missing}")

    os.makedirs(output_folder, exist_ok=True)

    files = glob.glob(os.path.join(a1_folder, "A1_B*_5D_ela.csv"))
    for path in files:
        print(f"Processing {path} ...")

        m = re.search(r"_B(\d+)_", os.path.basename(path))
        if not m:
            print(f"Could not extract budget from filename {path}, skipping.")
            continue
        budget = int(m.group(1))

        df = pd.read_csv(path)

        preds_b = preds[preds["budget"] == budget][["fid", "iid", "rep"] + algo_cols + variance_cols]

        df_clean = df.drop(columns=[c for c in algo_cols if c in df.columns])

        merged = df_clean.merge(preds_b, on=["fid", "iid", "rep"], how="left")

        out_path = os.path.join(output_folder, os.path.basename(path))
        merged.to_csv(out_path, index=False)
        print(f"Wrote updated file to {out_path}")

# This function adds the lookahead predictions to the switching data
# Currently, we first attach binary labels and then insert the predictions, weird workflow so I might change that later
def add_preds_to_ela_folder(ela_dir, preds_csv, pattern="A1_B*_5D_ela.csv", out_dir=None):
    ela_dir = Path(ela_dir)
    preds = pd.read_csv(preds_csv)[["fid","iid","rep","budget"] + [f"pred_t{i}" for i in range(0,20)]]
    preds[["fid","iid","rep","budget"]] = preds[["fid","iid","rep","budget"]].astype(int)

    out_dir = Path(out_dir) if out_dir else (ela_dir / "with_preds")
    out_dir.mkdir(parents=True, exist_ok=True)

    for p in ela_dir.glob(pattern):

        budget = int(p.stem.split("_B")[1].split("_")[0])
        if budget == 1000: continue

        max_lookahead = (1000 - budget) // 50
        use_cols = [f"pred_t{i}" for i in range(0, max_lookahead + 1)]


        ela = pd.read_csv(p)
        ela[["fid","iid","rep"]] = ela[["fid","iid","rep"]].astype(int)
        ela["budget"] = budget

        ela = ela.merge(
            preds[["fid","iid","rep","budget"] + use_cols],
            on=["fid","iid","rep","budget"],
            how="left"
        ).drop(columns=["budget"])

        cols = list(ela.columns)
        insert_at = cols.index("switch")
        for c in reversed(use_cols):
            cols.insert(insert_at, cols.pop(cols.index(c)))
        ela = ela[cols]

        ela.to_csv(out_dir / p.name, index=False)

if __name__ == "__main__":
    add_preds_to_ela_folder(
        ela_dir="../data/switch_data/auc/A1_data_algo_features_switch_auc_highest",
        preds_csv="../data/lookahead_performances/lookahead_performances_all_epms_auc/predicted_switchpoint_performances_test.csv",
        out_dir="../data/switch_data/auc/A1_data_algo_features_switch_auc_highest_with_lookahead_predictions"
    )