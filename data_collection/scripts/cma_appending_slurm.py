import os
import sys
import pandas as pd
from glob import glob
from io import StringIO
import numpy as np

def extract_final_internal_state(dat_path, target_iid, target_rep):
    try:
        # Read and clean repeated headers
        with open(dat_path, "r") as f:
            lines = f.readlines()

        cleaned_lines = []
        header_seen = False
        for line in lines:
            if line.strip().startswith("evaluations"):
                if not header_seen:
                    cleaned_lines.append(line)
                    header_seen = True
                # else: skip repeated headers
            else:
                cleaned_lines.append(line)

        # Parse cleaned content into DataFrame
        df = pd.read_csv(StringIO("".join(cleaned_lines)), delim_whitespace=True)

        # Convert iid and rep to int for matching
        df["rep"] = pd.to_numeric(df["rep"], errors="coerce").astype(int)
        df["iid"] = pd.to_numeric(df["iid"], errors="coerce").astype(int)
        target_rep = int(target_rep)
        target_iid = int(target_iid)

        # Filter for target rep and iid
        df = df[(df["rep"] == target_rep) & (df["iid"] == target_iid)]
        if df.empty:
            return None

        # Get final row (maximum evaluations)
        final_row = df.loc[df["evaluations"].idxmax()]
        state_dict = final_row.loc["sigma":"mhl_mean"].to_dict()

        # Remove unwanted keys
        for key in ["t", "ps_squared", "ps_ratio"]:
            state_dict.pop(key, None)

        return state_dict
    
    except Exception as e:
        print(f"Error processing {dat_path}: {e}")
        return None
    

def append_cma_state_to_ela(ela_dir, run_dir, output_dir, budgets):
    os.makedirs(output_dir, exist_ok=True)

    print("Starting CMA state appending process...")

    for budget in budgets:
        print(f"Processing budget: {budget}")
        ela_path = os.path.join(ela_dir, f"A1_B{budget}_5D_ela.csv")
        run_path = os.path.join(run_dir, f"A1_B{budget}_5D")

        if not os.path.isfile(ela_path):
            print(f"Skipping: {ela_path} not found.")
            continue
        if not os.path.isdir(run_path):
            print(f"Skipping: {run_path} not found.")
            continue

        df_ela = pd.read_csv(ela_path)
        df_ela["iid"] = df_ela["iid"].astype(int)
        df_ela["rep"] = df_ela["rep"].astype(int)

        appended_data = []
        for _, row in df_ela.iterrows():
            fid, iid, rep = int(row["fid"]), int(row["iid"]), int(row["rep"])
            pattern = os.path.join(run_path, f"data_f{fid}*", f"IOHprofiler_f{fid}_DIM5.dat")
            dat_files = glob(pattern)
            if not dat_files:
                print(f"No matching file for fid={fid}, iid={iid}, rep={rep} at budget {budget}")
                appended_data.append({})
                continue

            state = extract_final_internal_state(dat_files[0], iid, rep)
            if state is None:
                print(f"  ✘ No state found in {dat_files[0]} for iid={iid}, rep={rep}")
                state = {}
            appended_data.append(state)

        df_state = pd.DataFrame(appended_data)
        df_combined = pd.concat([df_ela.reset_index(drop=True), df_state.reset_index(drop=True)], axis=1)

        out_path = os.path.join(output_dir, f"A1_B{budget}_5D_ela_with_state.csv")
        df_combined.to_csv(out_path, index=False)
        print(f"Saved: {out_path}")

# Function that adds the new "internal" features
def append_standard_deviation_stats(budget, ela_path, raw_data_path, output_path):
    df_ela = pd.read_csv(ela_path)
    df_raw = pd.read_csv(raw_data_path)

    x_cols = [col for col in df_raw.columns if col.startswith("x")]
    tail_counts = {8: [1], 16: [1,2], 24: [1, 2, 3], 32: [1, 2, 4], 40: [1, 2, 5], float("inf"): [1, 2, 5]}
    applicable_ns = next(v for k, v in tail_counts.items() if budget <= k)

    appended_rows = []

    for (fid, iid, rep), group in df_raw.groupby(["fid", "iid", "rep"]):
        group = group.reset_index(drop=True)
        row = {"fid": fid, "iid": iid, "rep": rep}

        for n in applicable_ns:
            k = 8 * n
            tail = group.iloc[-k:] if len(group) >= k else group

            row[f"std_y_last_{n}"] = float(np.std(tail["true_y"].values, ddof=1))

            stds_x = np.std(tail[x_cols].values, axis=0, ddof=1)
            row[f"mean_std_x_last_{n}"] = float(np.mean(stds_x))

        appended_rows.append(row)

    df_stats = pd.DataFrame(appended_rows)
    df_combined = pd.merge(df_ela, df_stats, on=["fid", "iid", "rep"], how="left")

    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))

    df_combined.to_csv(output_path, index=False)
    print(f"Tail statistics added and saved to: {output_path}")



if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python append_cma_state_to_ela.py <budget>")
        sys.exit(1)

    budget = int(sys.argv[1])

    # budget  =100

    # append_cma_state_to_ela(
    #     ela_dir="../data/raw_ela_data/A1_data_ela_test",
    #     run_dir="../data/run_data_5D/A1_data_5D_test",
    #     output_dir="../data/ela_with_cma/A1_data_5D_test",
    #     budgets=[budget]
    # )

    append_standard_deviation_stats(budget=budget,
                                    ela_path=f"../data/ela_with_cma/A1_data_5D/A1_B{budget}_5D_ela_with_state.csv",
                                    raw_data_path=f"../data/run_data_5D/A1_data_5D/A1_B{budget}_5D.csv",
                                    output_path=f"../data/ela_with_cma_std/A1_data_5D/A1_B{budget}_5D_ela_with_state.csv")