#!/usr/bin/env python3

import pandas as pd
import numpy as np
import os
import warnings
import sys
import argparse

# Add pflacco module path if needed
sys.path.append(os.path.join(os.path.dirname(__file__), 'pflacco'))

from classical_ela_features import ( # type: ignore
    calculate_ela_distribution,
    calculate_ela_meta,
    calculate_ela_level,
    calculate_dispersion,
    calculate_information_content,
    calculate_nbc
)

def calculate_ela_features(budget):
    base_folder = "../data/run_data_5D/A1_data_5D_affine_test"   
    output_folder = "../data/raw_ela_data/A1_data_ela_affine_test"           

    os.makedirs(output_folder, exist_ok=True)
    filename = f"A1_B{budget}_5D.csv"
    filepath = os.path.join(base_folder, filename)
    df = pd.read_csv(filepath)

    print(f"Processing file: {filepath}")

    x_cols = [col for col in df.columns if col.startswith("x")]
    output_path = os.path.join(output_folder, f"A1_B{budget}_5D_ela.csv")

    first_write = True  # controls header

    for (fid, type, rep), group in df.groupby(["fid", "type", "rep"]):
        int_rep = int(rep)
        np.random.seed(int_rep)
        print(f"Processing fid: {fid}, type: {type}, rep: {rep}, budget: {budget}")
        group = group.reset_index(drop=True)
        X = group[x_cols].to_numpy()
        # Changed truey_y to raw_y for testing purposes
        y = np.asarray(group["true_y"].values, dtype=float).flatten()



        features = {}
        features.update(calculate_ela_distribution(X, y))

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            features.update(calculate_ela_meta(X, y))

        if budget > 16:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                if budget <= 88:
                    if budget <= 32:
                        features.update(calculate_ela_level(X, y, ela_level_quantiles=[0.50]))
                    else:
                        features.update(calculate_ela_level(X, y, ela_level_quantiles=[0.25, 0.50]))
                else:
                    features.update(calculate_ela_level(X, y))

        features.update(calculate_dispersion(X, y))
        assert isinstance(y, np.ndarray)
        assert y.ndim == 1
        assert y.shape[0] == X.shape[0]

        # Set range of epsilon values for information content to deal with early convergence
        features.update(calculate_information_content(X, y,
                                                      ic_epsilon=np.insert(10 ** np.linspace(start=-7, stop=15, num=1000), 0, 0)))
        
        if budget <= 16:
            # For budgets <= 12, we use the raw_y values
            features.update(calculate_nbc(X, y, fast_k = 2))
        else:
            features.update(calculate_nbc(X, y))

        # # ------------------------------------------------------------------
        # # NEW: best-so-far progression features over intervals of evaluations
        # # ------------------------------------------------------------------
        # next_mult = ((budget + 7) // 8) * 8
        # intervals, labels = make_intervals(next_mult)

        # if intervals:
        #     # index by evaluations for quick lookup of current_best
        #     eval_indexed = group.set_index("evaluations")

        #     for (start, end), perc_label in zip(intervals, labels):
        #         # ensure both evaluation points exist
        #         if start not in eval_indexed.index or end not in eval_indexed.index:
        #             continue

        #         low_bsf = float(eval_indexed.loc[start, "current_best"])
        #         high_bsf = float(eval_indexed.loc[end, "current_best"])
        #         width = end - start
        #         if width <= 0:
        #             continue

        #         # (high - low) / (width / 8) = (high - low) * 8 / width
        #         gradient = ((high_bsf - low_bsf) * 8.0) / width

        #         feat_name = f"bsf-progression_{perc_label}"
        #         features[feat_name] = gradient

        # Add identifying metadata
        features["fid"] = fid
        features["type"] = type
        features["rep"] = rep

        # if fid in [1, 2, 3, 4, 5]:
        #     features["high_level_category"] = 1
        # elif fid in [6, 7, 8, 9]:
        #     features["high_level_category"] = 2
        # elif fid in [10, 11, 12, 13, 14]:
        #     features["high_level_category"] = 3
        # elif fid in [15, 16, 17, 18, 19]:
        #     features["high_level_category"] = 4
        # elif fid in [20, 21, 22, 23, 24]:
        #     features["high_level_category"] = 5
        # else:
        #     features["high_level_category"] = None
        features["high_level_category"] = "affine"

        # Remove ela_meta.quad_w_interact.adj_r2 if budget <= 56

        if budget <= 56:
            features.pop('ela_meta.quad_w_interact.adj_r2', None)
            if budget <= 16:
                features.pop('ela_meta.lin_w_interact.adj_r2', None)
    
        for key in list(features.keys()):
            if key.endswith(".costs_runtime"):
                features.pop(key, None)
        # Create DataFrame for one row, reorder columns
        row_df = pd.DataFrame([features])
        cols = ["fid", "type", "rep", "high_level_category"]
        ordered_cols = cols + [col for col in row_df.columns if col not in cols]
        row_df = row_df[ordered_cols]

        # Append row to file
        row_df.to_csv(output_path, mode='a', header=first_write, index=False)
        first_write = False  # only write header once

    print(f"Completed processing for budget: {budget}")

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

def make_intervals(n: float):
    candidates = []
    if  n <= 104:
        if n < 16:
            candidates = []
        if n == 16:
            candidates = [(8, 16)]
        elif n == 24:
            candidates = [(8, 16), (16, 24)]
        elif n == 32:
            candidates = [(8, 16), (16, 24), (24, 32)]
        elif n == 40:
            candidates = [(8, 16), (16, 24), (24, 32), (32, 40)]
        elif n == 48:
            candidates = [(8, 16), (16, 32), (32, 40), (40, 48)]
        elif n == 56:
            candidates = [(8, 16), (16, 32), (32, 48), (48, 56)]
        elif n == 64:
            candidates = [(8, 16), (16, 40), (40, 56), (56, 64)]
        elif n == 72:
            candidates = [(8, 16), (16, 40), (40, 64), (64, 72)]
        elif n == 80:
            candidates = [(8, 16), (16, 48), (48, 72), (72, 80)]
        elif n == 88:
            candidates = [(8, 16), (16, 48), (48, 80), (80, 88)]
        elif n == 96:
            candidates = [(8, 16), (16, 56), (56, 88), (88, 96)]
        elif n == 104:
            candidates = [(8, 16), (16, 56), (56, 96), (96, 104)]
    else:
        n /= 8
        print(0.10 * n)
        b1 = 8 * max(1, round(0.10 * n)) 
        b2 = 8 * round(0.50 * n) 
        b3 = 8 * round(0.90 * n) 
        b4 = 8 * round(n) 

        # Build intervals as (start, end)
        candidates = [
            (8, b1),
            (b1, b2),
            (b2, b3),
            (b3, b4)
        ]

    unique_intervals = []
    for start, end in candidates:
        if end <= start:
            continue
        if unique_intervals and (start, end) == unique_intervals[-1]:
            continue
        unique_intervals.append((start, end))

    labels_map = {
        0: [],
        1: ["1.0"],
        2: ["0.5", "1.0"],
        3: ["0.5", "0.9", "1.0"],
        4: ["0.1", "0.5", "0.9", "1.0"],
    }
    labels = labels_map.get(len(unique_intervals), ["0.1", "0.5", "0.9", "1.0"][:len(unique_intervals)])

    return unique_intervals, labels


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--budget", type=int, required=True, help="Budget to process")
    args = parser.parse_args()
    budget = args.budget
    calculate_ela_features(budget=budget)
    # # with warnings.catch_warnings():
    # #     warnings.filterwarnings("ignore", category=RuntimeWarning)
    # #     warnings.filterwarnings("ignore", category=UserWarning)
    # #     append_standard_deviation_stats(budget=budget,
    # #                                     ela_path=f"../data/ela_with_cma/A1_data_with_cma_testSet/A1_B{budget}_5D_ela_with_state.csv",
    # #                                     raw_data_path=f"../data/run_data_csvs/A1_data_testSet/A1_B{budget}_5D.csv",
    # #                                     output_path=f"../data/ela_with_cma_std/A1_data_ela_cma_std_testSet/A1_B{budget}_5D_ela_with_state.csv")
    # append_standard_deviation_stats(
    #     budget = args.budget,
    #     ela_path = f"../data/ela_with_cma/A1_data_5D_test/A1_B{budget}_5D_ela_with_state.csv",
    #     raw_data_path = f"../data/run_data_5D/A1_data_5D_test/A1_B{budget}_5D.csv",
    #     output_path = f"../data/ela_with_cma_std/A1_data_5D_test/A1_B{budget}_5D_ela_with_state.csv"
    # )
  