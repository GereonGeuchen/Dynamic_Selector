import os
import pandas as pd
import ioh
from ioh import ProblemClass
import warnings
from sklearn.preprocessing import MinMaxScaler
import numpy as np

# Function that goes through the IOH logger files and creates clean CSV files containing of the relevant data for the pflacco computation.      
def process_ioh_data(base_path):
    dim = 5
    for budget_dir in os.listdir(base_path):
        # if not (budget_dir == 'A1_B900_5D' or budget_dir == 'A1_B950_5D' or budget_dir == 'A1_B1000_5D'):
        #     continue
        budget_path = os.path.join(base_path, budget_dir)
        if not os.path.isdir(budget_path):
            continue

        all_rows = []

        for func_dir in os.listdir(budget_path):
            func_path = os.path.join(budget_path, func_dir)
            if not os.path.isdir(func_path):
                continue

            # Extract fid from directory name like 'data_f1_Sphere'
            try:
                fid = int(func_dir.split('_')[1][1:])
            except (IndexError, ValueError):
                print(f"Skipping malformed directory: {func_dir}")
                continue

            dat_file = os.path.join(func_path, f"IOHprofiler_f{fid}_DIM{dim}.dat")
            if not os.path.isfile(dat_file):
                continue

            try:
                df = pd.read_csv(dat_file, delim_whitespace=True, comment="#", dtype=str)
            except Exception as e:
                print(f"Error reading {dat_file}: {e}")
                continue

            # Filter out repeated header rows
            df = df[df['iid'] != 'iid']

            # Convert selected columns to numeric
            numeric_cols = ['evaluations', 'raw_y', 'rep', 'iid', 'x0', 'x1', 'x2', 'x3', 'x4']
            df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')

            # Group by iid and compute absolute objective values from regrets
            for iid_val, group in df.groupby('iid'):
                print(f"Processing fid={fid}, iid={iid_val}, budget dir={budget_dir}")
                try:
                    iid_int = int(float(iid_val))
                    problem = ioh.get_problem(fid, iid_int, dim, ProblemClass.BBOB)
                    optimum = problem.optimum.y
                except Exception as e:
                    print(f"Could not load problem fid={fid}, iid={iid_val}: {e}")
                    continue

                group = group[numeric_cols].copy()
                group['fid'] = fid
                # Absolute objective value: Regret + Optimum
                group['true_y'] = group['raw_y'] + optimum
                all_rows.append(group)

        if all_rows:
            combined = pd.concat(all_rows, ignore_index=True)

            # Reorder columns
            column_order = ['fid', 'iid', 'rep', 'evaluations', 'raw_y', 'true_y', 'x0', 'x1', 'x2', 'x3', 'x4']
            combined = combined[column_order]

            # Sort rows
            combined = combined.sort_values(by=['fid', 'iid', 'rep']).reset_index(drop=True)

            # Save CSV
            output_path = os.path.join(base_path, f"{budget_dir}.csv")
            combined.to_csv(output_path, index=False)
            print(f"Saved: {output_path}")

# Function that creates the A2_precisions.csv file from the run data.
def extract_a2_precisions(base_dir, output_file="A2_precisions.csv", algorithms=None, budgets=None, fids=range(1, 25), max_evals=1000):

    print(f"Extracting A2 precisions from {base_dir} with max_evals={max_evals}...")
    if algorithms is None:
        algorithms = ["BFGS", "DE", "MLSL", "Non-elitist", "PSO", "Elitist"]
    if budgets is None:
        budgets = [8*i for i in range(1, 13)] + [50*i for i in range(2, 21)]

    results = []

    dim = 5
    lower_bound = -5
    upper_bound = 5

    for algo in algorithms:
        for budget in budgets:
            if budget != 56:
                if budget % 50 != 0: continue
            folder_name = os.path.join(base_dir, f"A2_{algo}_B{budget}_5D")
            if not os.path.isdir(folder_name):
                continue
            for fid in fids:
                func_folders = [f for f in os.listdir(folder_name) if f.startswith(f"data_f{fid}_")]
                for func_folder in func_folders:
                    print(f"Processing {func_folder} for fid={fid}, algo={algo}, budget={budget}")
                    file_path = os.path.join(folder_name, func_folder, f"IOHprofiler_f{fid}_DIM5.dat")
                    if not os.path.isfile(file_path):
                        continue
                    try:
                        df = pd.read_csv(file_path, delim_whitespace=True, comment='%')
                        df['evaluations'] = pd.to_numeric(df['evaluations'], errors='coerce')
                        df['raw_y'] = pd.to_numeric(df['raw_y'], errors='coerce')
                        df['rep'] = pd.to_numeric(df['rep'], errors='coerce', downcast='integer')
                        df['iid'] = pd.to_numeric(df['iid'], errors='coerce', downcast='integer')
                        df = df.dropna(subset=['evaluations', 'raw_y', 'rep', 'iid'])

                        # Convert x_0 to x_4 to numeric
                        for i in range(5):
                            col = f'x{i}'
                            if col in df.columns:
                                df[col] = pd.to_numeric(df[col], errors='coerce')
                    except Exception as e:
                        print(f"Failed to read {file_path}: {e}")
                        continue

                    for (rep, iid), group in df.groupby(['rep', 'iid']):
                        subset = group[group['evaluations'] <= max_evals]
                        if subset.empty:
                            continue

                        # Step 2: Filter to in-bound rows only
                        x_cols = [f'x{i}' for i in range(dim)]
                        in_bounds = subset[
                            subset[x_cols].apply(
                                lambda row: all(lower_bound <= row[x] <= upper_bound for x in x_cols),
                                axis=1
                            )
                        ]

                        # Step 3: Find filtered minimum within bounds
                        min_row_filtered = in_bounds.loc[in_bounds['raw_y'].idxmin()]
                        filtered_precision = min_row_filtered['raw_y']

                        # Store result
                        results.append({
                            "fid": fid,
                            "iid": int(iid),
                            "rep": int(rep),
                            "budget": 50 if budget == 56 else budget,
                            "algorithm": algo,
                            "precision": filtered_precision,
                        })

                            
    result_df = pd.DataFrame(results)
    result_df.sort_values(by=["fid", "iid", "rep", "budget"], inplace=True)
    result_df.to_csv(output_file, index=False)
    return result_df

def add_algorithm_precisions(ela_dir, precision_csv, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # Load the full precision table
    precision_df = pd.read_csv(precision_csv)

    # Create a pivot for fast lookup: (fid, iid, rep, budget) → columns = algorithms
    precision_pivot = precision_df.pivot_table(
        index=['fid', 'iid', 'rep', 'budget'],
        columns='algorithm',
        values='precision'
    ).reset_index()

    # Iterate over ELA files
    for file in os.listdir(ela_dir):
        if not file.endswith('.csv'):
            continue

        ela_path = os.path.join(ela_dir, file)
        ela_df = pd.read_csv(ela_path)

        # Extract budget from filename
        budget = int(file.split('_')[1][1:])  
        ela_df['budget'] = budget

        # Merge on fid, iid, rep, budget
        merged = pd.merge(
            ela_df,
            precision_pivot,
            how='left',
            on=['fid', 'iid', 'rep', 'budget']
        )

        merged.drop(columns=['budget'], inplace=True) 
        # Write to output directory
        output_path = os.path.join(output_dir, file)
        merged.to_csv(output_path, index=False)

        print(f"Wrote {output_path}")

def normalize_ela_with_precisions(
    path_in,
    path_out,
    start_col=None,
    end_col=None
):
    df = pd.read_csv(path_in)

    # Columns that must NOT be normalized
    index_cols = ["fid", "iid", "rep", "high_level_category"]
    algo_cols = ["BFGS", "DE", "Elitist", "MLSL", "Non-elitist", "PSO"]

    cols = list(df.columns)

    if start_col is None or end_col is None:
        feature_cols = [
            c for c in df.columns
            if c not in index_cols and c not in algo_cols
        ]
    else:
        if start_col not in cols or end_col not in cols:
            raise ValueError(
                f"Range {start_col} → {end_col} not found. "
                f"Columns available: {cols}"
            )
        start_idx = cols.index(start_col)
        end_idx = cols.index(end_col)
        feature_cols = cols[start_idx:end_idx + 1]

    df_out = df.copy()

    if len(feature_cols) > 0:
        feature_scaler = MinMaxScaler()
        df_out[feature_cols] = feature_scaler.fit_transform(df_out[feature_cols])

    for _, group in df_out.groupby("fid"):
        idx = group.index
        algo_matrix = df_out.loc[idx, algo_cols].to_numpy()
        flat_vals = algo_matrix.flatten().reshape(-1, 1)

        scaler = MinMaxScaler(feature_range=(1e-12, 1))
        flat_scaled = scaler.fit_transform(flat_vals).flatten()
        df_out.loc[idx, algo_cols] = flat_scaled.reshape(algo_matrix.shape)

    df_out = df_out.sort_values(by=index_cols).reset_index(drop=True)

    out_dir = os.path.dirname(path_out)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir)

    df_out.to_csv(path_out, index=False)
    print(f"Saved normalized file to: {path_out}")

def normalize_test_ela(
    train_csv_path,
    test_csv_path,
    test_out_path,
    norm_ranges=None,
):
    """
    Normalize selected ELA features in the test set using a scaler
    fitted on the training data.

    Parameters
    ----------
    train_csv_path : str
    test_csv_path  : str
    test_out_path  : str
    norm_ranges    : list of (start_col, end_col) or None
        Column ranges (inclusive) that SHOULD be scaled.
        If None, all non-index columns are scaled (old behaviour).
    """

    # Load training and test data
    df_train = pd.read_csv(train_csv_path)
    df_test = pd.read_csv(test_csv_path)

    # Index columns (never scaled)
    index_cols = ["fid", "iid", "rep", "high_level_category"]
    all_cols = df_train.columns.tolist()

    # ---------------------------------------
    # 1) Determine which columns to normalize
    # ---------------------------------------
    if norm_ranges is None:
        # Backwards-compatible behaviour: scale all non-index columns
        feature_cols = [col for col in df_train.columns if col not in index_cols]
    else:
        norm_cols = []
        for start_col, end_col in norm_ranges:
            if start_col not in all_cols or end_col not in all_cols:
                raise ValueError(
                    f"Column '{start_col}' or '{end_col}' not found in dataframe."
                )

            start_idx = all_cols.index(start_col)
            end_idx   = all_cols.index(end_col)

            if end_idx < start_idx:
                raise ValueError(f"Invalid column range: {start_col} → {end_col}")

            norm_cols.extend(all_cols[start_idx:end_idx + 1])

        # Remove index columns from the normalization set, just in case
        feature_cols = [c for c in norm_cols if c not in index_cols]

    # ---------------------------------------
    # 2) Fit scaler on the training data
    # ---------------------------------------
    scaler = MinMaxScaler()
    scaler.fit(df_train[feature_cols])

    # ---------------------------------------
    # 3) Apply scaler to the test data
    # ---------------------------------------
    df_final = df_test.copy()
    df_final[feature_cols] = scaler.transform(df_test[feature_cols])
    df_final[feature_cols] = df_final[feature_cols].astype(float)

    # Index cols + all non-normalised feature cols are already in df_final unchanged

    # ---------------------------------------
    # 4) Sort & save
    # ---------------------------------------
    df_final = df_final.sort_values(by=["fid", "iid", "rep"]).reset_index(drop=True)

    out_dir = os.path.dirname(test_out_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir)

    df_final.to_csv(test_out_path, index=False)
    print(f"Saved normalized test file to: {test_out_path}")

def add_current_best(df: pd.DataFrame, gen_size: int = 8):

    df = df.copy()

    def process_one_run(run_df):
        # Assign generation index inside this run
        run_df = run_df.copy()
        run_df["generation"] = (run_df["evaluations"] - 1) // gen_size

        # Best raw_y per generation
        gen_min = run_df.groupby("generation")["raw_y"].min()
        gen_best_cum = gen_min.cummin()

        # Map cumulative best back to all rows
        run_df["current_best"] = run_df["generation"].map(gen_best_cum)

        return run_df.drop(columns=["generation"])

    # Apply per unique run
    return df.groupby(["fid", "iid", "rep"], group_keys=False).apply(process_one_run)

def normalize_ela_with_precisions_fid_scaling(path_in, path_out):
    df = pd.read_csv(path_in)

    # 1) Index-like columns (left untouched)
    index_cols = ["fid", "iid", "rep", "high_level_category"]

    # 2) Algorithm performance columns (per-fid scaling, kept as before)
    algo_cols = ["BFGS", "DE", "Elitist", "MLSL", "Non-elitist", "PSO"]

    cols = df.columns.tolist()

    # 3) Global ELA features: from 'ela_distr.skewness' to 'nbc.nb_fitness.cor'
    global_start = cols.index("ela_distr.skewness")
    global_end = cols.index("nbc.nb_fitness.cor")
    global_feature_cols = cols[global_start:global_end + 1]

    # 4) Remaining feature columns (state / run features):
    # everything that is not index, not algo, and not global ELA
    per_fid_feature_cols = [
        c for c in cols
        if c not in index_cols + algo_cols + global_feature_cols
    ]

    # ---- 1. Global min-max scaling for global_feature_cols ----
    global_scaler = MinMaxScaler()
    df_global_scaled = pd.DataFrame(
        global_scaler.fit_transform(df[global_feature_cols]),
        columns=global_feature_cols,
        index=df.index
    )

    # ---- 2. Per-fid min-max scaling for remaining feature columns ----
    if per_fid_feature_cols:
        df_state_scaled = df[per_fid_feature_cols].copy()

        for fid, group in df.groupby("fid"):
            scaler = MinMaxScaler()
            scaled_vals = scaler.fit_transform(group[per_fid_feature_cols])
            df_state_scaled.loc[group.index, :] = scaled_vals
    else:
        df_state_scaled = df[per_fid_feature_cols].copy()  # empty, but keeps structure

    # ---- 3. Algo columns: per-fid scaling with flattening (unchanged) ----
    df_scaled_algos = df[algo_cols].copy()

    for fid, group in df.groupby("fid"):
        algo_matrix = group[algo_cols].to_numpy()
        flat_vals = algo_matrix.flatten().reshape(-1, 1)

        scaler = MinMaxScaler(feature_range=(1e-12, 1))
        flat_scaled = scaler.fit_transform(flat_vals).flatten()

        scaled_matrix = flat_scaled.reshape(algo_matrix.shape)
        df_scaled_algos.loc[group.index] = scaled_matrix

    # ---- 4. Combine everything ----
    df_final = df.copy()
    df_final[global_feature_cols] = df_global_scaled
    if per_fid_feature_cols:
        df_final[per_fid_feature_cols] = df_state_scaled
    df_final[algo_cols] = df_scaled_algos

    df_final = df_final.sort_values(by=["fid", "iid", "rep"]).reset_index(drop=True)

    out_dir = os.path.dirname(path_out)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir)

    df_final.to_csv(path_out, index=False)
    print(f"Saved normalized file to: {path_out}")

def normalize_precision_per_fid(df, col="precision", min_scale=1e-12, max_scale=1.0):
    df = df.copy()

    def scale_group(g):
        x = g[[col]].values  # 2D shape for sklearn

        if np.all(x == x[0]):  
            # Constant precision -> assign midpoint
            midpoint = (min_scale + max_scale) / 2
            g[col] = midpoint
        else:
            scaler = MinMaxScaler(feature_range=(min_scale, max_scale))
            g[col] = scaler.fit_transform(x)

        return g

    df = df.groupby("fid", group_keys=False).apply(scale_group)
    return df

# Simple min-max normalisation of just ELA columns
def minmax_normalize_ela_columns(
    df: pd.DataFrame,
    feature_range=(0.0, 1.0),
    n_prefix_cols: int = 4,
    n_suffix_cols: int = 6,
) -> pd.DataFrame:
    df = df.copy()

    # Split columns
    cols = df.columns.tolist()
    # prefix_cols = cols[:n_prefix_cols]
    # suffix_cols = cols[-n_suffix_cols:] if n_suffix_cols > 0 else []
    mid_cols = cols[n_prefix_cols:len(cols) - n_suffix_cols]

    # Apply MinMaxScaler to middle columns
    scaler = MinMaxScaler(feature_range=feature_range)
    df[mid_cols] = scaler.fit_transform(df[mid_cols])

    return df

if __name__ == "__main__":
    # budgets = [50*i for i in range(1, 21)]
    
    # for budget in budgets:
    #     df = pd.read_csv(f"../data/ela/A1_data_ela_2/A1_B{budget}_5D_ela.csv")
    #     df_norm = minmax_normalize_ela_columns(df)
    #     df_norm.to_csv(f"../data/ela_normalized/A1_B{budget}_5D_ela.csv", index=False)    
    # with warnings.catch_warnings():
    #     warnings.simplefilter("ignore")
    #     extract_a2_precisions(
    #         base_dir="../data/run_data_5D/A2_data_5D",
    #         output_file="../data/A2_precisions_2.csv",
    #         max_evals=1000
    #     )

    # normalize_precision_per_fid(
    #     df = pd.read_csv("../data/A2_precisions_2.csv")
    # ).to_csv("../data/A2_precisions_2_normalized.csv", index=False)

    # add_algorithm_precisions(
    #     ela_dir="../data/ela/A1_data_ela_2_normalized",
    #     precision_csv="../data/A2_precisions_2_normalized.csv",
    #     output_dir="../data/ela/A1_data_ela_2_with_precisions"
    # )

    # df = pd.read_csv("../data/A2_precisions_normalized.csv")

    # # Apply log10 to precisions column
    # df["precision"] = df["precision"].apply(lambda x: np.log10(x))

    # df.to_csv("../data/A2_precisions_normalized_log10.csv", index=False)
    # with warnings.catch_warnings():
    #     warnings.simplefilter("ignore")
    #     extract_a2_precisions(
    #         base_dir="../data/run_data_5D/A2_data_5D_test",
    #         output_file="../data/A2_precisions_test_2.csv",
    #         max_evals=1000
    #     )
    budgets = [8*i for i in range(1, 13) if 8*i != 56] + [50*i for i in range(2, 21)]
    budgets = [56]
    for budget in budgets:
        normalize_test_ela(
            train_csv_path=f"../data/ela/A1_data_ela/A1_B{budget}_5D_ela.csv",
            test_csv_path=f"../data/ela/A1_data_ela_test/A1_B{budget}_5D_ela.csv",
            test_out_path=f"../data/ela/A1_data_ela_test_normalized_2/A1_B{budget}_5D_ela.csv"
        )