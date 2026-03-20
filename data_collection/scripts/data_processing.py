import os
import pandas as pd
import ioh
from ioh import ProblemClass
import warnings
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from pathlib import Path

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

def process_ioh_data_affine(base_path, dim=5):
    allowed_types = {"1to5", "6to7", "1to7"}
    x_cols = [f"x{i}" for i in range(dim)]
    numeric_cols = ["evaluations", "raw_y", "rep"] + x_cols

    for budget_dir in os.listdir(base_path):
        budget_path = os.path.join(base_path, budget_dir)
        if not os.path.isdir(budget_path):
            continue

        all_rows = []

        for run_dir in os.listdir(budget_path):
            run_path = os.path.join(budget_path, run_dir)
            if not os.path.isdir(run_path):
                continue

            # Expect: data_f{fid}_affine_test_problem_{type}_{inst}
            if not run_dir.startswith("data_f"):
                continue

            parts = run_dir.split("_")
            # ["data", "f1121", "affine", "test", "problem", "1to5", "0"]
            if len(parts) != 7:
                continue
            if parts[0] != "data" or not parts[1].startswith("f"):
                continue
            if parts[2:5] != ["affine", "test", "problem"]:
                continue

            fid_str = parts[1][1:]  # after 'f'
            if not fid_str.isdigit():
                continue
            fid = int(fid_str)

            combo_type = parts[5]
            if combo_type not in allowed_types:
                continue

            inst_str = parts[6]
            if not inst_str.isdigit():
                continue
            # inst_id = int(inst_str)  # ignored

            dat_file = os.path.join(run_path, f"IOHprofiler_f{fid}_DIM{dim}.dat")
            if not os.path.isfile(dat_file):
                continue

            try:
                df = pd.read_csv(dat_file, delim_whitespace=True, comment="#", dtype=str)
            except Exception as e:
                print(f"Error reading {dat_file}: {e}")
                continue

            # Some IOH logs can contain repeated header rows; filter defensively if needed
            # (Your previous code used df[df['iid'] != 'iid']; we won't rely on iid now.)
            if "rep" in df.columns:
                df = df[df["rep"] != "rep"]

            # Ensure needed columns exist
            missing = [c for c in (["rep", "evaluations", "raw_y"] + x_cols) if c not in df.columns]
            if missing:
                print(f"Skipping {dat_file}: missing columns {missing}")
                continue

            # Convert numeric columns
            df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
            df = df.dropna(subset=numeric_cols)

            # Add index columns
            df["fid"] = fid
            df["type"] = combo_type

            # true_y is just raw_y (no problem lookup)
            df["true_y"] = df["raw_y"]

            # Keep only what you want
            keep_cols = ["fid", "type", "rep", "evaluations", "raw_y", "true_y"] + x_cols
            all_rows.append(df[keep_cols])

        if all_rows:
            combined = pd.concat(all_rows, ignore_index=True)

            # Sort rows by your new index
            combined = combined.sort_values(
                by=["fid", "type", "rep", "evaluations"]
            ).reset_index(drop=True)

            output_path = os.path.join(base_path, f"{budget_dir}.csv")
            combined.to_csv(output_path, index=False)
            print(f"Saved: {output_path}")
        else:
            print(f"No data found for budget dir: {budget_dir}")

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

    total_reps = 0
    total_oob_reps = 0

    for algo in algorithms:
        for budget in budgets:
            print(f"Processing algorithm={algo}, budget={budget}...")
            # if budget != 56:
            #     if budget % 50 != 0: continue
            folder_name = os.path.join(base_dir, f"A2_{algo}_B{budget}_5D")
            if not os.path.isdir(folder_name):
                print(f"Directory not found: {folder_name}")
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
                        
                        total_reps += 1

                        # Step 2: Filter to in-bound rows only
                        x_cols = [f'x{i}' for i in range(dim)]
                        in_bounds = subset[
                            subset[x_cols].apply(
                                lambda row: all(lower_bound <= row[x] <= upper_bound for x in x_cols),
                                axis=1
                            )
                        ]

                        # Check if ANY evaluation in this rep violates bounds
                        row_oob = (subset[x_cols] < lower_bound).any(axis=1) | \
                                (subset[x_cols] > upper_bound).any(axis=1)

                        if row_oob.any():
                            total_oob_reps += 1

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

                            
    # result_df = pd.DataFrame(results)
    # result_df.sort_values(by=["fid", "iid", "rep", "budget"], inplace=True)
    # result_df.to_csv(output_file, index=False)
    print(f"Total reps processed: {total_reps}")
    print(f"Total out-of-bounds reps: {total_oob_reps}")
    # return result_df

# Function that creates the A2_precisions.csv file from the run data.
def extract_a2_precisions_affine(
        base_dir, output_file="A2_precisions_affine.csv", algorithms=None, budgets=None, fids=None, max_evals=1000,
        dim=5, lower_bound=-5, upper_bound=5):
    print(f"Extracting A2 affine precisions from {base_dir} with max_evals={max_evals}...")

    if algorithms is None:
        algorithms = ["BFGS", "DE", "MLSL", "Non-elitist", "PSO", "Elitist"]
    if budgets is None:
        budgets = [50*i for i in range(1, 21)]

    # If not provided, use your described fid range
    if fids is None:
        fids = range(1121, 1166)
    fids_set = set(fids)

    allowed_types = {"1to5", "6to7", "1to7"}
    x_cols = [f"x{i}" for i in range(dim)]

    results = []

    for algo in algorithms:
        for budget in budgets:
            print(f"Processing algorithm={algo}, budget={budget}...")
            outer_folder = os.path.join(base_dir, f"A2_{algo}_B{budget}_5D")
            if not os.path.isdir(outer_folder):
                print(f"Directory not found: {outer_folder}")
                continue

            for inner in os.listdir(outer_folder):
                inner_path = os.path.join(outer_folder, inner)
                if not os.path.isdir(inner_path):
                    continue

                # Expect: data_f{fid}_affine_test_problem_{type}_{inst}
                if not inner.startswith("data_f"):
                    continue

                parts = inner.split("_")
                # Example parts:
                # ["data", "f1121", "affine", "test", "problem", "1to5", "0"]
                if len(parts) != 7:
                    continue
                if parts[0] != "data" or not parts[1].startswith("f"):
                    continue
                if parts[2:5] != ["affine", "test", "problem"]:
                    continue

                fid_str = parts[1][1:]  # after 'f'
                if not fid_str.isdigit():
                    continue
                fid = int(fid_str)
                if fid not in fids_set:
                    continue

                combo_type = parts[5]
                if combo_type not in allowed_types:
                    continue

                inst_str = parts[6]
                if not inst_str.isdigit():
                    continue
                # inst_id = int(inst_str)  # intentionally ignored

                file_path = os.path.join(inner_path, f"IOHprofiler_f{fid}_DIM5.dat")
                if not os.path.isfile(file_path):
                    continue

                try:
                    df = pd.read_csv(file_path, delim_whitespace=True, comment="%")
                    df["evaluations"] = pd.to_numeric(df["evaluations"], errors="coerce")
                    df["raw_y"] = pd.to_numeric(df["raw_y"], errors="coerce")
                    df["rep"] = pd.to_numeric(df["rep"], errors="coerce", downcast="integer")

                    for col in x_cols:
                        if col in df.columns:
                            df[col] = pd.to_numeric(df[col], errors="coerce")

                    df = df.dropna(subset=["evaluations", "raw_y", "rep"] + x_cols)
                except Exception as e:
                    print(f"Failed to read {file_path}: {e}")
                    continue

                for rep, group in df.groupby("rep"):
                    subset = group[group["evaluations"] <= max_evals]
                    if subset.empty:
                        continue

                    # Filter to in-bounds rows only
                    in_bounds = subset[
                        subset[x_cols].apply(
                            lambda row: all(lower_bound <= row[c] <= upper_bound for c in x_cols),
                            axis=1
                        )
                    ]
                    if in_bounds.empty:
                        continue

                    best_precision = float(in_bounds["raw_y"].min())

                    results.append({
                        "fid": fid,
                        "type": combo_type,
                        "rep": int(rep),
                        "budget": 50 if budget == 56 else budget,
                        "algorithm": algo,
                        "precision": best_precision,
                    })

    result_df = pd.DataFrame(results)
    if result_df.empty:
        result_df = pd.DataFrame(columns=["fid", "type", "rep", "budget", "algorithm", "precision"])
        result_df.to_csv(output_file, index=False)
        return result_df

    # Disregard instance (folder suffix): keep best across inst 0..14
    result_df = (
        result_df
        .groupby(["fid", "type", "rep", "budget", "algorithm"], as_index=False)["precision"]
        .min()
    )

    result_df.sort_values(by=["fid", "type", "rep", "budget", "algorithm"], inplace=True)
    result_df.to_csv(output_file, index=False)
    return result_df


def add_algorithm_precisions(ela_dir, precision_csv, output_dir):
    """Adds algorithm precisions to ELA feature files. Used to create the dataset with which the selection models are trained.
    """
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
    """This function normalizes ELA features using min-max scaling, 
        and algorithm precisions using min-max scaling per fid across all algorithms."""
    df = pd.read_csv(path_in)

    # Columns that must NOT be normalized
    index_cols = ["fid", "iid", "high_level_category", "rep"]
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
    Used to create the ELA files for the test set. That is, it trains the min-max scalers on the training set,
    and uses the trained scalers to normalize the test set.
    """

    # Load training and test data
    df_train = pd.read_csv(train_csv_path)
    df_test = pd.read_csv(test_csv_path)

    # Index columns (never scaled)
    index_cols = ["fid", "iid", "rep", "high_level_category"]
    algo_cols = ["BFGS", "DE", "Elitist", "MLSL", "Non-elitist", "PSO"]
    all_cols = df_train.columns.tolist()

    # Determine which columns to normalize
    if norm_ranges is None:
        # Backwards-compatible behaviour: scale all non-index columns
        feature_cols = [col for col in df_train.columns if col not in index_cols and col not in algo_cols]
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

    # Fit scaler on the training data
    scaler = MinMaxScaler()
    scaler.fit(df_train[feature_cols])

    # Apply scaler to the test data
    df_final = df_test.copy()
    df_final[feature_cols] = scaler.transform(df_test[feature_cols])
    df_final[feature_cols] = df_final[feature_cols].astype(float)

    # Sort & save
    df_final = df_final.sort_values(by=["fid", "iid", "rep"]).reset_index(drop=True)

    out_dir = os.path.dirname(test_out_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir)

    df_final.to_csv(test_out_path, index=False)
    print(f"Saved normalized test file to: {test_out_path}")

def add_current_best(df: pd.DataFrame, gen_size: int = 8):
    """Add 'current_best' column to csvs containing the raw evaluations. Used for "gradient" features.
       """
    df = df.copy()

    def process_one_run(run_df):
        run_df = run_df.copy()
        run_df["generation"] = (run_df["evaluations"] - 1) // gen_size

        gen_min = run_df.groupby("generation")["raw_y"].min()
        gen_best_cum = gen_min.cummin()

        run_df["current_best"] = run_df["generation"].map(gen_best_cum)

        return run_df.drop(columns=["generation"])

    return df.groupby(["fid", "iid", "rep"], group_keys=False).apply(process_one_run)

def normalize_precision_per_fid(df, col="precision", min_scale=1e-12, max_scale=1.0):
    """Normalizes precision files per fid using min-max scaling. Used to create the precision files
       for when we include selection model predictions as features.
    """
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

def attach_future_best_precisions(
    raw_ela_folder: str | Path,
    best_precisions_csv: str | Path,
    n_future: int = 20,
) -> list[Path]:
    """
    Attach the best precisions of the next n switching points to each ELA file. Used for the lookahaed EPM.
    """
    raw_ela_folder = Path(raw_ela_folder)
    best_precisions_csv = Path(best_precisions_csv)

    out_folder = raw_ela_folder.parent / "A1_data_ela_normalized_with_future_performances_20_afterwards"
    out_folder.mkdir(parents=True, exist_ok=True)

    # --- build lookahead table ---
    best = pd.read_csv(best_precisions_csv)
    best = best.sort_values(["fid", "iid", "rep", "budget"]).reset_index(drop=True)

    g = best.groupby(["fid", "iid", "rep"], sort=False)
    for k in range(0, n_future + 1):
        best[f"best_precision_t+{k}"] = g["best_precision"].shift(-k)

    future_cols = (
        ["fid", "iid", "rep", "budget"]
        + [f"best_precision_t+{k}" for k in range(0, n_future + 1)]
    )
    future_table = best[future_cols]

    written: list[Path] = []

    # --- attach to each ELA file ---
    for ela_path in raw_ela_folder.glob("A1_B*_5D_ela.csv"):
        name = ela_path.stem  # e.g. "A1_B50_5D_ela"
        parts = name.split("_")

        # expected: ["A1", "B50", "5D", "ela"]
        try:
            budget = int(parts[1][1:])  # strip leading 'B'
        except (IndexError, ValueError):
            continue  # skip unexpected filenames

        ela = pd.read_csv(ela_path)
        ela["budget"] = budget  # temporary, for merge only

        ela_aug = ela.merge(
            future_table,
            on=["fid", "iid", "rep", "budget"],
            how="left",
        )

        # always drop budget again
        ela_aug = ela_aug.drop(columns=["budget"])
        future_colnames = [f"best_precision_t+{k}" for k in range(0, n_future + 1)]
        drop_cols = [c for c in future_colnames if c in ela_aug.columns and ela_aug[c].isna().all()]
        ela_aug = ela_aug.drop(columns=drop_cols)

        out_path = out_folder / ela_path.name
        ela_aug.to_csv(out_path, index=False, na_rep="")  # blanks instead of NaNs
        written.append(out_path)

    return written


def aggregate_precision_by_budget_algorithm(
    input_df: pd.DataFrame,
    output_path: str
):

    result = (
        input_df.groupby(["budget", "algorithm"], as_index=False)["precision"]
          .sum()
          .sort_values("precision", ascending=True)
    )

    result.to_csv(output_path, index=False)
    return result

def aggregate_precision_by_budget_algorithm(
    input_df: pd.DataFrame,
    output_path: str
):

    df_agg = (
        input_df
        .groupby(["fid", "iid", "algorithm", "budget"], as_index=False)
        .agg(
            precision=("precision", "sum")
        )
    )

    df_agg.to_csv(output_path, index=False)

def aggregate_ela_median(
    input_df: pd.DataFrame,
    output_path: str,
    index_cols=["fid", "iid", "high_level_category"]
):

    feature_cols = [c for c in input_df.columns if c not in index_cols]

    df_agg = (
        input_df
        .groupby(index_cols, as_index=False)
        .agg({col: "median" for col in feature_cols})
    )

    # Drop rep-column
    df_agg = df_agg.drop(columns=["rep"], errors="ignore")

    df_agg.to_csv(output_path, index=False)

def label_lhs_ela_with_precisions(
        ela_csv_path: str,
    precision_csv_path: str,
    out_csv_path: str,
    ):
    """
    Loads an ELA CSV with one row per (fid, iid), and a precision CSV with rows
    (fid, iid, algorithm, budget, precision). Ignores budget, pivots algorithm
    into columns, and merges onto the ELA rows.

    Output columns = ELA columns + one column per algorithm (e.g., BFGS, DE, PSO, ...).
    """
    df_ela = pd.read_csv(ela_csv_path)
    df_prec = pd.read_csv(precision_csv_path)
    df_ela = df_ela[df_ela["rep"] == 0]
    df_prec = df_prec[df_prec["rep"] == 0]
    # Pivot precision file: (fid, iid) index, algorithm columns, precision values
    df_prec_wide = (
        df_prec
        .pivot_table(
            index=["fid", "iid"],
            columns="algorithm",
            values="precision",
        )
        .reset_index()
    )

    # Merge onto ELA
    df_labeled = df_ela.merge(
        df_prec_wide,
        on=["fid", "iid"],
        how="left"
    )

    # Write result
    os.makedirs(os.path.dirname(out_csv_path), exist_ok=True)
    df_labeled.to_csv(out_csv_path, index=False)

def add_algorithm_precisions_lhs(ela_path, precision_path):
    ela_df = pd.read_csv(ela_path)
    prec_df = pd.read_csv(precision_path)
    # Here, we also want to match reps, not just for rep = 0
    # First, pivot the precision file to have algorithms as columns
    prec_wide = (
        prec_df.pivot_table(
            index=["fid", "iid", "rep"],
            columns="algorithm",
            values="precision"
        )
        .reset_index()
    )

    # Merge onto ELA
    df_labeled = ela_df.merge(
        prec_wide,
        on=["fid", "iid", "rep"],
        how="left"
    )

    # Write result
    out_path = ela_path.replace(".csv", "_with_precisions.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df_labeled.to_csv(out_path, index=False)

if __name__ == "__main__":
    # attach_future_best_precisions(
    #     raw_ela_folder="../data/ela/A1_data_ela_normalized",
    #     best_precisions_csv="../data/A2_best_normalized_precisions.csv",
    #     n_future=20,
    # )
    # precision_df = pd.read_csv("../data/A2_precisions.csv")
    # # for each (fid,iid,rep,budget), find lowest precisoin across algorithms, and save as A2_best_precisions.csv
    # best_df = (
    #     precision_df
    #     .groupby(["fid", "iid", "rep", "budget"], as_index=False)
    #     .agg(best_precision=("precision", "min"))
    # )
    # best_df.to_csv("../data/A2_best_precisions.csv", index=False)
    attach_future_best_precisions(
        raw_ela_folder="../data/ela/A1_data_ela_normalized",
        best_precisions_csv="../data/A2_best_afterwards_normalized_precisions.csv",
        n_future=20,
    )