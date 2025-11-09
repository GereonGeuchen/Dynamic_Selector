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
                            "budget": budget,
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
        if budget == 50: continue
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

def normalize_ela_with_precisions(path_in, path_out):
    df = pd.read_csv(path_in)

    index_cols = ["fid", "iid", "rep"]
    algo_cols = ["BFGS", "DE", "MLSL", "Non-elitist", "PSO", "Elitist"]
    feature_cols = [col for col in df.columns if col not in index_cols + algo_cols]

    # Normalize feature columns globally to [0, 1]
    feature_scaler = MinMaxScaler()
    df_scaled_features = pd.DataFrame(
        feature_scaler.fit_transform(df[feature_cols]),
        columns=feature_cols,
        index=df.index
    )

    # Normalize algorithm columns jointly per fid using 1D flattening
    df_scaled_algos = df[algo_cols].copy()

    for _, group in df.groupby(["fid"]):
        algo_matrix = group[algo_cols].to_numpy()  # shape (num_rows, 6)
        flat_vals = algo_matrix.flatten().reshape(-1, 1)  # shape (num_rows * 6, 1)

        scaler = MinMaxScaler(feature_range=(1e-12, 1))
        flat_scaled = scaler.fit_transform(flat_vals).flatten()

        # Reshape back and insert
        scaled_matrix = flat_scaled.reshape(algo_matrix.shape)
        df_scaled_algos.loc[group.index] = scaled_matrix

    # Combine everything
    df_final = pd.concat([df[index_cols], df_scaled_features, df_scaled_algos], axis=1)
    df_final = df_final.sort_values(by=["fid", "iid", "rep"]).reset_index(drop=True)
    if not os.path.exists(os.path.dirname(path_out)):
        os.makedirs(os.path.dirname(path_out))
    df_final.to_csv(path_out, index=False)
    print(f"Saved normalized file to: {path_out}")

def normalize_and_log_precision_files(precision_path, output_path):
    df = pd.read_csv(precision_path)

    scaler = MinMaxScaler(feature_range=(1e-12, 1))

    def scale_and_log(group):
        group = group.copy()
        # scale in place
        group["precision"] = scaler.fit_transform(group[["precision"]])
        # take log 
        group["precision"] = np.log10(group["precision"])
        return group

    df = df.groupby("fid", group_keys=False).apply(scale_and_log)

    df.to_csv(output_path, index=False)

def normalize_test_ela(train_csv_path, test_csv_path, test_out_path):
    # Load training and test data
    df_train = pd.read_csv(train_csv_path)
    df_test = pd.read_csv(test_csv_path)

    # Define index columns
    index_cols = ["fid", "iid", "rep"]

    # Identify feature columns (anything that's not an index col)
    feature_cols = [col for col in df_train.columns if col not in index_cols]

    # Fit scaler on training data's feature columns
    feature_scaler = MinMaxScaler()
    feature_scaler.fit(df_train[feature_cols])

    # Transform test data's feature columns
    df_scaled_features = pd.DataFrame(
        feature_scaler.transform(df_test[feature_cols]),
        columns=feature_cols,
        index=df_test.index
    )

    # Reattach index columns and save
    df_final = pd.concat([df_test[index_cols], df_scaled_features], axis=1)
    df_final = df_final.sort_values(by=["fid", "iid", "rep"]).reset_index(drop=True)

    if not os.path.exists(os.path.dirname(test_out_path)):
        os.makedirs(os.path.dirname(test_out_path))
    df_final.to_csv(test_out_path, index=False)

    print(f"Saved normalized test file to: {test_out_path}")

if __name__ == "__main__":
    base_data_path = "../data/run_data_5D/A2_data_5D_test"
    # Ignore future warnings for cleaner output
    warnings.filterwarnings("ignore", category=FutureWarning)
    # normalize_and_log_precision_files("A2_precisions.csv", "A2_precisions_normalized_log10.csv")
    # process_ioh_data(base_data_path)
    # add_algorithm_precisions(
    #     ela_dir="../data/ela_with_cma_std/A1_data_5D",
    #     precision_csv="A2_precisions.csv",
    #     output_dir="../data/ela_with_precisions/A1_data_5D"
    # 

    budgets = [8*i for i in range(1, 13)] + [50*i for i in range(2, 21)]

    for budget in budgets:
        normalize_test_ela(
            train_csv_path=f"../data/ela_with_cma_std/A1_data_5D/A1_B{budget}_5D_ela_with_state.csv",
            test_csv_path=f"../data/ela_with_cma_std/A1_data_5D_test/A1_B{budget}_5D_ela_with_state.csv",
            test_out_path=f"../data/ela_normalised/A1_data_5D_test/A1_B{budget}_5D_ela_with_state.csv"
        )