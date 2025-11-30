import pandas as pd

def compute_best_budgets(path_in, path_out, keep_all_best=True):
    """
    path_in:  input CSV (your file above)
    path_out: output CSV with best budgets + 8-value window
    keep_all_best: 
        True  -> keep all budgets that achieve the minimal precision
        False -> keep only the last (largest) budget among those
    """
    df = pd.read_csv(path_in)

    # 1) For each (fid, iid, rep, budget), take the best (lowest) precision over algorithms
    per_budget = (
        df.groupby(['fid', 'iid', 'rep', 'budget'], as_index=False)['precision']
          .min()
          .rename(columns={'precision': 'min_precision'})
    )

    # 2) For each (fid, iid, rep), select the best budget(s)
    def select_best(group):
        min_prec = group['min_precision'].min()
        best = group[group['min_precision'] == min_prec]

        if not keep_all_best:
            # keep only the last (largest) budget among the best ones
            max_budget = best['budget'].max()
            best = best[best['budget'] == max_budget]

        return best

    best_budgets = (
        per_budget
        .groupby(['fid', 'iid', 'rep'], as_index=False, group_keys=False)
        .apply(select_best)
    )

    # 3) For each chosen budget, compute the next multiple of 8 and the 7 numbers before it
    def add_window(row):
        best_budget = int(row['budget'])

        # next multiple of 8 >= best_budget
        next_mult_8 = ((best_budget + 7) // 8) * 8

        start = next_mult_8 - 7  # seven numbers before
        for i, val in enumerate(range(start, next_mult_8 + 1)):
            row[f'win_{i}'] = val

        # you might also want the multiple-of-8 explicitly named:
        # row['next_multiple_of_8'] = next_mult_8

        return row

    best_budgets = best_budgets.apply(add_window, axis=1)

    # 4) Rename budget column to make it clear
    best_budgets = best_budgets.rename(columns={'budget': 'best_budget'})

    # 5) Save to file
    best_budgets.to_csv(path_out, index=False)

def add_optimal_column(path_runs, path_best_budgets, path_out, window_prefix="win_"):
    """
    path_runs: CSV with columns including
        ['fid', 'iid', 'rep', 'evaluations', ...]
    path_best_budgets: CSV created before, with columns including
        ['fid', 'iid', 'rep', 'best_budget', 'min_precision',
         'next_multiple_of_8', 'win_0', ..., 'win_7']
        (possibly multiple rows per (fid, iid, rep) if keep_all_best=True)
    path_out: where to write the augmented runs CSV
    window_prefix: prefix of window columns in best_budgets (default 'win_')
    """

    # 1) Read input files
    runs = pd.read_csv(path_runs)
    best = pd.read_csv(path_best_budgets)

    # 2) Ensure consistent integer types for keys
    # (your CSV shows iid, rep as floats)
    for col in ['fid', 'iid', 'rep', 'evaluations']:
        runs[col] = runs[col].astype(int)

    for col in ['fid', 'iid', 'rep']:
        best[col] = best[col].astype(int)

    # 3) Collect the window columns (win_0 ... win_7 etc.)
    win_cols = [c for c in best.columns if c.startswith(window_prefix)]
    if not win_cols:
        raise ValueError(f"No window columns starting with '{window_prefix}' found in best_budgets file.")

    # 4) Unroll the window columns into rows:
    #    each row: (fid, iid, rep, evaluations) where evaluations is in a best window
    best_long = best.melt(
        id_vars=['fid', 'iid', 'rep'],
        value_vars=win_cols,
        var_name='window_idx',
        value_name='evaluations'
    )

    best_long = best_long.dropna(subset=['evaluations'])
    best_long['evaluations'] = best_long['evaluations'].astype(int)

    # We only need unique combinations of fid, iid, rep, evaluations
    best_long = best_long[['fid', 'iid', 'rep', 'evaluations']].drop_duplicates()
    best_long['optimal'] = True  # mark these as optimal

    # 5) Merge: left-join the runs with the "optimal evaluations"
    merged = runs.merge(
        best_long,
        on=['fid', 'iid', 'rep', 'evaluations'],
        how='left'
    )

    # 6) Fill NaNs in 'optimal' with False
    merged['optimal'] = merged['optimal'].fillna(False)

    # 7) Save result
    merged.to_csv(path_out, index=False)


add_optimal_column(
    path_runs="../data/A1_B1000_5D_with_current_best.csv",
    path_best_budgets="../data/best_budgets_last.csv",
    path_out="../data/A1_B1000_5D_with_optimal_last.csv"
)