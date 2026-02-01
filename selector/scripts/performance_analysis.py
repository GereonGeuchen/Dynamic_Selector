import pandas as pd 
import numpy as np
from scipy.stats import permutation_test

sbs_sum =  1688.1819# 1836.7794 for classic test data
sum_best_switching = 1254.1039


# Receives a precision file, returns the best-performing static (algorithm, budget) pair
# File has columns (fid,iid,rep,budget,algorithm,precision)
def find_sbs(df: pd.DataFrame):
    # Group by algorithm and budget, compute sum of precisions
    grouped = df.groupby(['algorithm', 'budget'])['precision'].sum().reset_index()
    # Find the row with the minimum precision sum
    best_row = grouped.loc[grouped['precision'].idxmin()]
    print(f"Best static algorithm: {best_row['algorithm']} with budget {best_row['budget']} achieving total precision {best_row['precision']}")

def find_best_switching_sum(df: pd.DataFrame):
    sum_best_switching = 0
    # Go through the results file, in every row find the minimum precision among all static columns, return the sum
    static_cols = [col for col in df.columns if col.startswith("static_B")]
    for index, row in df.iterrows():
        min_precision = min([row[col] for col in static_cols])
        sum_best_switching += min_precision
    print(f"Best switching static total precision: {sum_best_switching}")

def print_performances(df: pd.DataFrame):
    budgets = [50*i for i in range(1, 21)]
    vbs_sum = df["vbs_precisions"].sum()
    print(f"VBS total precision: {vbs_sum}")
    for col in df.columns:
        if col == "selector_precision":
            total_precision = df[col].sum()
            print(f"Selector total precision: {total_precision}")
            print(f"Selector closed gap: {(total_precision - sbs_sum) / (vbs_sum - sbs_sum)}")
            # print(f"Selector closed gap (best switching): {(total_precision - sbs_sum) / (sum_best_switching - sbs_sum)}")
        if col.startswith("static_B"):
            total_precision = df[col].sum()
            budget = col.split("_B")[1]
            print(f"Static budget {budget} total precision: {total_precision}")
            print(f"Static budget {budget} closed gap: {(total_precision - sbs_sum) / (vbs_sum - sbs_sum)}")
            # print(f"Static budget {budget} closed gap (best switching): {(total_precision - sbs_sum) / (sum_best_switching - sbs_sum)}")

if __name__== "__main__":
    df = pd.read_csv("../results/selector_results_with_lookahead_affine_test.csv")
    df_prec = pd.read_csv("../data/A2_precisions_affine_test.csv")
    
        # Columns to convert explicitly
    explicit_cols = ["vbs_precisions", "selector_precision"]

    # Columns starting with "static"
    static_cols = [c for c in df.columns if c.startswith("static")]

    cols_to_convert = explicit_cols + static_cols

    # Convert to float
    df[cols_to_convert] = df[cols_to_convert].apply(
        lambda x: pd.to_numeric(x, errors='coerce')
    )

    # Print sum of precisions of BFGS, 650
    print_performances(df)
    