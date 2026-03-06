import pandas as pd 
import numpy as np
from scipy.stats import permutation_test

sbs_sum =  1836.7794 # 515.87  For affine, 6 to 7 # 603.455 for affine, 1 to 5 # 1688.1819 for affine across all # 1836.7794 for classic test data
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
    # print(f"VBS total precision: {vbs_sum}")
    for col in df.columns:
        if col == "selector_precision":
            total_precision = df[col].sum()
            print(f"Selector total precision: {total_precision}")
            print(f"Selector closed gap: {(total_precision - sbs_sum) / (vbs_sum - sbs_sum)}")
            # print(f"Selector closed gap (best switching): {(total_precision - sbs_sum) / (sum_best_switching - sbs_sum)}")
        # if col.startswith("static_B"):
        #     total_precision = df[col].sum()
        #     budget = col.split("_B")[1]
        #     print(f"Static budget {budget} total precision: {total_precision}")
        #     print(f"Static budget {budget} closed gap: {(total_precision - sbs_sum) / (vbs_sum - sbs_sum)}")
            # print(f"Static budget {budget} closed gap (best switching): {(total_precision - sbs_sum) / (sum_best_switching - sbs_sum)}")

if __name__== "__main__":
    # for i in range(0, 20):
    #     df = pd.read_csv(f"../results/all_epms/selector_results_with_lookahead_all_epms_{i}.csv")
    #     print(f"=== Results for lookahead with {i} EPMs ===")
    #     print_performances(df)
    
    # for i in range(0, 20):
    #     df = pd.read_csv(f"../results/all_epms_algo_features/selector_results_with_lookahead_all_epms_algo_features_{i}.csv")
    #     print(f"=== Results for lookahead with {i} EPMs and algo features ===")
    #     print_performances(df)
    df = pd.read_csv(f"../results/selector_results_with_algo_features.csv")
    print_performances(df)