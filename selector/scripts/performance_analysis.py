import pandas as pd 

if __name__== "__main__":
    result_csv = "../results/selector_results_algo_per_run_highest.csv"
    best_identified_budgets = "../data/best_budgets.csv"
    # print("test")
    results = pd.read_csv(result_csv)
    for col in results.columns:
        if col == "selector_precision":
            print("Selector: ", results[col].sum())
        elif col.startswith("static"):
            print(f"{col}: ", results[col].sum())

    # precision_csv = "../data/A2_precisions_test.csv"
    # budgets = [8*i for i in range(1, 13)] + [50*i for i in range(2, 21)]
    # algos = ["Elitist", "Non-elitist", "MLSL", "BFGS", "PSO", "DE"]
    # for algo in algos:
    #     for budget in budgets:
    #         # if algo != "BFGS": continue
    #         df_filtered = pd.read_csv(precision_csv)
    #         df_filtered = df_filtered[df_filtered["budget"] == budget]
    #         df_filtered = df_filtered[df_filtered["algorithm"] == algo]
    #         total_precision = df_filtered["precision"].sum()
    #         print(f"Budget: {budget}, Algo: {algo}, Total Precision: {total_precision}")

    # Go through each line, find the lowest precision among all static selectors, sum those precisions
    # results = pd.read_csv(result_csv)
    # static_cols = [col for col in results.columns if col.startswith("static")]
    # total_min_static_precision = 0
    # for index, row in results.iterrows():
    #     min_precision = row[static_cols].min()
    #     total_min_static_precision += min_precision
    # print("Total Minimum Static Selector Precision: ", total_min_static_precision)

    # For each fid in the results file, find the corresponding best budget from best_identified_budgets file and sum the precisions, return overall sum
 
    # best_budgets = pd.read_csv(best_identified_budgets)

    # # Make sure types match
    # results["fid"] = results["fid"].astype(int)
    # best_budgets["fid"] = best_budgets["fid"].astype(int)
    # best_budgets["best_budget"] = best_budgets["best_budget"].astype(int)

    # # 1) For each fid, choose the row in best_budgets with the highest total_precision
    # best_budgets_unique = (
    #     best_budgets
    #     .sort_values(["fid", "total_precision"], ascending=[True, False])
    #     .drop_duplicates(subset="fid")   # keep best (highest total_precision) per fid
    # )

    # total_best_budget_precision = 0.0

    # # 2) For each fid, look up the static_<best_budget> column in results and sum it
    # for _, bb_row in best_budgets_unique.iterrows():
    #     fid = int(bb_row["fid"])
    #     best_budget = int(bb_row["best_budget"])
    #     col = f"static_B{best_budget}"

    #     if col not in results.columns:
    #         print(f"Warning: column {col} not found in results; skipping fid {fid}")
    #         continue

    #     # sum precision over all rows in 'results' that have this fid at this budget
    #     fid_sum = results.loc[results["fid"] == fid, col].sum()
    #     total_best_budget_precision += fid_sum

    # print("Total Best Budget Static Selector Precision:", total_best_budget_precision)