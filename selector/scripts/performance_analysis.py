import pandas as pd 

if __name__== "__main__":
    result_csv = "../results/selector_results.csv"
    print("test")
    results = pd.read_csv(result_csv)
    for col in results.columns:
        if col == "selector_precision":
            print("Selector: ", results[col].sum())
        elif col.startswith("static"):
            print(f"{col}: ", results[col].sum())

    precision_csv = "../data/A2_precisions_test.csv"
    budgets = [8*i for i in range(1, 13)] + [50*i for i in range(2, 21)]
    algos = ["Elitist", "Non-elitist", "MLSL", "BFGS", "PSO", "DE"]
    for budget in budgets:
        for algo in algos:
            if algo != "BFGS": continue
            df_filtered = pd.read_csv(precision_csv)
            df_filtered = df_filtered[df_filtered["budget"] == budget]
            df_filtered = df_filtered[df_filtered["algorithm"] == algo]
            total_precision = df_filtered["precision"].sum()
            print(f"Budget: {budget}, Algo: {algo}, Total Precision: {total_precision}")