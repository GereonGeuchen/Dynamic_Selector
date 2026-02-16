import pandas as pd

algos = ["BFGS", "DE", "Elitist", "MLSL", "Non-elitist", "PSO"]
df = pd.read_csv("../data/results/per_instance_selector_results_150_all_reps.csv") 
print("Overall performance:", df["precision"].sum())
for algo in algos:
    print(f"Performance of {algo}: {df[algo].sum()}")