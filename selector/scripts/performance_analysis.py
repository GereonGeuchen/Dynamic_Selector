import pandas as pd 
import numpy as np

if __name__== "__main__":
    df = pd.read_csv("../results/selector_results_with_algo_features_variance.csv")
    for col in df.columns:
        if col == "selector_precision":
            print(f"Selector: {df[col].sum()}")
        if col.startswith("static_B"):
            print(f"{col}: {df[col].sum()}")