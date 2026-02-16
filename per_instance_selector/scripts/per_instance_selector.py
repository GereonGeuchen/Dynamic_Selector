# Simulation of the per-instance selector

from logging import warning
import os
import pandas as pd
import joblib
import warnings


ALGORITHMS = ["BFGS", "DE", "Elitist", "MLSL", "Non-elitist", "PSO"]
ELA_PATH = "../data/ela_lhs_150_all_reps_test.csv"
MODEL_DIRECTORY = "../data/models/per_instance_selector_models_150_all_reps_trained"
PRECISION_PATH = "../data/A2_precisions_scratch_850_test.csv"
OUTPUT_PATH = "../data/results"

if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)

def main():
    # 1. Load the models
    models = {}
    for algo in ALGORITHMS:
        model_path = os.path.join(MODEL_DIRECTORY, f"{algo}_rf_trained.pkl")
        if os.path.exists(model_path):
            models[algo] = joblib.load(model_path)
            print(f"Loaded model for {algo} from {model_path}, {models[algo].model_class.get_params()}")
        else:
            print(f"Model for {algo} not found at {model_path}")

    # 2. Load the ELA features and precision data
    if os.path.exists(ELA_PATH):
        ela_data = pd.read_csv(ELA_PATH)
        print(f"Loaded ELA features from {ELA_PATH}")
    else:
        print(f"ELA features not found at {ELA_PATH}")
        return
    
    if os.path.exists(PRECISION_PATH):
        precision_data = pd.read_csv(PRECISION_PATH)
        print(f"Loaded precision data from {PRECISION_PATH}")
    else:
        print(f"Precision data not found at {PRECISION_PATH}")
        return
    
    # 3. Make predictions and evaluate: For each (fid,iid,rep) in the ELA file, store one row 
    #    in the output file with the following columns: fid, iid, rep, predicted_algo, precision, precisions of all algorithms
    results = []
    for index, row in ela_data.iterrows():
        fid, iid, rep = row["fid"], row["iid"], row["rep"]
        # if rep != 0: continue
        features = row.drop(["fid", "iid", "rep", "high_level_category"]).values.reshape(1, -1)
        
        precision_values = {algo: precision_data[(precision_data["fid"] == fid) & (precision_data["iid"] == iid) & (precision_data["rep"] == rep)
                             & (precision_data["algorithm"] == algo)]["precision"].values[0] for algo in ALGORITHMS}
        
        # Make predictions with each model and select the best algorithm
        predicted_algo = None
        best_precision = float("inf")
        
        for algo, model in models.items():
            predicted_precision = model.predict(features)[0]
            if predicted_precision < best_precision:
                best_precision = predicted_precision
                predicted_algo = algo
        
        results.append({
            "fid": fid,
            "iid": iid,
            "rep": rep,
            "predicted_algo": predicted_algo,
            "precision": precision_values[predicted_algo],
            **precision_values
        })

        # Save results to a CSV file
        results_df = pd.DataFrame(results)
        output_file = os.path.join(OUTPUT_PATH, "per_instance_selector_results_150_all_reps.csv")
        results_df.to_csv(output_file, index=False)

if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()