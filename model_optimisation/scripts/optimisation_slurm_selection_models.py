import sys
import pandas as pd
import os
import joblib
from asf.selectors import PerformanceModel, tune_selector

def tune_performance_model(budget: int):
    data = pd.read_csv(f"../data/A1_data_5D/A1_B{budget}_5D_ela_with_state.csv")
    # precision_data = pd.read_csv(f"../data/split_precision_csvs/precision_budget_{budget}.csv")
    print(f"Using file: ../data/A1_data_5D/A1_B{budget}_5D_ela_with_state.csv")
    features = data.iloc[:, 4:-6]
    targets = data.iloc[:, -6:]
    groups = data["iid"]

    pipeline = tune_selector(
        X=features,
        y=targets,
        selector_class=[(PerformanceModel, {})],  # model is defined in configspace
        selector_kwargs={"random_state": 42},
        budget=budget,
        maximize=False,
        groups=groups.values,
        cv=5,
        runcount_limit=200,
        seed=42,
        output_dir=f"./smac_output_performance_normalised/B{budget}_performance",
        predict_log=True
    )
    os.makedirs("algo_performance_models_normalised", exist_ok=True)
    joblib.dump(pipeline, f"algo_performance_models_normalised/model_B{budget}.pkl")


def tune_switching_model(budget: int):
    if budget < 100 and budget != 50:
        data = pd.read_csv(f"../data/ela_with_optimal_precisions/A1_data_ela_with_optimal_precisions_early/A1_B{budget}_5D_ela_with_state.csv")
        number_of_predictions = 19 + ( (96 - budget) // 8 ) + 1
    else:
        data = pd.read_csv(f"../data/ela_with_optimal_precisions/A1_data_ela_with_optimal_precisions_late/A1_B{budget}_5D_ela_with_state.csv")
        number_of_predictions = (1000 - budget) // 50 + 1  # Adjusted for the new dataset

    features = data.iloc[:, 4:-number_of_predictions]
    targets = data.iloc[:, -number_of_predictions:]

    print(f"Target cols: {targets.columns.tolist()}")

    groups = data["iid"]

    pipeline = tune_selector(
        X=features,
        y=targets,
        selector_class=[(PerformanceModel, {})],  # model is defined in configspace
        selector_kwargs={"random_state": 42},
        budget=budget,
        maximize=False,
        groups=groups.values,
        cv=5,
        runcount_limit=75,
        seed=42,
        output_dir=f"./smac_output_switching/B{budget}_switching"
    )
    os.makedirs("switching_prediction_models", exist_ok=True)
    joblib.dump(pipeline, f"switching_prediction_models/model_B{budget}.pkl")

# Loads a configured, untrained selection model, trains it and saves the trained model
def train_and_save_selector_only(budget: int):

    input_path = f"algo_performance_models/model_B{budget}.pkl"
    data_path = f"../data/ela_algo/A1_data_5D/A1_B{budget}_5D_ela_with_state.csv"
    save_path = f"../data/trained_models/algo_performance_models_trained/selector_B{budget}_trained.pkl"
    y_cols = -6

    print(f"Loading pipeline: {input_path} and data: {data_path}")
    pipeline = joblib.load(input_path)
    selector = pipeline.selector  # extract selector only

    data = pd.read_csv(data_path)
    features = data.iloc[:, 4:y_cols]
    targets = data.iloc[:, y_cols:]
    print(f"Target columns: {targets.columns.tolist()}")
    features.index = list(zip(data["fid"], data["iid"], data["rep"]))
    targets.index = features.index

    selector.algorithms = list(targets.columns)  
    selector.fit(features, targets)

    print(f"Trained selector on {features.shape[0]} rows")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    joblib.dump(selector, save_path)
    print(f"Saved trained selector to: {save_path}")



if __name__ == "__main__":
    
    budget = int(sys.argv[1])
    # tune_performance_model(budget)
    # elif mode == "switching":
    #     tune_switching_model(budget)
    train_and_save_selector_only(budget)