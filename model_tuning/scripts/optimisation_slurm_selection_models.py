import sys
import pandas as pd
import os
import joblib
from asf.selectors import PerformanceModel, tune_selector
from asf.predictors import RandomForestRegressorWrapper
import pathlib
import tempfile
from ConfigSpace import ConfigurationSpace

def tune_performance_model(budget: int):
    data = pd.read_csv(f"../data/A1_data_ela_normalized_with_precisions/A1_B{budget}_5D_ela.csv")
    # precision_data = pd.read_csv(f"../data/split_precision_csvs/precision_budget_{budget}.csv")
    print(f"Using file: ../data/A1_data_ela_normalized_with_precisions/A1_B{budget}_5D_ela.csv")
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
        runcount_limit=100,
        seed=42,
        output_dir=f"./smac_output_performance_no_state_2/B{budget}_performance",
        predict_log=True
    )
    os.makedirs("algo_performance_models_no_state_2", exist_ok=True)
    joblib.dump(pipeline, f"algo_performance_models_no_state_2/model_B{budget}.pkl")
    
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

    input_path = f"../data/models/untrained_models/algo_performance_models_algo_features/model_B{budget}.pkl"
    data_path = f"../data/A1_data_ela_normalized_with_precisions/A1_B{budget}_5D_ela.csv"
    save_path = f"../data/models/trained_models/algo_performance_models_trained_algo_features/selector_B{budget}_trained.pkl"
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

def rf_constructor(**kwargs):
    return RandomForestRegressorWrapper(
        init_params={"random_state": 42, **kwargs},
    )

def atomic_joblib_dump(obj, final_path: str, compress=0):
    final_path = pathlib.Path(final_path)
    final_path.parent.mkdir(parents=True, exist_ok=True)

    fd, tmp_name = tempfile.mkstemp(
        dir=final_path.parent,
        prefix=final_path.name + ".",
        suffix=".tmp",
    )
    os.close(fd)
    tmp_path = pathlib.Path(tmp_name)

    try:
        joblib.dump(obj, tmp_path, compress=compress)

        with open(tmp_path, "rb") as f:
            os.fsync(f.fileno())

        os.replace(tmp_path, final_path)
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass

def make_default_performance_model():
    cs = ConfigurationSpace()
    cs_transform = {}

    cs, cs_transform = PerformanceModel.get_configuration_space(
        cs=cs,
        cs_transform=cs_transform,
        parent_param=None,
        parent_value=str(PerformanceModel.__name__),
    )

    default_config = cs.get_default_configuration()

    return PerformanceModel.get_from_configuration(
        default_config,
        cs_transform=None,
        random_state=42,
    )



def train_default_selector(budget: int):
    data_path = f"../data/selection_data/auc/A1_data_ela_normalised_with_aucs/A1_B{budget}_5D_ela.csv"
    save_path = f"../data/models/trained_models/auc/algo_performance_models_trained_untuned_auc/selector_B{budget}_untuned_trained.pkl"
    y_cols = -6

    print(f"Loading data: {data_path}")
    data = pd.read_csv(data_path)
    features = data.iloc[:, 4:y_cols]
    targets = data.iloc[:, y_cols:]
    print(f"Target columns: {targets.columns.tolist()}")
    features.index = list(zip(data["fid"], data["iid"], data["rep"]))
    targets.index = features.index

    selector = make_default_performance_model()
    selector.algorithms = list(targets.columns)
    selector.fit(features, targets)

    print(f"Trained default selector on {features.shape[0]} rows")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    atomic_joblib_dump(selector, save_path)
    print(f"Saved default trained selector to: {save_path}")

if __name__ == "__main__":
    
    budget = int(sys.argv[1])
    # tune_performance_model(budget)
    # elif mode == "switching":
    #     tune_switching_model(budget)
    train_default_selector(budget)

    # cs = PerformanceModel.get_configuration_space()
    # print(cs.get_default_configuration())

    # selector = PerformanceModel.get