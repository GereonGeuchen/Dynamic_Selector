"""
This file contains the code for the dynamic selector.
"""

import argparse
import joblib
import pandas as pd
import os

from asf.predictors import RandomForestClassifierWrapper, RandomForestRegressorWrapper
from asf.selectors import PerformanceModel
from ConfigSpace import ConfigurationSpace

from sklearn.preprocessing import MinMaxScaler

# Experiment setup. Changing these constants changes the data folders and column
# names that the selector reads from and writes to.
TOTAL_BUDGET = 1000
BUDGET_STEP = 50
DIM = 5
METRIC = "regret"

METRIC_COLUMN = f"achieved_{METRIC}"
ACTUAL_METRIC_COLUMN = f"actual_{METRIC}"
PREDICTED_METRIC_COLUMN = f"predicted_{METRIC}"
VBS_METRIC_COLUMN = f"vbs_{METRIC}"
SWITCHING_BUDGETS = [BUDGET_STEP * i for i in range(1, TOTAL_BUDGET // BUDGET_STEP + 1)]
TRAINING_SWITCHING_BUDGETS = [budget for budget in SWITCHING_BUDGETS if budget != TOTAL_BUDGET]
LOOKAHEAD_TARGET_PREFIX = "t_"

TRAIN_IIDS = [1, 2, 3, 4, 5]
TEST_IIDS = [6, 7]

NO_SWITCH_ALGORITHM = "Non-elitist"
ALGORITHMS = [NO_SWITCH_ALGORITHM, "Elitist", "PSO", "DE", "BFGS", "MLSL"]
SWITCH_ALGORITHMS = [algo for algo in ALGORITHMS if algo != NO_SWITCH_ALGORITHM]

KEY_COLS = ["fid", "iid", "rep"]
META_COLS = ["fid", "iid", "rep", "high_level_category"]

# Folder names are centralized so cached training data and final model training
# cannot accidentally drift to different directories.
SELECTION_TRAINING_DATA_FOLDER = "selection_model_training_data"
SWITCHING_TRAINING_DATA_FOLDER = "switching_model_training_data"
LOOKAHEAD_TRAINING_DATA_FOLDER = "lookahead_model_training_data"

def metric_scoped_path(base_path: str) -> str:
    """Return a metric-specific subdirectory under a base path, e.g. ./data/regret."""
    normalized = os.path.normpath(base_path)
    if os.path.basename(normalized) == METRIC:
        return normalized
    return os.path.join(base_path, METRIC)

# Path helpers encode the file naming convention created by data_collection.py.
def achieved_metric_path(data_path: str, algorithm: str, budget: int) -> str:
    return os.path.join(data_path, f"achieved_{METRIC}s/achieved_{METRIC}s_{algorithm}_B{budget}_{DIM}D.csv")

def ela_features_path(data_path: str, algorithm: str, budget: int) -> str:
    return os.path.join(data_path, f"ela_features/{algorithm}_B{budget}_{DIM}D/ELA_features.csv")

def selection_training_data_path(data_path: str, budget: int) -> str:
    return os.path.join(data_path, SELECTION_TRAINING_DATA_FOLDER, f"selection_model_training_data_budget_{budget}.csv")

def switching_training_data_path(data_path: str, budget: int) -> str:
    return os.path.join(data_path, SWITCHING_TRAINING_DATA_FOLDER, f"switching_model_training_data_budget_{budget}.csv")

def lookahead_training_data_path(data_path: str, budget: int) -> str:
    return os.path.join(data_path, LOOKAHEAD_TRAINING_DATA_FOLDER, f"lookahead_model_training_data_budget_{budget}.csv")

def is_no_switch_algorithm(algorithm: str) -> bool:
    return algorithm == NO_SWITCH_ALGORITHM

def switch_budget_for_algorithm(algorithm: str, switch_budget: int) -> int:
    # The no-switch baseline is stored at the total budget, not at every candidate switch budget.
    if is_no_switch_algorithm(algorithm):
        return TOTAL_BUDGET
    return switch_budget

def get_metric_value(metrics: pd.DataFrame, fid: int, iid: int, rep: int, algorithm: str, switch_budget: int):
    metric_budget = switch_budget_for_algorithm(algorithm, switch_budget)
    return metrics[
        (metrics["fid"] == fid) &
        (metrics["iid"] == iid) &
        (metrics["rep"] == rep) &
        (metrics["a1_budget"] == metric_budget) &
        (metrics["algname"] == algorithm)
    ][METRIC_COLUMN].values[0]

def selection_feature_cols(data: pd.DataFrame) -> list[str]:
    return [col for col in data.columns if col not in META_COLS + ALGORITHMS]

def lookahead_feature_cols(data: pd.DataFrame) -> list[str]:
    return [col for col in data.columns if col not in META_COLS and not col.startswith(LOOKAHEAD_TARGET_PREFIX)]

def lookahead_target_cols(data: pd.DataFrame) -> list[str]:
    return [col for col in data.columns if col.startswith(LOOKAHEAD_TARGET_PREFIX)]

def switching_feature_cols(data: pd.DataFrame) -> list[str]:
    return [col for col in data.columns if col not in META_COLS + ["optimal_budget"]]

def drop_all_nan_ela_columns(ela_features: pd.DataFrame) -> pd.DataFrame:
    """Remove ELA columns that are entirely missing in the dataframe."""
    return ela_features.dropna(axis=1, how="all")

def train_lookahead_models(lookahead_model_training_data: pd.DataFrame) -> dict:
    # One regressor is trained per lookahead target t_0, t_1, ...
    feature_cols = lookahead_feature_cols(lookahead_model_training_data)
    X_train = lookahead_model_training_data[feature_cols].copy()

    lookahead_models = {}
    for target_col in lookahead_target_cols(lookahead_model_training_data):
        lookahead_model = make_default_wrapper_model(wrapper_type="RandomForestRegressorWrapper")
        lookahead_model.fit(X_train, lookahead_model_training_data[target_col])
        lookahead_models[target_col] = lookahead_model

    return lookahead_models

def add_lookahead_predictions(ela_features: pd.DataFrame, lookahead_models: dict) -> pd.DataFrame:
    # The switching model was trained on ELA features plus predicted future metric values.
    features_with_lookahead = ela_features.copy()
    for target_col, lookahead_model in lookahead_models.items():
        features_with_lookahead[target_col] = lookahead_model.predict(ela_features)
    return features_with_lookahead

def make_default_performance_model():
    """
    Creates a default performance model using the default configuration of the PerformanceModel class as specified in ASF.

    Returns
    -------
    performance_model: PerformanceModel
        The created PerformanceModel.
    """
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

def make_default_wrapper_model(wrapper_type: str = "RandomForestClassifierWrapper"):
    """
    Creates a default wrapper model using the default configuration of the specified wrapper type.

    Parameters
    ----------
    wrapper_type: str, optional
        The type of wrapper model to create. Must be either "RandomForestClassifierWrapper" or "RandomForestRegressorWrapper".
    
    Returns
    -------
    wrapper_model: RandomForestClassifierWrapper or RandomForestRegressorWrapper
    """
    if wrapper_type == "RandomForestClassifierWrapper":
        default_classifier_config = RandomForestClassifierWrapper.get_configuration_space().get_default_configuration()
        default_classifier = RandomForestClassifierWrapper.get_from_configuration(default_classifier_config, random_state=42)()
        return default_classifier
    else:
        default_regressor_config = RandomForestRegressorWrapper.get_configuration_space().get_default_configuration()
        default_regressor = RandomForestRegressorWrapper.get_from_configuration(default_regressor_config, random_state=42)()
        return default_regressor

def normalise_selection_model_data(selection_model_data) -> tuple[pd.DataFrame, MinMaxScaler]:
    """
    Normalises the ELA features in the selection model data using min-max normalisation between 0 and 1 across all functions 
    for the ELA features, and between 1e-12 and 1 across all algorithms for each function.

    Parameters
    ----------
    selection_model_data: dict
        A dictionary where the keys are the switching budgets and the values are the corresponding selection model data as pandas DataFrames.

    Returns
    -------
    normalised_selection_model_data: dict
        A dictionary with the same structure as selection_model_data, but with the ELA features normalised.
    ela_scaler: MinMaxScaler
        The fitted MinMaxScaler used to normalise the ELA features, which can be used to normalise the ELA features of the test instances in the same way as the training data.
    """

    algo_cols = ALGORITHMS
    ela_feature_cols = selection_feature_cols(selection_model_data)
    
    normalised_selection_model_data = selection_model_data.copy()

    # Normalise algo cols
    for _, group in normalised_selection_model_data.groupby("fid"):
        idx = group.index
        algo_matrix = normalised_selection_model_data.loc[idx, algo_cols].to_numpy()
        flat_vals = algo_matrix.flatten().reshape(-1, 1)

        scaler = MinMaxScaler(feature_range=(1e-12, 1))
        flat_scaled = scaler.fit_transform(flat_vals).flatten()
        normalised_selection_model_data.loc[idx, algo_cols] = flat_scaled.reshape(algo_matrix.shape)

    # Normalise ELA feature cols

    ela_scaler = MinMaxScaler()
    normalised_selection_model_data[ela_feature_cols] = ela_scaler.fit_transform(normalised_selection_model_data[ela_feature_cols])

    return normalised_selection_model_data, ela_scaler

def create_selection_model_data(data_path: str, switching_budgets: list, store_data: bool = False):
    """
    Creates the data for training the selection model. It reads the ELA features and achieved metric values from data_collection.py,
    and matches the ELA features with the corresponding metric values according to the switching budget. We use the ELA features of 
    Non-elitist, as this is the A1 algorithms.

    Parameters
    ----------
    data_path: str
        Path to the parent folder containing the data collected in data_collection.py
        The folder structure must be the same as the one created by data_collection.py,
    normalise: bool, optional
        Whether to normalise the ELA features. Default is True.
    a2_algos: list, optional
        List of A2 algorithms to consider. Default is ["Elitist", "PSO", "DE", "BFGS", "MLSL", "Non-elitist"].
    store_data: bool, optional
        Whether to store the created selection model data as csv files. Default is False.

    Returns
    -------
    selection_model_data: dict
        A dictionary where the keys are the switching budgets and the values are the corresponding selection model data as pandas DataFrames.
    """
    selection_model_data = {}
    ela_scalers = {}

    # We load the ELA features based on which we switch. These are the features generated by the A1 algorithm, namely Non-elitist CMA-ES
    ela_features = drop_all_nan_ela_columns(pd.read_csv(ela_features_path(data_path, NO_SWITCH_ALGORITHM, TOTAL_BUDGET)))
    no_switch_metrics = pd.read_csv(achieved_metric_path(data_path, NO_SWITCH_ALGORITHM, TOTAL_BUDGET))


    for budget in switching_budgets:
        if budget == TOTAL_BUDGET: continue

        metrics = pd.concat(
            [
                *(pd.read_csv(achieved_metric_path(data_path, algo, budget)) for algo in SWITCH_ALGORITHMS),
                no_switch_metrics,
            ],
            ignore_index=True,
        )
    
        metrics_wide = (
            metrics
            .pivot(
                index=KEY_COLS,
                columns="algname",
                values=METRIC_COLUMN,
            )
            .rename_axis(columns=None)
            .reset_index()
        )


        ela_features_subset = ela_features[ela_features["ela_budget"] == budget].copy()

        # remove non-feature columns you do not want in final data
        drop_cols = ["a1_budget", "ela_budget", "a2_algorithm"]
        ela_features_subset = ela_features_subset.drop(
            columns=[c for c in drop_cols if c in ela_features_subset.columns]
        )

        selection_model_data_budget = ela_features_subset.merge(
            metrics_wide,
            on=KEY_COLS,
            how="inner"
        )

        # Drop all rows in which iid is not in the training set
        selection_model_data_budget = selection_model_data_budget[selection_model_data_budget["iid"].isin(TRAIN_IIDS)]

        # Drop all columns that are nan
        selection_model_data_budget = selection_model_data_budget.dropna(axis=1, how="all")

        selection_model_data[budget] = selection_model_data_budget

        selection_model_data[budget], ela_scalers[budget] = normalise_selection_model_data(selection_model_data_budget)

        if store_data:
            output_path = os.path.join(metric_scoped_path(data_path), SELECTION_TRAINING_DATA_FOLDER)
            os.makedirs(output_path, exist_ok=True)
            budget_output_path = os.path.join(output_path, f"selection_model_training_data_budget_{budget}.csv")
            selection_model_data[budget].to_csv(budget_output_path, index=False)
            print(f"Selection model training data for budget {budget} saved to {budget_output_path}")

    return selection_model_data, ela_scalers

def get_crossvalidated_predictions(selection_model_training_data: dict, store: bool = False, data_output_path: str = "./data", 
                                   no_switch_metric_path: str | None = None) -> pd.DataFrame:
    """
    Performs leave-one-instance-out cross-validation for each switching budget, and returns a DataFrame containing the predictions and actual metric values for each (fid, iid, rep) and switching budget.
    
    Parameters
    ----------
    selection_model_training_data : dict
        A dictionary containing the training data for each switching budget.
    safe : bool, optional
        Whether to save the predictions as a csv file, by default False.
    data_output_path : str, optional
        The path to the data directory in which to store the predictions, by default "./data".
    no_switch_metric_path : str, optional
        The path to the no-switch metric CSV file.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the predictions and actual metric values for each (fid, iid, rep) and switching budget.
    """
    if no_switch_metric_path is None:
        no_switch_metric_path = achieved_metric_path(data_output_path, NO_SWITCH_ALGORITHM, TOTAL_BUDGET)

    no_switch_metrics = pd.read_csv(no_switch_metric_path)

    # Only keep rows where iid is in the training set
    no_switch_metrics = no_switch_metrics[no_switch_metrics["iid"].isin(TRAIN_IIDS)]

    switching_budgets = sorted(selection_model_training_data.keys())

    algo_cols = ALGORITHMS

    prediction_rows = []

    instances = sorted(
        selection_model_training_data[switching_budgets[0]]["iid"].unique()
    )

    for budget in switching_budgets:
        print(f"Processing budget {budget}...")
        data_budget = selection_model_training_data[budget].copy()

        feature_cols = selection_feature_cols(data_budget)

        for instance in instances:
            print(f"Processing instance {instance}...")
            cv_training_data = data_budget[data_budget["iid"] != instance]
            cv_test_data = data_budget[data_budget["iid"] == instance]

            train_keys = list(
                cv_training_data[KEY_COLS]
                .itertuples(index=False, name=None)
            )

            test_keys = list(
                cv_test_data[KEY_COLS]
                .itertuples(index=False, name=None)
            )

            X_train = cv_training_data[feature_cols].copy()
            y_train = cv_training_data[algo_cols].copy()
            X_test = cv_test_data[feature_cols].copy()

            X_train.index = train_keys
            y_train.index = train_keys
            X_test.index = test_keys

            selector = make_default_performance_model()
            selector.algorithms = algo_cols
            selector.fit(X_train, y_train)

            predictions = selector.predict(X_test)


            for (fid, iid, rep), [(algo, _)] in predictions.items():
                actual_value = cv_test_data.loc[
                    (cv_test_data["fid"] == fid)
                    & (cv_test_data["iid"] == iid)
                    & (cv_test_data["rep"] == rep),
                    algo
                ].values[0]

                prediction_rows.append({
                    "fid": fid,
                    "iid": iid,
                    "rep": rep,
                    "budget": budget,
                    "selected_algorithm": algo,
                    ACTUAL_METRIC_COLUMN: actual_value,
                })

    # Add the no-switch metric values for each instance as well
    for _, row in no_switch_metrics.iterrows():
        prediction_rows.append({
            "fid": row["fid"],
            "iid": row["iid"],
            "rep": row["rep"],
            "budget": TOTAL_BUDGET,  # We can use total budget to indicate no-switch
            "selected_algorithm": NO_SWITCH_ALGORITHM,
            ACTUAL_METRIC_COLUMN: row[METRIC_COLUMN],
        })

    res = pd.DataFrame(prediction_rows)

    res = res.sort_values(by=KEY_COLS + ["budget"]).reset_index(drop=True)

    if store:
        output_dir = metric_scoped_path(data_output_path)
        output_path = os.path.join(output_dir, "crossvalidated_predictions.csv")

        os.makedirs(output_dir, exist_ok=True)

        res.to_csv(output_path, index=False)
        print(f"Cross-validated predictions saved to {output_path}")

    return res

def find_optimal_budgets_per_run(crossvalidated_predictions: pd.DataFrame, tie_breaking_strategy: str = "highest_budget", store: bool = False, data_output_path: str = "./data") -> dict:
    """
    For each (fid, iid, rep), we find the budget where the actual metric is lowest across all budgets for that (fid,iid,rep).
    If there are ties, we select the budget according to the specified tie-breaking strategy.

    Parameters
    ----------
    crossvalidated_predictions: pd.DataFrame
        A DataFrame containing the predictions and actual metric values for each (fid, iid, rep) and switching budget, as returned by get_crossvalidated_predictions.
    tie_breaking_strategy: str, optional
        The strategy to use for breaking ties when multiple budgets have the same lowest actual metric. 
        Must be one of "highest_budget" or "lowest_budget". Default is "highest_budget".
    store: bool, optional
        Whether to save the optimal budgets per run as a csv file, by default False.
    data_output_path: str, optional
        The path to the data directory in which to store the optimal budgets per run, by default "./data".
    """

    optimal_budgets = {}

    for (fid, iid, rep), group in crossvalidated_predictions.groupby(KEY_COLS):
        min_metric = group[ACTUAL_METRIC_COLUMN].min()
        best_budgets = group[group[ACTUAL_METRIC_COLUMN] == min_metric]["budget"].tolist()

        if tie_breaking_strategy == "highest_budget":
            selected_budget = max(best_budgets)
        elif tie_breaking_strategy == "lowest_budget":
            selected_budget = min(best_budgets)
        else:
            raise ValueError(f"Invalid tie-breaking strategy: {tie_breaking_strategy}")

        optimal_budgets[(fid, iid, rep)] = selected_budget

    if store:
        output_path = os.path.join(data_output_path, "optimal_budgets.csv")
        pd.DataFrame([
            {"fid": fid, "iid": iid, "rep": rep, "optimal_budget": budget}
            for (fid, iid, rep), budget in optimal_budgets.items()
        ]).to_csv(output_path, index=False)
        print(f"Optimal budgets saved to {output_path}")

    return optimal_budgets

def get_crossvalidated_lookahead_predictions(lookahead_model_training_data: pd.DataFrame) -> pd.DataFrame:
    """
    Performs leave-one-instance-out cross-validation for the lookahead model, and returns a DataFrame containing the predictions for each (fid, iid, rep) and switching budget.

    Parameters
    ----------
    lookahead_model_training_data : pd.DataFrame
        A DataFrame containing the training data for the lookahead model.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the predictions for each (fid, iid, rep) and switching budget.
    """
    feature_cols = lookahead_feature_cols(lookahead_model_training_data)
    
    prediction_rows = []

    instances = sorted(lookahead_model_training_data["iid"].unique())

    for instance in instances:
        print(f"Processing instance {instance}...")
        cv_training_data = lookahead_model_training_data[lookahead_model_training_data["iid"] != instance]
        cv_test_data = lookahead_model_training_data[lookahead_model_training_data["iid"] == instance]

        train_keys = list(cv_training_data[KEY_COLS].itertuples(index=False, name=None))
        test_keys = list(cv_test_data[KEY_COLS].itertuples(index=False, name=None))

        X_train = cv_training_data[feature_cols].copy()
        y_train = cv_training_data[lookahead_target_cols(cv_training_data)].copy()
        X_test = cv_test_data[feature_cols].copy()

        X_train.index = train_keys
        y_train.index = train_keys
        X_test.index = test_keys
        
        for t_col in y_train.columns:
            selector = make_default_wrapper_model(wrapper_type="RandomForestRegressorWrapper")
            selector.fit(X_train, y_train[t_col])

            predictions = selector.predict(X_test)

            for (fid, iid, rep), pred in zip(X_test.index, predictions):
                prediction_rows.append({
                    "fid": fid,
                    "iid": iid,
                    "rep": rep,
                    "t_col": t_col,
                    PREDICTED_METRIC_COLUMN: pred,
                })

    # Pivot the predictions to have one column per t_col
    res = (
    pd.DataFrame(prediction_rows)
    .pivot(
        index=KEY_COLS,
        columns="t_col",
        values=PREDICTED_METRIC_COLUMN,
    )
    .reset_index()
)

    res.columns.name = None

    # Sort t_0, t_1, ..., t_19 numerically
    t_cols = sorted([col for col in res.columns if col.startswith(LOOKAHEAD_TARGET_PREFIX)], key=lambda x: int(x.split("_")[1]))

    # Reorder the columns to have KEY_COLS first, then t_0, t_1, ..., t_19

    res = res[KEY_COLS + t_cols]

    res = res.sort_values(by=KEY_COLS).reset_index(drop=True)

    return res

def create_switch_model_data(selection_model_training_data: dict[int, pd.DataFrame], lookahead_model_training_data: dict[int, pd.DataFrame], tie_breaking_strategy: str = "highest_budget", store_final_data: bool = True, data_output_path: str = "./data", 
                             store_crossvalidated_predictions: bool = True, store_optimal_budgets: bool = True, no_switch_metric_path: str | None = None) -> pd.DataFrame:
    """
    Creates switching-model training data using leave-one-instance-out CV.
 
    Parameters
    ----------
    selection_model_training_data: dict[int, pd.DataFrame]
        A dictionary where the keys are the switching budgets and the values are the corresponding selection model training data as pandas DataFrames.
    lookahead_model_training_data: dict[int, pd.DataFrame]
        A dictionary where the keys are the switching budgets and the values are the corresponding lookahead model training data as pandas DataFrames.
    tie_breaking_strategy: str, optional
        The strategy to use for breaking ties when multiple budgets have the same lowest actual metric. 
        Must be one of "highest_budget" or "lowest_budget". Default is "highest_budget".
    store_final_data: bool, optional
        Whether to save the created switching model training data as csv files. Default is True.
    data_output_path: str, optional
        Path to the folder where the created switching model training data should be stored if store_final_data is True. Default is "./data".
    store_crossvalidated_predictions: bool, optional
        Whether to save the cross-validated predictions as csv files. Default is True.
    store_optimal_budgets: bool, optional
        Whether to save the optimal budgets per run as csv files. Default is True.
    
    Returns
    -------
    switching_model_training_data: dict[int, pd.DataFrame]
        A dictionary where the keys are the switching budgets and the values are the corresponding switching model training
    """
    switching_model_training_data = {}

    switching_budgets = sorted(selection_model_training_data.keys())

    crossvalidated_predictions_path = os.path.join(metric_scoped_path(data_output_path), "crossvalidated_predictions.csv")
    if os.path.exists(crossvalidated_predictions_path):
        print(f"Loading cross-validated predictions from {crossvalidated_predictions_path}...")
        crossvalidated_predictions = pd.read_csv(crossvalidated_predictions_path)
    else:
        crossvalidated_predictions = get_crossvalidated_predictions(
            selection_model_training_data,
            store=store_crossvalidated_predictions,
            data_output_path=data_output_path,
            no_switch_metric_path=no_switch_metric_path,
        )



    optimal_budgets_per_run = find_optimal_budgets_per_run(crossvalidated_predictions, tie_breaking_strategy=tie_breaking_strategy, store=store_optimal_budgets, data_output_path=data_output_path)

    for switching_budget in switching_budgets:
        switching_model_training_data[switching_budget] = selection_model_training_data[switching_budget].copy()

        # Remove algo columns
        algo_cols = ALGORITHMS
        switching_model_training_data[switching_budget] = switching_model_training_data[switching_budget].drop(columns=algo_cols)

        print(f"Getting cross-validated lookahead predictions for budget {switching_budget}...")
        lookahead_predictions = get_crossvalidated_lookahead_predictions(lookahead_model_training_data[switching_budget])

        
        switching_model_training_data[switching_budget] = switching_model_training_data[switching_budget].merge(
            lookahead_predictions,
            on=KEY_COLS,
            how="left"
        )

        # Add optimal budget column. Entry is true iff the optimal budget for that (fid, iid, rep) less or equal the current switching_budget
        switching_model_training_data[switching_budget]["optimal_budget"] = switching_model_training_data[switching_budget].apply(
            lambda row: optimal_budgets_per_run[(row["fid"], row["iid"], row["rep"])] <= switching_budget,
            axis=1
        )

    if store_final_data:   
        output_path = os.path.join(data_output_path, SWITCHING_TRAINING_DATA_FOLDER)
        os.makedirs(output_path, exist_ok=True)

        for budget, df in switching_model_training_data.items():
            budget_output_path = switching_training_data_path(data_output_path, budget)
            df.to_csv(budget_output_path, index=False)
            print(f"Switching model training data for budget {budget} saved to {budget_output_path}")

    return switching_model_training_data

def create_lookahead_model_data(selection_model_training_data: dict[int, pd.DataFrame], normalize_lookahead_performances: bool = True, store_final_data: bool = True, data_output_path: str = "./data") -> dict[int, pd.DataFrame]:
    """
    Creates lookahead-model training data using leave-one-instance-out CV.
 
    Parameters
    ----------
    selection_model_training_data: dict[int, pd.DataFrame]
        A dictionary where the keys are the switching budgets and the values are the corresponding selection model training data as pandas DataFrames.
    normalize_lookahead_performances: bool, optional
        Whether to normalize the lookahead performances. Default is True.
    store_final_data: bool, optional
        Whether to save the created lookahead model training data as csv files. Default is True.
    data_output_path: str, optional
        Path to the folder where the created lookahead model training data should be stored if store_final_data is True. Default is "./data".
    
    Returns
    -------
    lookahead_model_training_data: dict[int, pd.DataFrame]
        A dictionary where the keys are the switching budgets and the values are the corresponding lookahead model training data as pandas DataFrames.
    """
    lookahead_model_training_data = {}

    switching_budgets = SWITCHING_BUDGETS

    metrics_across_budget_and_algos = pd.concat(
        [
            pd.read_csv(achieved_metric_path(data_output_path, algo, budget))
            for algo in SWITCH_ALGORITHMS
            for budget in switching_budgets if budget != TOTAL_BUDGET
        ],
        ignore_index=True,
    )

    metrics_no_switch = pd.read_csv(achieved_metric_path(data_output_path, NO_SWITCH_ALGORITHM, TOTAL_BUDGET))

    metrics_across_budget_and_algos = pd.concat(
        [metrics_across_budget_and_algos, metrics_no_switch],
        ignore_index=True,
    )

    # For each (fid,iid,rep,budget), find the lowest achieved metric value.
    metrics_across_budget_and_algos = metrics_across_budget_and_algos.groupby(KEY_COLS + ["a1_budget"], as_index=False)[METRIC_COLUMN].min()

    # For each (fid,iid,rep), if the best budget is the total budget, copy the value to all other budgets.
    metrics_across_budget_and_algos = (
        metrics_across_budget_and_algos
        .groupby(KEY_COLS, group_keys=False)
        .apply(
            lambda group: (
                group.assign(**{METRIC_COLUMN: group[METRIC_COLUMN].min()})
                if group.loc[group[METRIC_COLUMN].idxmin(), "a1_budget"] == TOTAL_BUDGET
                else group
            )
        )
        .reset_index(drop=True)
    )

    # Only keep rows where iid is in the training set
    metrics_across_budget_and_algos = metrics_across_budget_and_algos[metrics_across_budget_and_algos["iid"].isin(TRAIN_IIDS)]

    # Now create the data for the lookahead models
    for budget in switching_budgets:
        if budget == TOTAL_BUDGET: continue
        selection_model_data_budget = selection_model_training_data[budget].copy()
        # Remove algo columns
        algo_cols = ALGORITHMS
        selection_model_data_budget = selection_model_data_budget.drop(columns=algo_cols)
        # Now attach columns t_0,...,t_((1000-budget)/50) with the achieved metric values for each budget
        lookahead_budgets = list(range(budget, TOTAL_BUDGET + 1, BUDGET_STEP))
        lookahead_performances = metrics_across_budget_and_algos[
            (metrics_across_budget_and_algos["a1_budget"].isin(lookahead_budgets))
        ]

        # Pivot the lookahead performances to have one column per budget
        lookahead_performances_pivoted = lookahead_performances.pivot(
            index=KEY_COLS,
            columns="a1_budget",
            values=METRIC_COLUMN
        )

        # Rename the columns to t_0,...,t_((1000-budget)/50)
        lookahead_performances_pivoted.columns = [f"t_{i}" for i in range(len(lookahead_performances_pivoted.columns))]

        # Merge the lookahead performances with the selection model data
        lookahead_model_data_budget = selection_model_data_budget.merge(   
            lookahead_performances_pivoted,
            on=KEY_COLS,
            how="left"
        )

        if normalize_lookahead_performances:
            # Just like the normalization of the selection model data, we normalize the lookahead performances per function, across all lookahead budgets
            for _, group in lookahead_model_data_budget.groupby("fid"):
                idx = group.index
                lookahead_matrix = lookahead_model_data_budget.loc[idx, [f"t_{i}" for i in range(len(lookahead_budgets))]].to_numpy()
                flat_vals = lookahead_matrix.flatten().reshape(-1, 1)

                scaler = MinMaxScaler(feature_range=(1e-12, 1))
                flat_scaled = scaler.fit_transform(flat_vals).flatten()
                lookahead_model_data_budget.loc[idx, [f"t_{i}" for i in range(len(lookahead_budgets))]] = flat_scaled.reshape(lookahead_matrix.shape)

        lookahead_model_training_data[budget] = lookahead_model_data_budget

        if store_final_data:
            output_path = os.path.join(metric_scoped_path(data_output_path), LOOKAHEAD_TRAINING_DATA_FOLDER)
            os.makedirs(output_path, exist_ok=True)
            budget_output_path = os.path.join(output_path, f"lookahead_model_training_data_budget_{budget}.csv")
            lookahead_model_training_data[budget].to_csv(budget_output_path, index=False)
            print(f"Lookahead model training data for budget {budget} saved to {budget_output_path}")

    return lookahead_model_training_data

class DynamicSelector:
    def __init__(self, switching_budgets: list = SWITCHING_BUDGETS, data_path: str = "./data", results_path: str = "./results", model_path: str = "./models", load_models: bool = False):
        """
        Initializes the DynamicSelector.

        Parameters
        ----------
        switching_budgets: list, optional
            A list of switching budgets to consider. Default is [50, 100, ..., 1000].
        data_path: str, optional
            Path to the parent folder containing the data collected in data_collection.py
            The folder structure must be the same as the one created by data_collection.py,
        model_path: str, optional
            Path to the parent folder containing the trained models.
            The folder structure must be the same as the one created by train_models
        load_models: bool, optional
            Whether to load the trained models from model_path. If False, the models will be initialized as None and need to be trained using train_models before evaluation. Default is False.
        """
        self.results_path = metric_scoped_path(results_path)
        self.switching_budgets = list(switching_budgets)
        self.raw_data_path = data_path
        self.data_path = metric_scoped_path(data_path)
        self.model_path = metric_scoped_path(model_path)
        
        if load_models:
            self.models = self.load_models_from_folder()
        else:
            self.models = {
                budget: {
                    "selection_model": None,
                    "switching_model": None,
                    "lookahead_models": None,
                    "ela_scaler": None,
                }
                for budget in switching_budgets
            }
          

    def load_models_from_folder(self) -> dict:
        models = {}
        for budget in self.switching_budgets:
            if budget == TOTAL_BUDGET: continue
            budget_path = os.path.join(self.model_path, f"budget_{budget}")
            selection_model_path = os.path.join(budget_path, "selection_model.joblib")
            switching_model_path = os.path.join(budget_path, "switching_model.joblib")
            lookahead_models_path = os.path.join(budget_path, "lookahead_models.joblib")
            ela_scaler_path = os.path.join(budget_path, f"ela_scaler.joblib")

            if not os.path.exists(selection_model_path) or not os.path.exists(switching_model_path) or not os.path.exists(lookahead_models_path):
                raise FileNotFoundError(f"Model files for budget {budget} not found in {budget_path}")

            selection_model = joblib.load(selection_model_path)
            switching_model = joblib.load(switching_model_path)
            lookahead_models = joblib.load(lookahead_models_path)
            ela_scaler = joblib.load(ela_scaler_path)

            models[budget] = {
                "selection_model": selection_model,
                "switching_model": switching_model,
                "lookahead_models": lookahead_models,
                "ela_scaler": ela_scaler,
            }

            print(f"Models for budget {budget} loaded successfully")

        return models

    def train_models(self, training_data_is_stored: bool = False, store_trained_models: bool = True):
        
        if not training_data_is_stored:
            selection_model_training_data, ela_scalers = create_selection_model_data(data_path=self.raw_data_path, switching_budgets=self.switching_budgets, store_data=True)
            lookahead_model_training_data = create_lookahead_model_data(selection_model_training_data, normalize_lookahead_performances=True, store_final_data=True, data_output_path=self.raw_data_path)
            switching_model_training_data = create_switch_model_data(
                selection_model_training_data,
                lookahead_model_training_data,
                tie_breaking_strategy="highest_budget",
                store_crossvalidated_predictions=True,
                store_optimal_budgets=True,
                store_final_data=True,
                data_output_path=self.data_path,
                no_switch_metric_path=achieved_metric_path(self.raw_data_path, NO_SWITCH_ALGORITHM, TOTAL_BUDGET),
            )
        else:
            selection_model_training_data = {}
            lookahead_model_training_data = {}
            switching_model_training_data = {}
            ela_scalers = {}

            missing_scalers = []
            for budget in self.switching_budgets:
                if budget == TOTAL_BUDGET:
                    continue
                scaler_path = os.path.join(self.model_path, f"budget_{budget}/ela_scaler.joblib")
                if not os.path.exists(scaler_path):
                    missing_scalers.append(budget)

            # It might happen that the selection training data is stored, but the scalers are not. In that case, we need to rebuild the selection training data and scalers.
            if missing_scalers:
                print(
                    "ELA scalers are missing for budgets "
                    f"{missing_scalers}. Rebuilding selection training data and scalers..."
                )
                _, ela_scalers = create_selection_model_data(
                    data_path=self.raw_data_path,
                    switching_budgets=self.switching_budgets,
                    store_data=False,
                )
            else:
                for budget in self.switching_budgets:
                    if budget == TOTAL_BUDGET: continue
                    ela_scalers[budget] = joblib.load(os.path.join(self.model_path, f"budget_{budget}/ela_scaler.joblib"))

            for budget in self.switching_budgets:
                if budget == TOTAL_BUDGET: continue
                selection_model_training_data[budget] = pd.read_csv(selection_training_data_path(self.data_path, budget))


            for budget in self.switching_budgets:
                if budget == TOTAL_BUDGET: continue
                lookahead_model_training_data[budget] = pd.read_csv(lookahead_training_data_path(self.data_path, budget))
                switching_model_training_data[budget] = pd.read_csv(switching_training_data_path(self.data_path, budget))


        for budget in self.switching_budgets:
            if budget == TOTAL_BUDGET: continue
            print(f"Training models for budget {budget}...")
            # 1. Selection model: choose the best algorithm if we switch at this budget.
            print(f"Training selection model for budget {budget}...")
            selection_model_train_data = selection_model_training_data[budget]
            X_train = selection_model_train_data[selection_feature_cols(selection_model_train_data)]
            y_train = selection_model_train_data[ALGORITHMS]

            selector = make_default_performance_model()
            selector.algorithms = ALGORITHMS
            selector.fit(X_train, y_train)

            self.models[budget]["selection_model"] = selector

            # 2. Lookahead models: predict future metric values t_0, t_1, ...
            # These predictions become features for the switching model.
            print(f"Training lookahead models for budget {budget}...")
            lookahead_models = train_lookahead_models(lookahead_model_training_data[budget])
            self.models[budget]["lookahead_models"] = lookahead_models

            # 3. Switching model: decide whether the optimal switch point has been reached.
            switching_model_train_data = switching_model_training_data[budget]
            X_train_switch = switching_model_train_data[switching_feature_cols(switching_model_train_data)]
            y_train_switch = switching_model_train_data["optimal_budget"]

            switching_model = make_default_wrapper_model(wrapper_type="RandomForestClassifierWrapper")
            switching_model.fit(X_train_switch, y_train_switch)
            self.models[budget]["switching_model"] = switching_model
            self.models[budget]["ela_scaler"] = ela_scalers[budget]

            if store_trained_models:
                model_budget_path = os.path.join(self.model_path, f"budget_{budget}")
                os.makedirs(model_budget_path, exist_ok=True)

                joblib.dump(selector, os.path.join(model_budget_path, "selection_model.joblib"))
                joblib.dump(lookahead_models, os.path.join(model_budget_path, "lookahead_models.joblib"))
                joblib.dump(switching_model, os.path.join(model_budget_path, "switching_model.joblib"))

                joblib.dump(self.models[budget]["ela_scaler"], os.path.join(model_budget_path, f"ela_scaler.joblib"))

    def simulate_single_run(self, fid: int, iid: int, rep: int, ela_rep: pd.DataFrame, metrics: pd.DataFrame) -> dict:
        """
        Simulates a single run of the dynamic selector on a given instance, and returns the results.

        Parameters
        ----------
        fid: int
            The function ID of the instance.
        iid: int
            The instance ID of the instance.
        rep: int
            The repetition number of the instance.
        ela_rep: pd.DataFrame
            A DataFrame containing the ELA features for this repetition.
        metrics: pd.DataFrame
            A DataFrame containing the achieved metric values for all algorithms on this instance for different budgets.

        Returns
        -------
        dict
            A dictionary containing the results of the simulation, including the selected algorithm, whether to switch or not, and the achieved metric value.
        """
        switch_decision = False
        switch_budget = None
        selected_algorithm = None
        achieved_metric = None

        for budget in self.switching_budgets:
            if budget == TOTAL_BUDGET: continue
            ela_row = ela_rep[ela_rep["ela_budget"] == budget].drop(columns=["ela_budget"])

            ela_row = drop_all_nan_ela_columns(ela_row)
            # Use the scaler fitted on the selection-model training data for this budget.
            ela_scaler = self.models[budget]["ela_scaler"]
            ela_features = pd.DataFrame(
                ela_scaler.transform(ela_row),
                columns=ela_row.columns,
                index=[(fid, iid, rep)]
            )

            # The switching model expects ELA features enriched with lookahead predictions.
            switching_model = self.models[budget]["switching_model"]
            lookahead_models = self.models[budget]["lookahead_models"]
            switching_features = add_lookahead_predictions(ela_features, lookahead_models)
           
            switch_decision = switching_model.predict(switching_features)[0]

            if switch_decision:
                # Once the switch model says "now", the selection model chooses the A2 algorithm.
                selection_model = self.models[budget]["selection_model"]
                algo_predicion = selection_model.predict(ela_features)
                predicted_algo = list(algo_predicion.values())[0][0][0]
                print(f"Switch decision: {switch_decision}, predicted algorithm: {predicted_algo}")
                achieved_metric = get_metric_value(metrics, fid, iid, rep, predicted_algo, budget)

                switch_budget = budget
                selected_algorithm = predicted_algo

                break

        if not switch_decision:
            predicted_algo = NO_SWITCH_ALGORITHM
            achieved_metric = get_metric_value(metrics, fid, iid, rep, predicted_algo, TOTAL_BUDGET)

            switch_budget = TOTAL_BUDGET
            selected_algorithm = predicted_algo
        
        vbs_metric = metrics[
            (metrics["fid"] == fid) &
            (metrics["iid"] == iid) &
            (metrics["rep"] == rep)
            ][METRIC_COLUMN].min()

        return{
            "fid": fid,
            "iid": iid,
            "rep": rep,
            VBS_METRIC_COLUMN: vbs_metric,
            "switch_budget": switch_budget,
            "selected_algorithm": selected_algorithm,
            METRIC_COLUMN: achieved_metric,
        }
        

    def evaluate(self) -> None:

        # 1. Check that models are loaded
        for budget in self.switching_budgets:
            if budget == TOTAL_BUDGET: continue
            if (
                self.models[budget]["selection_model"] is None
                or self.models[budget]["switching_model"] is None
                or self.models[budget]["lookahead_models"] is None
                or self.models[budget]["ela_scaler"] is None
            ):
                raise ValueError(f"Models for budget {budget} are not loaded. Models must be trained and loaded before evaluation.")

        # 2. Load data
        metrics = pd.concat(
            [
                pd.read_csv(achieved_metric_path(self.raw_data_path, algo, budget))
                for algo in SWITCH_ALGORITHMS
                for budget in self.switching_budgets if budget != TOTAL_BUDGET
            ],
            ignore_index=True,
        )
        metrics_no_switch = pd.read_csv(achieved_metric_path(self.raw_data_path, NO_SWITCH_ALGORITHM, TOTAL_BUDGET))

        metrics = pd.concat(
            [metrics, metrics_no_switch],
            ignore_index=True,
        )

        ela_features = pd.read_csv(ela_features_path(self.raw_data_path, NO_SWITCH_ALGORITHM, TOTAL_BUDGET))
        # Only keep rows where iid is in the test set
        ela_features = ela_features[ela_features["iid"].isin(TEST_IIDS)]
        for fid in range(1, 25):
            for iid in TEST_IIDS:
                for rep in range(0, 20):
                    print(f"Evaluating on fid {fid}, iid {iid}, rep {rep}...")
                    # Get the ELA features for this (fid, iid, rep)
                    ela_features_rep = ela_features[
                        (ela_features["fid"] == fid) &
                        (ela_features["iid"] == iid) &
                        (ela_features["rep"] == rep)
                    ].drop(columns=["a1_budget", "a2_algorithm"] + META_COLS)

                    result = self.simulate_single_run(fid, iid, rep, ela_features_rep, metrics)

                    # Collect predictions of static selection models
                    for budget in self.switching_budgets:
                        if budget == TOTAL_BUDGET: continue
                        selector = self.models[budget]["selection_model"]
                        ela_row = ela_features_rep[ela_features_rep["ela_budget"] == budget].drop(columns=["ela_budget"])

                        ela_row = drop_all_nan_ela_columns(ela_row)

                        ela_features_scaled = self.models[budget]["ela_scaler"].transform(ela_row)
                        ela_features_scaled = pd.DataFrame(
                            ela_features_scaled,
                            columns=ela_row.columns,
                            index=[(fid, iid, rep)]
                        )
                        algo_prediction = selector.predict(ela_features_scaled)
                        predicted_algo = list(algo_prediction.values())[0][0][0]
                        algo_metric = get_metric_value(metrics, fid, iid, rep, predicted_algo, budget)

                        result[f"static_B{budget}"] = algo_metric
                    
                    result["no_switch"] = get_metric_value(metrics, fid, iid, rep, NO_SWITCH_ALGORITHM, TOTAL_BUDGET)

                    result_df = pd.DataFrame([result])
                    output_path = os.path.join(self.results_path, "selector_results.csv")
                    os.makedirs(self.results_path, exist_ok=True)
                    result_df.to_csv(output_path, mode="a", header=not os.path.exists(output_path), index=False)

def build_switching_training_data_from_stored_tables(data_path: str) -> None:
    raw_data_path = data_path
    data_path = metric_scoped_path(data_path)
    selection_model_training_data = {}
    lookahead_model_training_data = {}

    missing_selection_files = []
    missing_lookahead_files = []

    for budget in TRAINING_SWITCHING_BUDGETS:
        selection_path = selection_training_data_path(data_path, budget)
        lookahead_path = lookahead_training_data_path(data_path, budget)

        if not os.path.exists(selection_path):
            missing_selection_files.append(selection_path)
        if not os.path.exists(lookahead_path):
            missing_lookahead_files.append(lookahead_path)

    if missing_selection_files:
        print("Selection training data missing for one or more budgets. Rebuilding selection tables...")
        selection_model_training_data, _ = create_selection_model_data(
            data_path=raw_data_path,
            switching_budgets=SWITCHING_BUDGETS,
            store_data=True,
        )
    else:
        for budget in TRAINING_SWITCHING_BUDGETS:
            selection_model_training_data[budget] = pd.read_csv(selection_training_data_path(data_path, budget))

    if missing_lookahead_files:
        print("Lookahead training data missing for one or more budgets. Rebuilding lookahead tables...")
        lookahead_model_training_data = create_lookahead_model_data(
            selection_model_training_data,
            normalize_lookahead_performances=True,
            store_final_data=True,
            data_output_path=raw_data_path,
        )
    else:
        for budget in TRAINING_SWITCHING_BUDGETS:
            lookahead_model_training_data[budget] = pd.read_csv(lookahead_training_data_path(data_path, budget))

    create_switch_model_data(
        selection_model_training_data,
        lookahead_model_training_data,
        data_output_path=data_path,
        no_switch_metric_path=achieved_metric_path(raw_data_path, NO_SWITCH_ALGORITHM, TOTAL_BUDGET),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train, evaluate, or prepare data for the dynamic selector.")
    parser.add_argument(
        "--mode",
        choices=["build-switch-data", "train", "evaluate", "train-evaluate"],
        default="build-switch-data",
        help="Workflow to run. The default preserves the historical script behavior.",
    )
    parser.add_argument("--data-path", default="./data", help=f"Base data directory. Selector-generated tables use <data-path>/{METRIC}; raw data_collection inputs are read from either <data-path> or <data-path>/{METRIC}.")
    parser.add_argument("--results-path", default="./results", help=f"Base results directory. Outputs are written under <results-path>/{METRIC}.")
    parser.add_argument("--model-path", default="./models", help=f"Base model directory. Models are saved/loaded under <model-path>/{METRIC}.")
    parser.add_argument(
        "--training-data-is-stored",
        action="store_true",
        help="Load prepared training tables instead of recreating them during training.",
    )
    parser.add_argument(
        "--no-store-trained-models",
        action="store_true",
        help="Train models without writing joblib files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.mode == "build-switch-data":
        build_switching_training_data_from_stored_tables(args.data_path)
        return

    if args.mode == "evaluate":
        selector = DynamicSelector(
            data_path=args.data_path,
            results_path=args.results_path,
            model_path=args.model_path,
            load_models=True,
        )
        selector.evaluate()
        return

    selector = DynamicSelector(
        data_path=args.data_path,
        results_path=args.results_path,
        model_path=args.model_path,
        load_models=False,
    )
    selector.train_models(
        training_data_is_stored=args.training_data_is_stored,
        store_trained_models=not args.no_store_trained_models,
    )

    if args.mode == "train-evaluate":
        selector.evaluate()


if __name__ == "__main__":
    main()
