"""
This file contains the code for the dynamic selector.
"""

import joblib
import pandas as pd
from pathlib import Path
import os
import numpy as np
import warnings
import sys

from asf.predictors import RandomForestClassifierWrapper
from asf.selectors import PerformanceModel
from ConfigSpace import ConfigurationSpace

from sklearn.preprocessing import MinMaxScaler

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

def make_default_switching_model():
    """
    Creates a default switching model using the default configuration of the RandomForestClassifierWrapper class.

    Returns
    -------
    switching_model: RandomForestClassifierWrapper
        The created RandomForestClassifierWrapper.
    """
    default_classifier_config = RandomForestClassifierWrapper.get_configuration_space().get_default_configuration()
    default_classifier = RandomForestClassifierWrapper.get_from_configuration(default_classifier_config, random_state=42)()
    return default_classifier

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

    algo_cols = ["Non-elitist", "Elitist", "PSO", "DE", "BFGS", "MLSL"]
    ela_feature_cols = [col for col in selection_model_data.columns if col not in algo_cols + ['fid', 'iid', 'rep','high_level_category']]
    
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

def create_selection_model_data(data_path: str, switching_budgets: list, a2_algos: list = ["Elitist", "PSO", "DE", "BFGS", "MLSL", "Non-elitist"], store_data: bool = False):
    """
    Creates the data for training the selection model. It reads the ELA features and achieved regrets from data_collection.py,
    and matches the ELA features with the corresponding regrets according to the switching budget. We use the ELA features of 
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
    ela_features = pd.read_csv(os.path.join(data_path, "ela_features/Non-elitist_B1000_5D/ELA_features.csv"))
    non_elitist_regrets = pd.read_csv(os.path.join(data_path, "achieved_regrets/achieved_regrets_Non-elitist_B1000_5D.csv"))

    # Remove non-elitist from a2_algos if it is there, as we will add the regrets for non-elitist separately to ensure they are included in the data
    a2_algos = [algo for algo in a2_algos if algo != "Non-elitist"]

    for budget in switching_budgets:
        # regret_dfs = []
        if budget == 1000: continue

        regrets = pd.concat(
            [
                *(pd.read_csv(os.path.join(
                    data_path,
                    f"achieved_regrets/achieved_regrets_{algo}_B{budget}_5D.csv"
                )) for algo in a2_algos),
                non_elitist_regrets,
            ],
            ignore_index=True,
        )
    
        regrets_wide = (
            regrets
            .pivot(
                index=["fid", "iid", "rep"],
                columns="algname",
                values="achieved_regret",
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
            regrets_wide,
            on=["fid", "iid", "rep"],
            how="inner"
        )

        # Drop all rows in which iid is not in the training set (1,2,3,4,5)
        selection_model_data_budget = selection_model_data_budget[selection_model_data_budget["iid"].isin([1, 2, 3, 4, 5])]

        selection_model_data[budget] = selection_model_data_budget

        selection_model_data[budget], ela_scalers[budget] = normalise_selection_model_data(selection_model_data_budget)

        if store_data:
            output_path = os.path.join(data_path, "selection_model_training_data")
            os.makedirs(output_path, exist_ok=True)
            budget_output_path = os.path.join(output_path, f"selection_model_training_data_budget_{budget}.csv")
            selection_model_data[budget].to_csv(budget_output_path, index=False)
            print(f"Selection model training data for budget {budget} saved to {budget_output_path}")

    return selection_model_data, ela_scalers

def get_crossvalidated_predictions(selection_model_training_data: dict, store: bool = False, data_output_path: str = "./data", 
                                   no_switch_regrets_path: str = "./data/achieved_regrets/achieved_regrets_Non-elitist_B1000_5D.csv") -> pd.DataFrame:
    """
    Performs leave-one-instance-out cross-validation for each switching budget, and returns a DataFrame containing the predictions and actual regrets for each (fid, iid, rep) and switching budget.
    
    Parameters
    ----------
    selection_model_training_data : dict
        A dictionary containing the training data for each switching budget.
    safe : bool, optional
        Whether to save the predictions as a csv file, by default False.
    data_output_path : str, optional
        The path to the data directory in which to store the predictions, by default "./data".
    no_switch_regrets_path : str, optional
        The path to the no-switch regrets CSV file, by default "./data/achieved_regrets/achieved_regrets_Non-elitist_B1000_5D.csv".

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the predictions and actual regrets for each (fid, iid, rep) and switching budget.
    """
    no_switch_regrets = pd.read_csv(no_switch_regrets_path)

    # Only keep rows where iid is in the training set (1,2,3,4,5)
    no_switch_regrets = no_switch_regrets[no_switch_regrets["iid"].isin([1, 2, 3, 4, 5])]

    switching_budgets = sorted(selection_model_training_data.keys())

    algo_cols = ["Non-elitist", "Elitist", "PSO", "DE", "BFGS", "MLSL"]

    prediction_rows = []

    instances = sorted(
        selection_model_training_data[switching_budgets[0]]["iid"].unique()
    )

    for budget in switching_budgets:
        print(f"Processing budget {budget}...")
        data_budget = selection_model_training_data[budget].copy()

        feature_cols = [
            col for col in data_budget.columns
            if col not in ["fid", "iid", "rep", "high_level_category"] + algo_cols
        ]

        for instance in instances:
            print(f"Processing instance {instance}...")
            cv_training_data = data_budget[data_budget["iid"] != instance]
            cv_test_data = data_budget[data_budget["iid"] == instance]

            train_keys = list(
                cv_training_data[["fid", "iid", "rep"]]
                .itertuples(index=False, name=None)
            )

            test_keys = list(
                cv_test_data[["fid", "iid", "rep"]]
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
                    "actual_regret": actual_value,
                })

    # Add the no-switch regrets for each instance as well
    for _, row in no_switch_regrets.iterrows():
        prediction_rows.append({
            "fid": row["fid"],
            "iid": row["iid"],
            "rep": row["rep"],
            "budget": 1000,  # We can use budget 1000 to indicate no-switch, as 1000 is the maximum budget
            "selected_algorithm": "Non-elitist",
            "actual_regret": row["achieved_regret"],
        })

    res = pd.DataFrame(prediction_rows)

    res = res.sort_values(by=["fid", "iid", "rep", "budget"]).reset_index(drop=True)

    if store:
        output_path = os.path.join(data_output_path, "crossvalidated_predictions.csv")

        if not os.path.exists(data_output_path):
            os.makedirs(data_output_path, exist_ok=True)

        res.to_csv(output_path, index=False)
        print(f"Cross-validated predictions saved to {output_path}")

    return res

def find_optimal_budgets_per_run(crossvalidated_predictions: pd.DataFrame, tie_breaking_strategy: str = "highest_budget", store: bool = False, data_output_path: str = "./data") -> dict:
    """
    For each (fid, iid, rep), we find the budget where actual_regret is the lowest across all budgets for that (fid,iid,rep).
    If there are ties, we select the budget according to the specified tie-breaking strategy.

    Parameters
    ----------
    crossvalidated_predictions: pd.DataFrame
        A DataFrame containing the predictions and actual regrets for each (fid, iid, rep) and switching budget, as returned by get_crossvalidated_predictions.
    tie_breaking_strategy: str, optional
        The strategy to use for breaking ties when multiple budgets have the same lowest actual_regret. 
        Must be one of "highest_budget" or "lowest_budget". Default is "highest_budget".
    store: bool, optional
        Whether to save the optimal budgets per run as a csv file, by default False.
    data_output_path: str, optional
        The path to the data directory in which to store the optimal budgets per run, by default "./data".
    """

    optimal_budgets = {}

    for (fid, iid, rep), group in crossvalidated_predictions.groupby(["fid", "iid", "rep"]):
        min_regret = group["actual_regret"].min()
        best_budgets = group[group["actual_regret"] == min_regret]["budget"].tolist()

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


def create_switch_model_data(selection_model_training_data: dict[int, pd.DataFrame], store_final_data: bool = True, data_output_path: str = "./data", 
                             store_crossvalidated_predictions: bool = False, store_optimal_budgets: bool = False) -> pd.DataFrame:
    """
    Creates switching-model training data using leave-one-instance-out CV.
 
    Parameters
    ----------
    selection_model_training_data: dict[int, pd.DataFrame]
        A dictionary where the keys are the switching budgets and the values are the corresponding selection model training data as pandas DataFrames.
    store_final_data: bool, optional
        Whether to save the created switching model training data as csv files. Default is True.
    data_output_path: str, optional
        Path to the folder where the created switching model training data should be stored if store_final_data is True. Default is "./data".
    store_crossvalidated_predictions: bool, optional
        Whether to save the cross-validated predictions as csv files. Default is False.
    store_optimal_budgets: bool, optional
        Whether to save the optimal budgets per run as csv files. Default is False.
    
    Returns
    -------
    switching_model_training_data: dict[int, pd.DataFrame]
        A dictionary where the keys are the switching budgets and the values are the corresponding switching model training
    """
    switching_model_training_data = {}

    switching_budgets = sorted(selection_model_training_data.keys())
    
    crossvalidated_predictions = get_crossvalidated_predictions(selection_model_training_data, store=store_crossvalidated_predictions, data_output_path=data_output_path)

    optimal_budgets_per_run = find_optimal_budgets_per_run(crossvalidated_predictions, store=store_optimal_budgets, data_output_path=data_output_path)

    for switching_budget in switching_budgets:
        switching_model_training_data[switching_budget] = selection_model_training_data[switching_budget].copy()

        # Remove algo columns
        
        algo_cols = ["Non-elitist", "Elitist", "PSO", "DE", "BFGS", "MLSL"]
        switching_model_training_data[switching_budget] = switching_model_training_data[switching_budget].drop(columns=algo_cols)

        # Add optimal budget column. Entry is true iff the optimal budget for that (fid, iid, rep) less or equal the current switching_budget
        switching_model_training_data[switching_budget]["optimal_budget"] = switching_model_training_data[switching_budget].apply(
            lambda row: optimal_budgets_per_run[(row["fid"], row["iid"], row["rep"])] <= switching_budget,
            axis=1
        )

    if store_final_data:   
        output_path = os.path.join(data_output_path, "switching_model_training_data")
        os.makedirs(output_path, exist_ok=True)

        for budget, df in switching_model_training_data.items():
            budget_output_path = os.path.join(output_path, f"switching_model_training_data_budget_{budget}.csv")
            df.to_csv(budget_output_path, index=False)
            print(f"Switching model training data for budget {budget} saved to {budget_output_path}")

    return switching_model_training_data

class DynamicSelector:
    def __init__(self, switching_budgets: list = [50*i for i in range(1, 21)], data_path: str = "./data", results_path: str = "./results", model_path: str = "./models", load_models: bool = False):
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
        self.results_path = results_path
        self.switching_budgets = switching_budgets
        self.data_path = data_path
        self.model_path = model_path
        
        if load_models:
            self.models = self.load_models_from_folder()
        else:
            self.models = {budget: {"selection_model": None, "switching_model": None, "ela_scaler": None} for budget in switching_budgets}
          

    def load_models_from_folder(self) -> dict:
        models = {}
        for budget in self.switching_budgets:
            if budget == 1000: continue
            budget_path = os.path.join(self.model_path, f"budget_{budget}")
            selection_model_path = os.path.join(budget_path, "selection_model.joblib")
            switching_model_path = os.path.join(budget_path, "switching_model.joblib")
            ela_scaler_path = os.path.join(budget_path, f"ela_scaler.joblib")

            if not os.path.exists(selection_model_path) or not os.path.exists(switching_model_path):
                raise FileNotFoundError(f"Model files for budget {budget} not found in {budget_path}")

            selection_model = joblib.load(selection_model_path)
            switching_model = joblib.load(switching_model_path)
            ela_scaler = joblib.load(ela_scaler_path)

            models[budget] = {
                "selection_model": selection_model,
                "switching_model": switching_model,
                "ela_scaler": ela_scaler,
            }

            print(f"Models for budget {budget} loaded successfully")

        return models

    def train_models(self, training_data_is_stored: bool = False, store_trained_models: bool = True):
        
        if not training_data_is_stored:
            selection_model_training_data, ela_scalers = create_selection_model_data(data_path=self.data_path, switching_budgets=self.switching_budgets, store_data=True)
            switching_model_training_data = create_switch_model_data(selection_model_training_data, store_crossvalidated_predictions=True, store_optimal_budgets=True, store_final_data=True, data_output_path=self.data_path)
        else:
            selection_model_training_data = {}
            switching_model_training_data = {}
            ela_scalers = {}

            for budget in self.switching_budgets:
                if budget == 1000: continue
                selection_model_training_data[budget] = pd.read_csv(os.path.join(self.data_path, f"selection_model_training_data/selection_model_training_data_budget_{budget}.csv"))
                switching_model_training_data[budget] = pd.read_csv(os.path.join(self.data_path, f"switching_model_training_data/switching_model_training_data_budget_{budget}.csv"))
                ela_scalers[budget] = joblib.load(os.path.join(self.model_path, f"budget_{budget}/ela_scaler.joblib"))


        for budget in self.switching_budgets:
            if budget == 1000: continue
            print(f"Training models for budget {budget}...")
            #1. Train the selection model for this budget 
            selection_model_train_data = selection_model_training_data[budget]
            X_train = selection_model_train_data.drop(columns=["fid", "iid", "rep", "high_level_category"] + ["Non-elitist", "Elitist", "PSO", "DE", "BFGS", "MLSL"])
            y_train = selection_model_train_data[["Non-elitist", "Elitist", "PSO", "DE", "BFGS", "MLSL"]]

            selector = make_default_performance_model()
            selector.algorithms = ["Non-elitist", "Elitist", "PSO", "DE", "BFGS", "MLSL"]
            selector.fit(X_train, y_train)

            self.models[budget]["selection_model"] = selector

            #2. Train the switching model for this budget
            switching_model_train_data = switching_model_training_data[budget]
            X_train_switch = switching_model_train_data.drop(columns=["fid", "iid", "rep", "high_level_category", "optimal_budget"])
            y_train_switch = switching_model_train_data["optimal_budget"]

            switching_model = make_default_switching_model()
            switching_model.fit(X_train_switch, y_train_switch)
            self.models[budget]["switching_model"] = switching_model
            self.models[budget]["ela_scaler"] = ela_scalers[budget]

            if store_trained_models:
                model_budget_path = os.path.join(self.model_path, f"budget_{budget}")
                os.makedirs(model_budget_path, exist_ok=True)

                joblib.dump(selector, os.path.join(model_budget_path, "selection_model.joblib"))
                joblib.dump(switching_model, os.path.join(model_budget_path, "switching_model.joblib"))

                joblib.dump(self.models[budget]["ela_scaler"], os.path.join(model_budget_path, f"ela_scaler.joblib"))

    def simulate_single_run(self, fid: int, iid: int, rep: int, ela_features_instance: pd.DataFrame, regrets: pd.DataFrame) -> dict:
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
        ela_features_instance: pd.DataFrame
            A DataFrame containing the ELA features for this instance.
        regrets: pd.DataFrame
            A DataFrame containing the achieved regrets for all algorithms on this instance for different budgets.

        Returns
        -------
        dict
            A dictionary containing the results of the simulation, including the selected algorithm, whether to switch or not, and the achieved regret.
        """
        switch_decision = False
        switch_budget = None
        selected_algorithm = None
        achieved_regret = None

        for budget in self.switching_budgets:
            if budget == 1000: continue
            ela_row = ela_features_instance[ela_features_instance["ela_budget"] == budget].drop(columns=["ela_budget"])

            # Get scaler
            ela_scaler = self.models[budget]["ela_scaler"]
            ela_features = pd.DataFrame(
                ela_scaler.transform(ela_row),
                columns=ela_row.columns,
                index=[(fid, iid, rep)]
            )
            # print(f"Scaled ELA features for budget {budget}:\n{ela_features}")

            # Get switching model
            switching_model = self.models[budget]["switching_model"]
            switch_decision = switching_model.predict(ela_features)[0]

            if switch_decision:
                selection_model = self.models[budget]["selection_model"]
                algo_predicion = selection_model.predict(ela_features)
                predicted_algo = list(algo_predicion.values())[0][0][0]
                print(f"Switch decision: {switch_decision}, predicted algorithm: {predicted_algo}")
                achieved_regret = regrets[
                    (regrets["fid"] == fid) &
                    (regrets["iid"] == iid) &
                    (regrets["rep"] == rep) &
                    (regrets["a1_budget"] == budget if predicted_algo != "Non-elitist" else regrets["a1_budget"] == 1000) &
                    (regrets["algname"] == predicted_algo)
                ]["achieved_regret"].values[0]

                switch_budget = budget
                selected_algorithm = predicted_algo

                break

        if not switch_decision:
            predicted_algo = "Non-elitist"
            achieved_regret = regrets[
                (regrets["fid"] == fid) &
                (regrets["iid"] == iid) &
                (regrets["rep"] == rep) &
                (regrets["a1_budget"] == 1000) &
                (regrets["algname"] == "Non-elitist")
            ]["achieved_regret"].values[0]

            switch_budget = 1000
            selected_algorithm = predicted_algo
        
        vbs_precision = regrets[
            (regrets["fid"] == fid) &
            (regrets["iid"] == iid) &
            (regrets["rep"] == rep)
            ]["achieved_regret"].min()

        return{
            "fid": fid,
            "iid": iid,
            "rep": rep,
            "vbs_precision": vbs_precision,
            "switch_budget": switch_budget,
            "selected_algorithm": selected_algorithm,
            "achieved_regret": achieved_regret,
        }
        

    def evaluate(self) -> np.ndarray:

        # 1. Check that models are loaded
        for budget in self.switching_budgets:
            if budget == 1000: continue
            if self.models[budget]["selection_model"] is None or self.models[budget]["switching_model"] is None or self.models[budget]["ela_scaler"] is None:
                raise ValueError(f"Models for budget {budget} are not loaded. Models must be trained and loaded before evaluation.")

        # 2. Load data
        regrets = pd.concat(
            [
                pd.read_csv(os.path.join(self.data_path, f"achieved_regrets/achieved_regrets_{algo}_B{budget}_5D.csv"))
                for algo in ["Elitist", "PSO", "DE", "BFGS", "MLSL"]
                for budget in self.switching_budgets if budget != 1000
            ],
            ignore_index=True,
        )
        regrets_no_switch = pd.read_csv(os.path.join(self.data_path, "achieved_regrets/achieved_regrets_Non-elitist_B1000_5D.csv"))

        regrets = pd.concat(
            [regrets, regrets_no_switch],
            ignore_index=True,
        )

        ela_features = pd.read_csv(os.path.join(self.data_path, "ela_features/Non-elitist_B1000_5D/ELA_features.csv"))
        # Only keep rows where iid is in the test set (6,7)
        ela_features = ela_features[ela_features["iid"].isin([6, 7])]
        for fid in range(1, 25):
            for iid in [6, 7]:
                for rep in range(0, 20):
                    print(f"Evaluating on fid {fid}, iid {iid}, rep {rep}...")
                    # Get the ELA features for this (fid, iid, rep)
                    ela_features_instance = ela_features[
                        (ela_features["fid"] == fid) &
                        (ela_features["iid"] == iid) &
                        (ela_features["rep"] == rep)
                    ].drop(columns=["a1_budget", "a2_algorithm", "fid", "iid", "rep", "high_level_category"])

                    result = self.simulate_single_run(fid, iid, rep, ela_features_instance, regrets)

                    # Collect predictions of static selection models
                    for budget in self.switching_budgets:
                        if budget == 1000: continue
                        selector = self.models[budget]["selection_model"]
                        ela_row = ela_features_instance[ela_features_instance["ela_budget"] == budget].drop(columns=["ela_budget"])
                        ela_features_scaled = self.models[budget]["ela_scaler"].transform(ela_row)
                        ela_features_scaled = pd.DataFrame(
                            ela_features_scaled,
                            columns=ela_row.columns,
                            index=[(fid, iid, rep)]
                        )
                        algo_prediction = selector.predict(ela_features_scaled)
                        predicted_algo = list(algo_prediction.values())[0][0][0]

                        algo_regret = regrets[
                            (regrets["fid"] == fid) &
                            (regrets["iid"] == iid) &
                            (regrets["rep"] == rep) &
                            (regrets["a1_budget"] == budget if predicted_algo != "Non-elitist" else regrets["a1_budget"] == 1000) &
                            (regrets["algname"] == predicted_algo)
                        ]["achieved_regret"].values[0]

                        result[f"static_B{budget}"] = algo_regret
                    
                    result["no_switch"] = regrets[
                        (regrets["fid"] == fid) &
                        (regrets["iid"] == iid) &
                        (regrets["rep"] == rep) &
                        (regrets["a1_budget"] == 1000) &
                        (regrets["algname"] == "Non-elitist")
                    ]["achieved_regret"].values[0]

                    result_df = pd.DataFrame([result])
                    output_path = os.path.join(self.results_path, "selector_results.csv")
                    os.makedirs(self.results_path, exist_ok=True)
                    result_df.to_csv(output_path, mode="a", header=not os.path.exists(output_path), index=False)

if __name__ == "__main__":    
    # # Example usage
    selector = DynamicSelector(data_path="./data", results_path="./results", load_models=True)
    # selector.train_models(training_data_is_stored=False, store_trained_models=True)
    results = selector.evaluate()
    # for budget in [50*i for i in range(1, 20)]:
    #     scaler = joblib.load(f"./models/budget_{budget}/ela_scaler.joblib")
    #     print(budget, scaler)