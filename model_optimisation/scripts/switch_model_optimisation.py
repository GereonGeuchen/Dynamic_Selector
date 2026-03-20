import numpy as np
import pandas as pd
from pathlib import Path
import os
import joblib
from functools import partial
from itertools import combinations
from multiprocessing import Pool
import sys

# ========== ConfigSpace and SMAC imports ==========
from ConfigSpace import ConfigurationSpace
from smac import HyperparameterOptimizationFacade, Scenario

# === Import your RandomForestClassifierWrapper ===
from asf.predictors import RandomForestClassifierWrapper

# === Your switching budgets ===
SWITCHING_BUDGETS = [50*i for i in range(1, 20)]

# === Your instance IDs for evaluation ===
FIDS = list(range(1, 25))
IIDS = [1, 2, 3, 4, 5] 
REPS = list(range(20))



# === Paths ===

NUM_LOOKAHEAD_EPMS = 0  # This will be set from command line argument
USE_ALGO_FEATURES = False  # Whether to use the ELA features for the performance models or not, will be set from command line argument
ELA_DIR_SWITCH = "../data/A1_data_algo_features_switch_with_lookahead_predictions_all_epms_afterwards_normalized"
ELA_DIR_ALGO = "../data/A1_data_ela_normalized_with_precisions"
PRECISION_FILE = "../data/A2_precisions_normalized_log10.csv"
CV_MODELS_DIR = "../data/models/trained_models/algo_performance_models_cv_algo_features"
UNTRAINED_PERF_MODELS_DIR = "../data/models/untrained_models/algo_performance_models_algo_features"
SMAC_OUTPUT_DIR = f"smac_lookaheads/smac_output_switch_lookahead_{NUM_LOOKAHEAD_EPMS}_normalized_afterwards"
OUTPUT_PATH = f"../data/models/tuned_models/switching_models_lookahead_{NUM_LOOKAHEAD_EPMS}_normalized_afterwards"


# ========== Helper classes ==========
def prepare_switch_data(df_switch):
    # This function either drops or keeps algo features, and keeps the right amount of lookahead predictions based on the global config variables
    if USE_ALGO_FEATURES:
        df_switch = df_switch.drop(columns=["pred_t0"], errors="ignore")
    else:
        df_switch = df_switch.drop(columns=["BFGS", "DE", "PSO", "MLSL", "Non-elitist", "Elitist"], errors="ignore")
    
    if NUM_LOOKAHEAD_EPMS is not None:
        df_switch = df_switch.drop(columns=[f"pred_t{t}" for t in range(NUM_LOOKAHEAD_EPMS + 1, 20)], errors="ignore")
    else:
        df_switch = df_switch.drop(columns=[f"pred_t{t}" for t in range(1, 20)], errors="ignore")

    return df_switch

class SwitchingSelectorCV:
    def __init__(self, precision_file):
        self.precision_df = pd.read_csv(precision_file)

    def simulate_single_run(self, fid, iid, rep, switching_models, performance_models):

        for budget in SWITCHING_BUDGETS:

            switch_model = switching_models.get(budget)
            perf_model = performance_models.get(budget)
            if switch_model is None or perf_model is None:
                if switch_model is None:
                    print(f"No switching model for budget {budget}")
                if perf_model is None:
                    print(f"No performance model for budget {budget}")
                #print("No model available for this budget, skipping.")
                continue

            ela_path_algo = Path(ELA_DIR_ALGO) / f"A1_B{budget}_5D_ela.csv"
            ela_path_switch = Path(ELA_DIR_SWITCH) / f"A1_B{budget}_5D_ela.csv"

            if not ela_path_algo.exists() or not ela_path_switch.exists():
                print("ELA file missing for this budget, skipping.")
                continue
                
            df_algo = pd.read_csv(ela_path_algo)
            df_switch = pd.read_csv(ela_path_switch).drop(columns=["switch"])

            df_switch = prepare_switch_data(df_switch)

            df_algo = df_algo.iloc[:, :-6]
            row_algo = df_algo[(df_algo["fid"] == fid) & (df_algo["iid"] == iid) & (df_algo["rep"] == rep)]
            row_switch = df_switch[(df_switch["fid"] == fid) & (df_switch["iid"] == iid) & (df_switch["rep"] == rep)]
            if row_algo.empty or row_switch.empty:
                print("No data row found for this configuration, skipping.")
                continue

            features_switch = row_switch.iloc[:, 4:]

            features_switch.index = [(fid, iid, rep)]
            should_switch = switch_model.predict(features_switch)[0]
            print(should_switch)

            if should_switch:
                print(f"Switching at budget {budget} for fid={fid}, iid={iid}, rep={rep}")
                features_algo = row_algo.iloc[:, 4:]
                features_algo.index = [(fid, iid, rep)]
                algo_prediction = perf_model.predict(features_algo)
                predicted_algorithm = list(algo_prediction.values())[0][0][0]

                match_row = self.precision_df[
                    (self.precision_df["fid"] == fid) &
                    (self.precision_df["iid"] == iid) &
                    (self.precision_df["rep"] == rep) &
                    (self.precision_df["budget"] == budget) &
                    (self.precision_df["algorithm"] == predicted_algorithm)
                ]

                precision = match_row["precision"].values[0] if not match_row.empty else np.inf
                return precision

        # Fallback to budget 1000 CMA-ES
        fallback_row = self.precision_df[
            (self.precision_df["fid"] == fid) &
            (self.precision_df["iid"] == iid) &
            (self.precision_df["rep"] == rep) &
            (self.precision_df["budget"] == 1000) &
            (self.precision_df["algorithm"] == "Non-elitist")
        ]
        fallback_precision = fallback_row["precision"].values[0] if not fallback_row.empty else np.inf
        return fallback_precision

def train_models_for_iid(test_iid, config, selector):
    train_iids = [iid for iid in IIDS if iid != test_iid]
    wrapper_partial = RandomForestClassifierWrapper.get_from_configuration(config, random_state=42)
    switching_models = {}
    performance_models = {}

    for budget in SWITCHING_BUDGETS:
        ela_path_switch = Path(ELA_DIR_SWITCH) / f"A1_B{budget}_5D_ela.csv"
        if not ela_path_switch.exists():
            continue
        train_df = pd.read_csv(ela_path_switch)

        train_df = prepare_switch_data(train_df)
        # train_df = train_df.drop(columns=["Elitist", "Non-elitist", "MLSL", "PSO", "DE", "BFGS"])
        train_df = train_df[train_df["iid"].isin(train_iids)]

        model = wrapper_partial()
        X_train = train_df.iloc[:, 4:].drop(columns=["switch"])
        y_train = train_df["switch"]
        model.fit(X_train, y_train)
        switching_models[budget] = model

        ela_path_algo = Path(ELA_DIR_ALGO) / f"A1_B{budget}_5D_ela.csv"
        if not ela_path_algo.exists():
            continue
        train_df = pd.read_csv(ela_path_algo)
        train_df = train_df[train_df["iid"].isin(train_iids)]
        X_train = train_df.iloc[:, 4:-6]
        y_train = train_df.iloc[:, -6:]

        trained_model_path = Path(CV_MODELS_DIR) / f"iid{test_iid}/selector_B{budget}_trained.pkl"
        if trained_model_path.exists():
            perf_model = joblib.load(trained_model_path)
        else:
            print(f"Training performance model for budget {budget}, iid {test_iid}")
            perf_model = joblib.load(f"{UNTRAINED_PERF_MODELS_DIR}/model_B{budget}.pkl").selector
            perf_model.fit(X_train, y_train)
            os.makedirs(os.path.dirname(trained_model_path), exist_ok=True)
            joblib.dump(perf_model, trained_model_path)

        performance_models[budget] = perf_model

    total_precision = 0.0
    for fid in FIDS:
        for rep in REPS:
            precision = selector.simulate_single_run(fid, test_iid, rep, switching_models, performance_models)
            total_precision += precision
    return total_precision

# ========== Objective function for SMAC ==========

def smac_objective(config, seed):
    np.random.seed(seed)
    selector = SwitchingSelectorCV(PRECISION_FILE)

    print(f"Evaluating config: {config}")
    with Pool(processes=5) as pool:  # Adjust number of processes
        results = pool.starmap(partial(train_models_for_iid, config=config, selector=selector), [(iid,) for iid in IIDS])

    total_cv_precision = sum(results)
    print(f"Config {config} → Total CV precision: {total_cv_precision}")
    return total_cv_precision

# ========== Main SMAC tuning routine ==========

def main():
    cs = RandomForestClassifierWrapper.get_configuration_space()

    scenario = Scenario(
        configspace=cs,
        n_trials=100,
        walltime_limit=np.inf,
        deterministic=True,
        output_directory=SMAC_OUTPUT_DIR,
        seed=42
    )

    smac = HyperparameterOptimizationFacade(scenario, smac_objective)
    best_config = smac.optimize()

    print("Best configuration found:")
    for k, v in best_config.items():
        print(f"  {k}: {v}")

    wrapper_partial = RandomForestClassifierWrapper.get_from_configuration(best_config, random_state=42)
    output_dir = Path(OUTPUT_PATH)
    output_dir.mkdir(parents=True, exist_ok=True)

    for budget in SWITCHING_BUDGETS:
        ela_path_switch = Path(ELA_DIR_SWITCH) / f"A1_B{budget}_5D_ela.csv"
        if not ela_path_switch.exists():
            continue
        train_df = pd.read_csv(ela_path_switch)

        train_df = prepare_switch_data(train_df)
        # train_df = train_df.drop(columns=["Elitist", "Non-elitist", "MLSL", "PSO", "DE", "BFGS"])
        X_train = train_df.iloc[:, 4:].drop(columns=["switch"])
        y_train = train_df["switch"]
        model = wrapper_partial()
        model.fit(X_train, y_train)
        model_path = output_dir / f"switching_model_B{budget}_trained.pkl"
        joblib.dump(model, model_path)
        print(f"Saved switching model for budget {budget} to {model_path}")

    print("All final switching models trained and saved successfully.")

if __name__ == "__main__":
    # Read the number of lookahead epms from command line
    
    if len(sys.argv) != 2:
        print("Usage: python switch_model_optimisation.py <num_lookahead_epms>")
        sys.exit(1)
    NUM_LOOKAHEAD_EPMS = int(sys.argv[1])

    SMAC_OUTPUT_DIR = f"smac_lookaheads_algo_features/smac_output_switch_lookahead_{NUM_LOOKAHEAD_EPMS}_normalized_afterwards"
    OUTPUT_PATH = f"../data/models/tuned_models/switching_models_lookahead_{NUM_LOOKAHEAD_EPMS}_normalized_afterwards"


    main()