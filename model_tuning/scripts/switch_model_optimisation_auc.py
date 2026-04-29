import numpy as np
import pandas as pd
from pathlib import Path
import os
import joblib
from functools import partial
from itertools import combinations
from multiprocessing import Pool
import sys
import tempfile

from sklearn.metrics import auc

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
ELA_DIR_SWITCH = "../data/A1_data_algo_features_switch_with_lookahead_all_epms"
ELA_DIR_ALGO = "../data/A1_data_ela_normalized_with_precisions"
PRECISION_FILE = "../data/A2_precisions_normalized_log10.csv"
CV_MODELS_DIR = "../data/models/trained_models/algo_performance_models_cv_algo_features"
UNTRAINED_PERF_MODELS_DIR = "../data/models/untrained_models/algo_performance_models_algo_features"

# === Overwritten by command line arguments ===
SMAC_OUTPUT_DIR = f"smac_lookaheads/smac_output_switch_lookahead_{NUM_LOOKAHEAD_EPMS}"
OUTPUT_PATH = f"../data/models/tuned_models/switching_models_lookahead_{NUM_LOOKAHEAD_EPMS}"
# =============================================


# ========== Helper classes ==========
def prepare_switch_data(df_switch):
    # This function either drops or keeps algo features, and keeps the right amount of lookahead predictions based on the global config variables
    if USE_ALGO_FEATURES:
        df_switch = df_switch.drop(columns=["pred_t0"], errors="ignore")
    else:
        df_switch = df_switch.drop(columns=["BFGS", "DE", "PSO", "MLSL", "Non-elitist", "Elitist"], errors="ignore")
    
    if NUM_LOOKAHEAD_EPMS is not None:
        df_switch = df_switch.drop(columns=[f"pred_t{t}" for t in range(NUM_LOOKAHEAD_EPMS + 1, 20)], errors="ignore")
    elif NUM_LOOKAHEAD_EPMS == -1:
        df_switch = df_switch.drop(columns=[f"pred_t{t}" for t in range(0, 20)], errors="ignore")  
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

                # match_row = self.precision_df[
                #     (self.precision_df["fid"] == fid) &
                #     (self.precision_df["iid"] == iid) &
                #     (self.precision_df["rep"] == rep) &
                #     (self.precision_df["budget"] == budget) &
                #     (self.precision_df["algorithm"] == predicted_algorithm)
                # ]

                run_file = pd.read_csv(f"../data/padded_runs/padded_A2_{predicted_algorithm}_B{budget}_5D.csv")

                run_file = run_file[
                    (run_file["fid"] == fid) &
                    (run_file["iid"] == iid) &
                    (run_file["rep"] == rep)
                ]

                print(len(run_file))

                print(run_file["rep"].nunique() == 1)

                auc_value = auc(run_file["evaluations"], np.log10(np.clip(run_file["raw_y"], 1e-12, None)))

                return auc_value

        # Fallback to budget 1000 CMA-ES
        # fallback_row = self.precision_df[
        #     (self.precision_df["fid"] == fid) &
        #     (self.precision_df["iid"] == iid) &
        #     (self.precision_df["rep"] == rep) &
        #     (self.precision_df["budget"] == 1000) &
        #     (self.precision_df["algorithm"] == "Non-elitist")
        # ]

        fallback_file = pd.read_csv(f"../data/padded_runs/padded_A2_Non-elitist_B1000_5D.csv")

        fallback_file = fallback_file[
                    (fallback_file["fid"] == fid) &
                    (fallback_file["iid"] == iid) &
                    (fallback_file["rep"] == rep)
        ]


        print(len(fallback_file))

        print(fallback_file["rep"].nunique() == 1)

        auc_value = auc(fallback_file["evaluations"], np.log10(np.clip(fallback_file["raw_y"], 1e-12, None)))

        return auc_value


        # fallback_precision = fallback_row["precision"].values[0] if not fallback_row.empty else np.inf
        # return fallback_precision

def train_models_for_iid(test_iid, config, selector):
    train_iids = [iid for iid in IIDS if iid != test_iid]
    wrapper_partial = RandomForestClassifierWrapper.get_from_configuration(config, random_state=42)
    switching_models = {}
    performance_models = {}

    for budget in SWITCHING_BUDGETS:
        ela_path_switch = Path(ELA_DIR_SWITCH) / f"A1_B{budget}_5D_ela.csv"
        if not ela_path_switch.exists():
            print(f"ELA file for switching features missing for budget {budget}, skipping model training.")

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
            print(f"ELA file for algorithm features missing for budget {budget}, skipping performance model training.")
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

    total_auc = 0.0
    for fid in FIDS:
        for rep in REPS:
            auc_value = selector.simulate_single_run(fid, test_iid, rep, switching_models, performance_models)
            total_auc += auc_value
    return total_auc

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

def atomic_joblib_dump(obj, final_path: Path, compress=0):
    final_path = Path(final_path)
    final_path.parent.mkdir(parents=True, exist_ok=True)

    # Create temp file in the same directory (required for atomic rename)
    with tempfile.NamedTemporaryFile(
        dir=final_path.parent, prefix=final_path.name + ".", suffix=".tmp", delete=False
    ) as tmp:
        tmp_path = Path(tmp.name)

    try:
        # Write pickle to temp file
        joblib.dump(obj, tmp_path, compress=compress)

        # Ensure bytes are flushed to disk
        with open(tmp_path, "rb") as f:
            os.fsync(f.fileno())

        # Atomically replace target
        os.replace(tmp_path, final_path)
    finally:
        # Cleanup if something failed before replace
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass

def train_default_switching_model():
    for budget in SWITCHING_BUDGETS:
        ela_path_switch = Path(ELA_DIR_SWITCH) / f"A1_B{budget}_5D_ela.csv"
        if not ela_path_switch.exists():
            print(f"ELA file for switching features missing for budget {budget}, skipping model training.")
            return None
        train_df = pd.read_csv(ela_path_switch)

        train_df = prepare_switch_data(train_df)
        X_train = train_df.iloc[:, 4:].drop(columns=["switch"])
        y_train = train_df["switch"]

        cs = RandomForestClassifierWrapper.get_configuration_space()
        default_config = cs.get_default_configuration()
        model = RandomForestClassifierWrapper.get_from_configuration(default_config, random_state=42)()

        model.fit(X_train, y_train)
        os.makedirs(OUTPUT_PATH, exist_ok=True)
        model_path = Path(OUTPUT_PATH) / f"switching_model_B{budget}_untuned_trained.pkl"
        atomic_joblib_dump(model, model_path)
        print(f"Trained and saved default switching model for budget {budget} to {model_path}")



if __name__ == "__main__":
    # Read the number of lookahead epms from command line
    
    if len(sys.argv) != 2:
        print("Usage: python switch_model_optimisation.py <num_lookahead_epms>")
        sys.exit(1)
    NUM_LOOKAHEAD_EPMS = int(sys.argv[1])

    # NUM_LOOKAHEAD_EPMS = 2

    SMAC_OUTPUT_DIR = f"smac_lookaheads_auc/smac_output_switch_lookahead_auc_{NUM_LOOKAHEAD_EPMS}"
    OUTPUT_PATH = f"../data/models/tuned_models/auc/switching_models_lookahead_auc_{NUM_LOOKAHEAD_EPMS}"

    # train_default_switching_model()
    main()

   