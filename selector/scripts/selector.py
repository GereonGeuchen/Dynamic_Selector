import joblib
import pandas as pd
from pathlib import Path
import os
import numpy as np
import warnings
import sys
from asf.predictors import RandomForestRegressorWrapper

USE_ALGO_FEATURES = False  # Whether to use the ELA features for the performance models or not, will be set from command line argument
NUM_LOOKAHEAD_EPMS = 0  # This will be set from command line argument

### Will be overwritten by command line arguments, but set to some default values for now
SWITCHING_MODEL_DIR = "../data/trained_models/untuned_models/auc/switching_models_lookahead_untuned"
SAVE_PATH = "../results/selector_results_with_lookahead_test.csv"
###

SELECTOR_MODEL_DIR = "../data/trained_models/untuned_models/auc/algo_performance_models_trained_untuned_auc"
ELA_DIR = "../data/A1_data_ela_test_normalized"
PRECISION_FILE = "../data/A2_precisions_test.csv"
BUDGETS = list(range(50, 1001, 50))
LOOKAHEAD_MODELS_DIRECTORY = "../data/trained_models/untuned_models/auc/lookahead_models_all_epms_auc"  # Directory where lookahead models are stored, e.g., lookahead_model_B500_t1_untuned_trained.pkl, lookahead_model_B500_t2_untuned_trained.pkl, etc.


class SwitchingSelector:
    def __init__(self, switching_model_dir=SWITCHING_MODEL_DIR, selector_model_dir=SELECTOR_MODEL_DIR, lookahead_models_directory=LOOKAHEAD_MODELS_DIRECTORY):
       
        self.switching_prediction_models = {}
        self.performance_models = {}

        switching_model_dir = Path(switching_model_dir)
        selector_model_dir = Path(selector_model_dir)
        print("Loading models...")
        print(f"Switching model dir: {switching_model_dir}")
        print(f"Selector model dir: {selector_model_dir}")
        # Load switching predictor models
        for model_path in switching_model_dir.glob("switching_model_B*_untuned_trained.pkl"):
            budget = int(model_path.stem.split("_")[2][1:])  # e.g., switching_model_B500 → 500
            self.switching_prediction_models[budget] = joblib.load(model_path)
            print(self.switching_prediction_models[budget].model_class.get_params())
            print(f"Loaded switching model for budget {budget}")

        # Load lookahead models if provided
        if lookahead_models_directory:
            lookahead_models_directory = Path(lookahead_models_directory)
            self.lookahead_models = {}
            for i in range(0, NUM_LOOKAHEAD_EPMS + 1):
                if USE_ALGO_FEATURES and i == 0: continue  # t0 predictions are already included as algo features
                for model_path in lookahead_models_directory.glob(f"lookahead_model_B*_t{i}_untuned_trained.pkl"):
                    budget = int(model_path.stem.split("_")[2][1:])  # e.g., lookahead_model_B500 → 500
                    self.lookahead_models[budget, i] = joblib.load(model_path)
                    print(f"Loaded lookahead model for budget {budget, i}: ")
                    print(self.lookahead_models[budget, i].model_class.get_params())

        # Load performance predictors
        for model_path in selector_model_dir.glob("selector_B*_untuned_trained.pkl"):
            budget = int(model_path.stem.split("_")[1][1:])  # e.g., selector_B1000_model → 1000
            self.performance_models[budget] = joblib.load(model_path)
            print(f"Loaded performance model for budget {budget}: ")
            print(self.performance_models[budget].regressors[0].model_class.get_params())

        

    def simulate_single_run(self, fid, iid, rep, ela_dir=ELA_DIR, precision_file=PRECISION_FILE, budgets=BUDGETS):

        precision_df = pd.read_csv(precision_file)
        for budget in budgets:
            ela_path = Path(ela_dir) / f"A1_B{budget}_5D_ela.csv"
            if not ela_path.exists():
                print("Ela path does not exist")
                continue

            df = pd.read_csv(ela_path)
            row = df[(df["fid"] == fid) & (df["iid"] == iid) & (df["rep"] == rep)]

            if row.empty:
                continue

            # Use ELA + CMA state only (skip id, fid, iid, rep, high_level_category)
            features = row.iloc[:, 4:]
            features.index = [(fid, iid, rep)]

            # Predict switching decision: True or False
            switch_model = self.switching_prediction_models.get(budget)
            performance_model = self.performance_models.get(budget)

            if performance_model is None:
                continue

            # First get all algo predictions of the performance model
            if USE_ALGO_FEATURES:
                algo_predictions = performance_model.generate_features(features)

                colnames = ["BFGS","DE","Elitist","MLSL","Non-elitist","PSO"]

                switching_data_df = pd.DataFrame(algo_predictions, index=features.index, columns=colnames)
            else:  
                # switching_data_df empty
                switching_data_df = pd.DataFrame(index=features.index)

            # Get variances of predictions

            # var_colnames = [f"var_{name}" for name in colnames]

            # vars_by_algo = []

            # for i in range(6):
            #     random_forest = performance_model.regressors[i].model_class
            #     random_forest_predictions = []

            #     for estimator in random_forest.estimators_:
            #         random_forest_predictions.append(
            #             estimator.predict(features).reshape(-1, 1)
            #         )

            #     var = np.var(np.concatenate(random_forest_predictions, axis=1), axis=1)
            #     vars_by_algo.append(var)  

            # for i, var in enumerate(vars_by_algo):
            #     algo_df[var_colnames[i]] = var

            # Get predictions from lookahead model if available. For budgets 900, we have t1, t2; for budget 950, t1 only.
            for i in range(0, NUM_LOOKAHEAD_EPMS + 1):
                if USE_ALGO_FEATURES and i == 0: continue  # t0 predictions are already included as algo features

                if (budget, i) in self.lookahead_models:
                    lookahead_model_t = self.lookahead_models[(budget, i)]
                    lookahead_pred_t = lookahead_model_t.predict(features)[0]
                    switching_data_df[f"pred_t{i}"] = lookahead_pred_t


            if switch_model is None:
                print(f"No switching model for budget {budget}, skipping...")
                continue

            # Attach predicted algorithm performances to features for switching decision
            switching_features = pd.concat([features, switching_data_df], axis=1)

            # print(f"Switching features for budget {budget}: {switching_features}")

            should_switch = switch_model.predict(switching_features)[0]
            # print(f"Switching prediction probability for budget {budget}: {should_switch}")
            # should_switch = prediction[0] # if hasattr(prediction, "__len__") else prediction

            if should_switch:
                # print(f"Switching at budget {budget} for fid={fid}, iid={iid}, rep={rep}")
                
                # Now decide which algorithm to switch to
                if performance_model is None:
                    print(f"No performance model for budget {budget}, skipping...")
                    continue

                algo_prediction = performance_model.predict(features)
                predicted_algorithm = list(algo_prediction.values())[0][0][0]

                # Look up precision for selected algorithm
                match_row = precision_df[
                    (precision_df["fid"] == fid) &
                    (precision_df["iid"] == iid) &
                    (precision_df["rep"] == rep) &
                    (precision_df["budget"] == budget) &
                    (precision_df["algorithm"] == predicted_algorithm)
                ]
                precision = match_row["precision"].values[0] if not match_row.empty else None

                vbs_precision = precision_df[
                    (precision_df["fid"] == fid) &
                    (precision_df["iid"] == iid) &
                    (precision_df["rep"] == rep)
                ]["precision"].min()

                return {
                    "fid": fid,
                    "iid": iid,
                    "rep": rep,
                    "switch_budget": budget,
                    "selected_algorithm": predicted_algorithm,
                    "predicted_precision": precision,
                    "vbs_precision": vbs_precision
                }

        # No budget triggered a switch → fallback
        fallback_budget = 1000
        fallback_algorithm = "Non-elitist"

        match_row = precision_df[
            (precision_df["fid"] == fid) &
            (precision_df["iid"] == iid) &
            (precision_df["rep"] == rep) &
            (precision_df["budget"] == fallback_budget)
        ]

        if not match_row.empty:
            precision = match_row[match_row["algorithm"] == fallback_algorithm]["precision"]
            precision = precision.values[0] if not precision.empty else None
        else:
            precision = None

        vbs_precision = precision_df[
            (precision_df["fid"] == fid) &
            (precision_df["iid"] == iid) &
            (precision_df["rep"] == rep)
        ]["precision"].min()

        return {
            "fid": fid,
            "iid": iid,
            "rep": rep,
            "switch_budget": None,
            "selected_algorithm": fallback_algorithm,
            "predicted_precision": precision,
            "vbs_precision": vbs_precision
        }


    def evaluate_selector_to_csv(
    self,
    fids,
    iids,
    reps,
    save_path="selector_results.csv",
    ela_dir="../data/ela_with_state_test_data",
    precision_file="../data/A2_precisions_test.csv"
    ):
        precision_df = pd.read_csv(precision_file)
        budgets = list(range(50, 1001, 50))
        # budgets = [8*i for i in range(1, 13)] + [50*i for i in range(2, 21)]  # Budgets from 50 to 1000 in steps of 50

        # Ensure output directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        for fid in fids:
            for iid in iids:
                for rep in reps:
                    print(f"Processing (fid={fid}, iid={iid}, rep={rep})...")

                    # Get VBS precision
                    # vbs_precision = precision_df[
                    #     (precision_df["fid"] == fid) &
                    #     (precision_df["iid"] == iid) &
                    #     (precision_df["rep"] == rep)
                    # ]["precision"].min()

                    row = {
                        "fid": fid,
                        "iid": iid,
                        "rep": rep,
                    #     "vbs_precision": vbs_precision,
                    }

                    # Selector result
                    result = self.simulate_single_run(fid, iid, rep, ela_dir, precision_file, budgets=budgets)
                    row["vbs_precisions"] = result["vbs_precision"]
                    row["selector_precision"] = result["predicted_precision"]
                    row["selector_switch_budget"] = result["switch_budget"] or 1000
                    row["selector_algorith"] = result["selected_algorithm"]

                    # Static switchers
                    for b in budgets:
                        col_name = f"static_B{b}"
                        if b < 1000:
                            ela_path = Path(ela_dir) / f"A1_B{b}_5D_ela.csv"
                            if not ela_path.exists():
                                row[col_name] = None
                                continue

                            df = pd.read_csv(ela_path)
                            instance_row = df[
                                (df["fid"] == fid) &
                                (df["iid"] == iid) &
                                (df["rep"] == rep)
                            ]
                            if instance_row.empty:
                                row[col_name] = None
                                continue

                            features = instance_row.iloc[:, 4:]
                            features.index = [(fid, iid, rep)]

                            model = self.performance_models.get(b)
                            if model is None:
                                row[col_name] = None
                                continue

                            algo_pred = model.predict(features)
                            algo = list(algo_pred.values())[0][0][0]

                            match = precision_df[
                                (precision_df["fid"] == fid) &
                                (precision_df["iid"] == iid) &
                                (precision_df["rep"] == rep) &
                                (precision_df["budget"] == b) &
                                (precision_df["algorithm"] == algo)
                            ]
                            row[col_name] = match["precision"].values[0] if not match.empty else None
                        else:
                            # Budget 1000 → use CMA-ES directly
                            match = precision_df[
                                (precision_df["fid"] == fid) &
                                (precision_df["iid"] == iid) &
                                (precision_df["rep"] == rep) &
                                (precision_df["budget"] == 1000) &
                                (precision_df["algorithm"] == "Non-elitist")
                            ]
                            row[col_name] = match["precision"].values[0] if not match.empty else None

                    # Append row to CSV
                    row_df = pd.DataFrame([row])
                    row_df.to_csv(save_path, mode="a", index=False, header=not os.path.exists(save_path))

        print(f"Incremental results saved to: {save_path}")

    # Only record choices of performance model at B=150 from Kostovska et al.
    def record_B150(self, ela_dir=ELA_DIR, precision_file=PRECISION_FILE):
        precision_df = pd.read_csv(precision_file)
        ela_path = Path(ela_dir) / f"A1_B150_5D_ela.csv"
        df = pd.read_csv(ela_path)

        model = self.performance_models.get(150)
        if model is None:
            print("No performance model for budget 150, skipping...")
            return

        records = []
        for fid in range(1, 25):
            for iid in [6, 7]:
                for rep in range(20):
                    print(f"Processing (fid={fid}, iid={iid}, rep={rep}) for B=150...")
                    row = df[
                        (df["fid"] == fid) &
                        (df["iid"] == iid) &
                        (df["rep"] == rep)
                    ]

                    if row.empty:
                        continue

                    features = row.iloc[:, 4:]
                    features.index = [(fid, iid, rep)]

                    algo_pred = model.predict(features)
                    algo = list(algo_pred.values())[0][0][0]

                    match = precision_df[
                        (precision_df["fid"] == fid) &
                        (precision_df["iid"] == iid) &
                        (precision_df["rep"] == rep) &
                        (precision_df["budget"] == 150) &
                        (precision_df["algorithm"] == algo)
                    ]
                    precision = match["precision"].values[0] if not match.empty else None

                    records.append({
                        "fid": fid,
                        "iid": iid,
                        "rep": rep,
                        "selected_algorithm": algo,
                        "precision": precision
                    })

        result_df = pd.DataFrame(records)
        result_df.to_csv("../results/B150_performance_model_choices.csv", index=False)
        print("B=150 performance model choices saved to ../results/B150_performance_model_choices.csv")


if __name__ == "__main__":

    #Read number of EPMs from command line argument
    NUM_LOOKAHEAD_EPMS = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    # NUM_LOOKAHEAD_EPMS = 2

    if USE_ALGO_FEATURES:
        SWITCHING_MODEL_DIR = f"../data/trained_models/switching_models_all_epms_algo_features_normalized_afterwards/switching_models_lookahead_algo_features_{NUM_LOOKAHEAD_EPMS}_normalized_afterwards"
        SAVE_PATH = f"../results/all_epms_algo_features_normalized_afterwards/selector_results_with_lookahead_all_epms_algo_features_{NUM_LOOKAHEAD_EPMS}.csv"
    else:
        SWITCHING_MODEL_DIR = f"../data/trained_models/untuned_models/auc/switching_models_lowest/switching_models_lookahead_untuned_{NUM_LOOKAHEAD_EPMS}"
        SAVE_PATH = f"../results/auc/lowest/selector_results_with_lookahead_all_epms_{NUM_LOOKAHEAD_EPMS}.csv"


    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        selector = SwitchingSelector(
            switching_model_dir=SWITCHING_MODEL_DIR,
            selector_model_dir=SELECTOR_MODEL_DIR,
            lookahead_models_directory=LOOKAHEAD_MODELS_DIRECTORY,
        )
        selector.evaluate_selector_to_csv(
            fids=list(range(1, 25)),
            iids=[6, 7],
            reps=list(range(20)),
            save_path=SAVE_PATH,
            ela_dir=ELA_DIR,
            precision_file=PRECISION_FILE
        )
        # selector.record_B150(ela_dir=ELA_DIR, precision_file=PRECISION_FILE)