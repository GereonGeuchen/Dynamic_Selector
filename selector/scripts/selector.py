import joblib
import pandas as pd
from pathlib import Path
import os

class SwitchingSelector:
    def __init__(self, selector_model_dir="switching_prediction_models", performance_model_dir="algo_performance_models"):
    
        self.switching_prediction_models = {}
        self.performance_models = {}

        selector_model_dir = Path(selector_model_dir)
        performance_model_dir = Path(performance_model_dir)

        # Load switching predictor models
        for model_path in selector_model_dir.glob("switching_model_B*_trained.pkl"):
            budget = int(model_path.stem.split("_")[2][1:])  # e.g., selector_model_B500 → 500
            self.switching_prediction_models[budget] = joblib.load(model_path)
            print(self.switching_prediction_models[budget].model_class.get_params())
            print(f"Loaded switching model for budget {budget}")

        # Load performance predictors
        for model_path in performance_model_dir.glob("selector_B*_trained.pkl"):
            budget = int(model_path.stem.split("_")[1][1:])  # e.g., performance_B1000_model → 1000
            self.performance_models[budget] = joblib.load(model_path)
            print(f"Loaded performance model for budget {budget}: ")
            print(self.performance_models[budget].regressors[0].model_class.get_params())

    def simulate_single_run(self, fid, iid, rep, ela_dir="../data/ela_with_state_test_data", precision_file="../data/A2_precisions_test.csv", budgets=range(50, 1001, 50)):

        precision_df = pd.read_csv(precision_file)
        for budget in budgets:
            ela_path = Path(ela_dir) / f"A1_B{budget}_5D_ela_with_state.csv"
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
            if switch_model is None:
                continue

            # New: binary classification
            prediction = switch_model.predict(features)
            should_switch = prediction[0] # if hasattr(prediction, "__len__") else prediction
            if should_switch:
                # Now decide which algorithm to switch to
                performance_model = self.performance_models.get(budget)
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
        # budgets = list(range(50, 1001, 50))
        budgets = [8*i for i in range(1, 13)] + [50*i for i in range(2, 21)]  # Budgets from 50 to 1000 in steps of 50

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
                            ela_path = Path(ela_dir) / f"A1_B{b}_5D_ela_with_state.csv"
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

if __name__ == "__main__":
    selector = SwitchingSelector(
        selector_model_dir="../data/trained_models/switching_models_highest",
        performance_model_dir="../data/trained_models/algo_performance_models_trained"
    )
    selector.evaluate_selector_to_csv(
        fids=list(range(1, 25)),
        iids=[6, 7],
        reps=list(range(20)),
        save_path="../results/selector_results_algo_highest.csv",
        ela_dir="../data/A1_data_5D_test",
        precision_file="../data/A2_precisions_test.csv"
    )
    # tuned_model = joblib.load("../data/models/tuned_models/switching_models_normalized/switching_model_B500_trained.pkl")
    # untuned_model = joblib.load("../data/models/trained_models/switching_normalized/selector_B500_trained.pkl")
    # print(tuned_model.model_class.get_params())
    # print(untuned_model.get_params())