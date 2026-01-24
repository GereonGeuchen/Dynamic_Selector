import joblib
import pandas as pd
import os
from functools import reduce
import numpy as np
import warnings

SELECTOR_DIR = "../data/models/algo_performance_models_algo_features"
ELA_TEMPLATE = "../data/ela/A1_data_ela_normalized_with_precisions/A1_B{budget}_5D_ela.csv"
OUTPUT_DIR = "../data/selector_performances/algo_features"

def crossvalidated_static_predictions(
    budget,
    selector_dir=SELECTOR_DIR,
    ela_template=ELA_TEMPLATE,
    precision_df=None
):
    selector_path = os.path.join(selector_dir, f"model_B{budget}.pkl")

    df = pd.read_csv(ela_template.format(budget=budget))
    X = df.iloc[:, 4:-6]
    y = df.iloc[:, -6:]
    meta = df[["fid", "iid", "rep"]]
    X.index = y.index = list(zip(meta["fid"], meta["iid"], meta["rep"]))

    predictions_results = []
    precision_results = []
    algorithm_results = []

    test_values = sorted(meta["iid"].unique())
    test_column = "iid"

    for test_fold in test_values:
        mask = meta[test_column] == test_fold

        print(f"Processing test {test_column} {test_fold} for budget {budget}...")

        train_keys = list(meta[~mask][["fid", "iid", "rep"]].itertuples(index=False, name=None))
        test_keys = list(meta[mask][["fid", "iid", "rep"]].itertuples(index=False, name=None))

        X_train, y_train = X.loc[train_keys], y.loc[train_keys]
        X_test = X.loc[test_keys]

        # Load the selector model
        pipeline = joblib.load(selector_path)
        selector = pipeline.selector

        selector.algorithms = list(y.columns)
        selector.fit(X_train, y_train)

        print(budget, selector.regressors[0].model_class.get_params())

        predictions = selector.predict(X_test)

        for (fid, iid, rep), [(algo, _)] in predictions.items():
            precision_results.append({
                "fid": fid,
                "iid": iid,
                "rep": rep,
                    f"static_B{budget}": precision_df.loc[
                    (precision_df["fid"] == fid) & 
                    (precision_df["iid"] == iid) & 
                    (precision_df["rep"] == rep) & 
                    (precision_df["budget"] == budget) & 
                    (precision_df["algorithm"] == algo),
                    "precision"
                ].values[0] if not precision_df.empty else None
            })
            algorithm_results.append({
                "fid": fid,
                "iid": iid,
                "rep": rep,
                f"alg_B{budget}": algo
            })


        # === NEW: full normalized predictions (raw values and variance) ===
        preds_raw = selector.generate_features(X_test)  # shape (n_test, n_algorithms)

        # Get uncertainties of predictions
        vars_by_algo = []

        for i in range(6):
            random_forest = selector.regressors[i].model_class
            random_forest_predictions = []

            for estimator in random_forest.estimators_:
                random_forest_predictions.append(
                    estimator.predict(X_test).reshape(-1, 1)
                )

            var = np.var(np.concatenate(random_forest_predictions, axis=1), axis=1)
            vars_by_algo.append(var)  

        # Build DataFrame with algorithm predictions
        preds_df = pd.DataFrame(
            preds_raw,
            columns=selector.algorithms,
        )

        # Add 6 variance columns
        for i, algo in enumerate(selector.algorithms):
            preds_df[f"var_{algo}"] = vars_by_algo[i]


        fid_list, iid_list, rep_list = zip(*test_keys)

        preds_df.insert(0, "fid", fid_list)
        preds_df.insert(1, "iid", iid_list)
        preds_df.insert(2, "rep", rep_list)
        preds_df.insert(3, "budget", budget)

        predictions_results.append(preds_df)

    preds_full = pd.concat(predictions_results, ignore_index=True)

    return (
        pd.DataFrame(precision_results),
        pd.DataFrame(algorithm_results),
        preds_full,  # normalized predictions
    )

    return pd.DataFrame(precision_results), pd.DataFrame(algorithm_results)



def build_full_crossvalidated_table(precision_path, output_dir = OUTPUT_DIR):
    all_dfs = []
    all_algos = []
    all_preds = []

    os.makedirs(output_dir, exist_ok=True)
    precision_output = os.path.join(output_dir, "predicted_static_precisions.csv")
    algo_output = os.path.join(output_dir, "selected_algorithms.csv")
    preds_output = os.path.join(output_dir, "all_normalized_predictions.csv")

    precision_df = pd.read_csv(precision_path)

    budgets = [50*i for i in range(1, 21)]

    for budget in budgets:
        print(f"Processing budget {budget}...")

        if budget < 1000:
            df_b, df_a, df_p = crossvalidated_static_predictions(budget, precision_df=precision_df)
            all_preds.append(df_p)
        else:
            # Use precision and algorithm "Same" directly
            df_b = precision_df.query("budget == 1000 and algorithm == 'Non-elitist'")
            df_b = df_b[["fid", "iid", "rep", "precision"]].rename(columns={"precision": "static_B1000"})

            df_a = df_b[["fid", "iid", "rep"]].copy()
            df_a["alg_B1000"] = "Non-elitist"
            df_p = None

        all_dfs.append(df_b)
        all_algos.append(df_a)

        # Save merged results incrementally
        df_prec = reduce(lambda l, r: pd.merge(l, r, on=["fid", "iid", "rep"], how="outer"), all_dfs)
        df_algo = reduce(lambda l, r: pd.merge(l, r, on=["fid", "iid", "rep"], how="outer"), all_algos)

        df_prec = df_prec.sort_values(["fid", "iid", "rep"]).reset_index(drop=True)
        df_algo = df_algo.sort_values(["fid", "iid", "rep"]).reset_index(drop=True)

        df_prec.to_csv(precision_output, index=False)
        df_algo.to_csv(algo_output, index=False)

        if all_preds:
            df_pred = pd.concat(all_preds, ignore_index=True)
            df_pred = df_pred.sort_values(["fid", "iid", "rep", "budget"]).reset_index(drop=True)
            df_pred.to_csv(preds_output, index=False)
            print(f"Saved: predictions for budgets <1000.")

        print(f"Saved: {precision_output}, {algo_output} [budget {budget}]")

    return df_prec, df_algo, df_pred if all_preds else None


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        build_full_crossvalidated_table(
            "../data/A2_precisions.csv"
        )