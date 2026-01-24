import os
import joblib
import numpy as np
import pandas as pd
import warnings

MODEL_DIR = "../data/models/lookahead_models_untrained"
ELA_TEMPLATE = "../data/ela/A1_data_ela_normalized_with_future_performances/A1_B{budget}_5D_ela.csv"

OUTPUT_DIR = "../data/lookahead_performances/just_ela"
OUT_FILE = "predicted_switchpoint_performances.csv"

MODEL_TEMPLATE = "lookahead_model_B{budget}_t{t}_untrained.pkl"

TARGET_COLS = {
    1: "best_precision_t+1",
    2: "best_precision_t+2",
    3: "best_precision_t+3",
}


def available_ts_for_budget(budget: int):
    if budget == 950:
        return [1]
    if budget == 900:
        return [1, 2]
    return [1, 2, 3]


def load_factory(model_dir: str, budget: int, t: int):
    path = os.path.join(model_dir, MODEL_TEMPLATE.format(budget=budget, t=t))
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing model factory: {path}")
    obj = joblib.load(path)
    return obj


def crossvalidated_switchpoint_predictions(budget: int):
    df = pd.read_csv(ELA_TEMPLATE.format(budget=budget))

    ts_budget = available_ts_for_budget(budget)

    # Features: keep consistent with your previous script / training
    X = df.iloc[:, 4:-len(ts_budget)]  # all columns except first 4 and last n target cols
    meta = df[["fid", "iid", "rep"]]
    keys = list(zip(meta["fid"], meta["iid"], meta["rep"]))
    X.index = keys

    # Targets by explicit column names (whatever exists)
    existing = {t: c for t, c in TARGET_COLS.items() if c in df.columns}
    if not existing:
        raise ValueError(f"No target columns found in CSV for budget={budget}. Expected one of {list(TARGET_COLS.values())}")

    y = df[list(existing.values())].copy()
    # rename to target_t1/2/3 for convenience
    rename_map = {col: f"target_t{t}" for t, col in existing.items()}
    y = y.rename(columns=rename_map)
    y.index = keys

    test_values = sorted(meta["iid"].unique())

    # Only train for t where BOTH (a) budget allows it and (b) column exists
    ts = [t for t in ts_budget if f"target_t{t}" in y.columns]

    out_rows = []

    for test_iid in test_values:
        print(f"  - Testing IID {test_iid}")
        mask = meta["iid"] == test_iid

        train_keys = list(meta[~mask][["fid", "iid", "rep"]].itertuples(index=False, name=None))
        test_keys  = list(meta[mask][["fid", "iid", "rep"]].itertuples(index=False, name=None))

        X_train = X.loc[train_keys]
        X_test  = X.loc[test_keys]
        y_train = y.loc[train_keys]

        preds_by_t = {}

        for t in ts:
            model = load_factory(MODEL_DIR, budget, t)
            

            y_train_t = y_train[f"target_t{t}"].to_numpy()
            good = ~np.isnan(y_train_t)
            X_train_t = X_train.iloc[good]
            y_train_t = y_train_t[good]

            if len(y_train_t) == 0:
                preds_by_t[t] = np.full((len(test_keys),), np.nan)
                continue

            model.fit(X_train_t, np.log10(y_train_t))
            preds_by_t[t] = np.asarray(model.predict(X_test)).reshape(-1)

        # store rows
        for i, (fid, iid, rep) in enumerate(test_keys):
            row = {"fid": fid, "iid": iid, "rep": rep, "budget": budget}
            row["pred_t1"] = float(preds_by_t[1][i]) if 1 in preds_by_t else np.nan
            row["pred_t2"] = float(preds_by_t[2][i]) if 2 in preds_by_t else np.nan
            row["pred_t3"] = float(preds_by_t[3][i]) if 3 in preds_by_t else np.nan
            out_rows.append(row)

    return pd.DataFrame(out_rows)


def build_full_crossvalidated_switchpoint_table():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, OUT_FILE)

    budgets = [50 * i for i in range(1, 21)]
    all_preds = []

    for budget in budgets:
        print(f"\n=== Processing budget {budget} ===")
        df_b = crossvalidated_switchpoint_predictions(budget)
        all_preds.append(df_b)

        df_out = pd.concat(all_preds, ignore_index=True)
        df_out = df_out.sort_values(["fid", "iid", "rep", "budget"]).reset_index(drop=True)
        df_out.to_csv(out_path, index=False)
        print(f"Saved: {out_path} [through budget {budget}]")

    return df_out


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        build_full_crossvalidated_switchpoint_table()
