import os
from asf.predictors import RandomForestRegressorWrapper
from smac import HyperparameterOptimizationFacade, Scenario

import pandas as pd
from pathlib import Path
import numpy as np
import joblib
import sys

from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score


# === Parameters ===
ela_data_dir = "../data/A1_data_ela_normalized_with_future_performances_20"
smac_output_dir = "smac_outputs/smac_output_lookahead_optimisation"
output_models_dir = "../data/models/tuned_models/lookahead_models_trained"
untrained_output_models_dir = "../data/models/untrained_models/lookahead_models_untrained"


def _target_col(horizon: int) -> str:
    if horizon not in range(1, 20):
        raise ValueError(f"horizon must be in 1..19, got {horizon}")
    return f"best_precision_t+{horizon}"


def _make_X_y_groups(df: pd.DataFrame, horizon: int):
    """
    Build features X, target y, and CV groups for a given horizon.
    Ensures no label leakage by dropping ALL columns starting with 'best_precision'.
    Also drops 'iid' from X to avoid instance-id leakage across GroupKFold.
    """
    target = _target_col(horizon)
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in file.")

    y = df[target]
    groups = df["iid"]

    # Build X: drop label columns first
    X = df.drop(columns=[c for c in df.columns if c.startswith("best_precision")])

    # Drop first 3 columns (index-like columns) from features
    X = X.iloc[:, 4:]

    # Drop iid from features (you CV-split on iid)
    if "iid" in X.columns:
        X = X.drop(columns=["iid"])

    return X, y, groups


def make_smac_objective_for_budget_and_horizon(budget: int, horizon: int):
    """
    Returns SMAC objective(config, seed) -> loss for this (budget, horizon).
    Uses 5-fold GroupKFold over iid and optimizes 1 - mean_R2 (SMAC minimizes).
    """
    ela_path = Path(ela_data_dir) / f"A1_B{budget}_5D_ela.csv"
    df = pd.read_csv(ela_path)

    X_all, y_all, groups = _make_X_y_groups(df, horizon=horizon)

    # You said there are exactly five iids
    n_groups = groups.nunique()
    if n_groups != 5:
        raise ValueError(f"Expected exactly 5 unique iids, got {n_groups} in {ela_path.name}")

    cv = GroupKFold(n_splits=5)

    def objective(config, seed: int = 42):
        wrapper_partial = RandomForestRegressorWrapper.get_from_configuration(config, random_state=seed)

        r2s = []
        for tr_idx, va_idx in cv.split(X_all, y_all, groups):
            model = wrapper_partial()

            # Train on log10 of target to stabilize variance
            y_tr = np.log10(y_all.iloc[tr_idx])
            y_va = np.log10(y_all.iloc[va_idx])

            model.fit(X_all.iloc[tr_idx], y_tr)

            y_pred_log = model.predict(X_all.iloc[va_idx])
            r2s.append(r2_score(y_va, y_pred_log))

        mean_r2 = float(np.mean(r2s))
        loss = 1.0 - mean_r2
        return loss

    return objective


def tune_lookahead_model(budget: int, horizon: int, n_trials: int = 100, seed: int = 42):
    """
    Tunes and trains ONE model for the given budget and horizon (t+1 .. t+19).
    Saves model to output_models_dir.
    """
    if horizon > (1000 - budget) // 50:
        print(f"Skipping B{budget}, t+{horizon}: horizon not available from budget.")
        return None

    ela_path = Path(ela_data_dir) / f"A1_B{budget}_5D_ela.csv"
    df = pd.read_csv(ela_path)

    target = _target_col(horizon)
    if target not in df.columns:
        print(f"Skipping B{budget}, {target}: column not present in {ela_path.name}.")
        return None

    cs = RandomForestRegressorWrapper.get_configuration_space()

    # Keep SMAC output separated per (budget, horizon) to avoid overwrites
    smac_out = Path(smac_output_dir) / f"B{budget}_t{horizon}"
    smac_out.mkdir(parents=True, exist_ok=True)

    scenario = Scenario(
        configspace=cs,
        n_trials=n_trials,
        walltime_limit=np.inf,
        deterministic=True,
        output_directory=str(smac_out),
        seed=seed,
    )

    smac_objective = make_smac_objective_for_budget_and_horizon(budget, horizon)

    smac = HyperparameterOptimizationFacade(scenario, smac_objective)
    best_config = smac.optimize()

    # Train final model on ALL data with best config
    wrapper_partial = RandomForestRegressorWrapper.get_from_configuration(best_config, random_state=seed)
    model = wrapper_partial()

    # Dump untrained, but optimized model
    if not os.path.exists(untrained_output_models_dir):
        os.makedirs(untrained_output_models_dir)

    untrained_model_path = Path(untrained_output_models_dir) / f"lookahead_model_B{budget}_t{horizon}_untrained.pkl"
    joblib.dump(model, untrained_model_path)
    print(f"Saved untrained lookahead model for budget {budget}, t+{horizon} to {untrained_model_path}")

    X_train, y_train, _groups = _make_X_y_groups(df, horizon=horizon)
    y_train_log = np.log10(y_train)
    model.fit(X_train, y_train_log)


    out_dir = Path(output_models_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_path = out_dir / f"lookahead_model_B{budget}_t{horizon}_trained.pkl"
    joblib.dump(model, model_path)

    print(f"Saved lookahead model for budget {budget}, t+{horizon} to {model_path}")
    return model_path


def main():
    if len(sys.argv) != 3:
        print("Usage: python tune_lookahead.py <budget> <horizon(1|2|3)>")
        sys.exit(1)

    budget = int(sys.argv[1])
    horizon = int(sys.argv[2])

    tune_lookahead_model(budget=budget, horizon=horizon)


if __name__ == "__main__":
    main()
