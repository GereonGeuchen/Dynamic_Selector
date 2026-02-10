import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from asf.predictors import RandomForestRegressorWrapper
from smac import HyperparameterOptimizationFacade, Scenario

from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score


# === Parameters (edit these) ===
ela_csv_path = "../data/ela_lhs_with_precisions.csv"  # ELA + algorithm precision columns
smac_output_dir = "smac_outputs/smac_output_lhs_r2"
output_models_dir = "../data/models/tuned_models/per_instance_selector_models_trained"
untrained_output_models_dir = "../data/models/untrained_models/per_instance_selector_models_untrained"

ALGO_COLS = ["BFGS", "DE", "Elitist", "MLSL", "Non-elitist", "PSO"]
LOG_EPS = 1e-12


def _make_X_y_groups_for_algorithm(df: pd.DataFrame, algo: str, group_by: str = "iid"):
    """
    X: ELA features only (drops fid/iid/high_level_category + all algorithm columns)
    y: continuous precision for the given algo
    groups: df[group_by] for GroupKFold
    """
    if algo not in df.columns:
        raise ValueError(f"Algorithm column '{algo}' not found in dataframe.")
    if group_by not in df.columns:
        raise ValueError(f"Expected grouping column '{group_by}', but it was not found.")

    groups = df[group_by]
    y = df[algo].astype(float)

    # Drop identifiers and ALL algorithm targets from X
    drop_cols = ["fid", "iid", "high_level_category"] + [c for c in ALGO_COLS if c in df.columns]
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])

    return X, y, groups


def make_smac_objective_for_algorithm(
    algo: str,
    group_by: str = "iid",
    use_log_target: bool = True,
):
    """
    Returns SMAC objective(config, seed) -> loss.
    Uses GroupKFold and optimizes 1 - mean_R2 (SMAC minimizes).
    """
    df = pd.read_csv(ela_csv_path)
    X_all, y_all, groups = _make_X_y_groups_for_algorithm(df, algo, group_by=group_by)

    n_groups = groups.nunique()
    if n_groups < 2:
        raise ValueError(f"Need at least 2 groups for GroupKFold, got {n_groups}.")

    cv = GroupKFold(n_splits=min(5, n_groups))

    def objective(config, seed: int = 42):
        wrapper_partial = RandomForestRegressorWrapper.get_from_configuration(config, random_state=seed)

        r2s = []
        for tr_idx, va_idx in cv.split(X_all, y_all, groups):
            model = wrapper_partial()

            y_tr = y_all.iloc[tr_idx].to_numpy(dtype=float)
            y_va = y_all.iloc[va_idx].to_numpy(dtype=float)

            # Optionally evaluate in log-space for stability
            if use_log_target:
                y_tr_fit = np.log10(np.maximum(y_tr, LOG_EPS))
                y_va_eval = np.log10(np.maximum(y_va, LOG_EPS))
            else:
                y_tr_fit = y_tr
                y_va_eval = y_va

            model.fit(X_all.iloc[tr_idx], y_tr_fit)
            y_pred = model.predict(X_all.iloc[va_idx])

            r2s.append(r2_score(y_va_eval, y_pred))

        mean_r2 = float(np.mean(r2s))
        return 1.0 - mean_r2

    return objective


def tune_and_train_model_for_algorithm(
    algo: str,
    n_trials: int = 100,
    seed: int = 42,
    group_by: str = "iid",
    use_log_target: bool = True,
):
    """
    Tunes via SMAC for ONE algorithm regressor (R2 objective),
    then trains on ALL data and saves untrained+trained models.
    """
    df = pd.read_csv(ela_csv_path)
    X_all, y_all, _groups = _make_X_y_groups_for_algorithm(df, algo, group_by=group_by)

    cs = RandomForestRegressorWrapper.get_configuration_space()

    smac_out = Path(smac_output_dir) / f"{algo}"
    smac_out.mkdir(parents=True, exist_ok=True)

    scenario = Scenario(
        configspace=cs,
        n_trials=n_trials,
        walltime_limit=np.inf,
        deterministic=True,
        output_directory=str(smac_out),
        seed=seed,
    )

    smac_objective = make_smac_objective_for_algorithm(
        algo,
        group_by=group_by,
        use_log_target=use_log_target,
    )

    smac = HyperparameterOptimizationFacade(scenario, smac_objective)
    best_config = smac.optimize()

    # Instantiate model with best config
    wrapper_partial = RandomForestRegressorWrapper.get_from_configuration(best_config, random_state=seed)
    model = wrapper_partial()

    # Save optimized but untrained model
    Path(untrained_output_models_dir).mkdir(parents=True, exist_ok=True)
    untrained_path = Path(untrained_output_models_dir) / f"{algo}_rf_untrained.pkl"
    joblib.dump(model, untrained_path)
    print(f"[{algo}] Saved untrained model to {untrained_path}")

    # Train on ALL data
    y_train = y_all.to_numpy(dtype=float)
    if use_log_target:
        y_train = np.log10(np.maximum(y_train, LOG_EPS))

    model.fit(X_all, y_train)

    # Save trained model
    Path(output_models_dir).mkdir(parents=True, exist_ok=True)
    trained_path = Path(output_models_dir) / f"{algo}_rf_trained.pkl"
    joblib.dump(model, trained_path)
    print(f"[{algo}] Saved trained model to {trained_path}")

    return trained_path

if __name__ == "__main__":
    for a in ALGO_COLS:
        tune_and_train_model_for_algorithm(
            a, n_trials=100, seed=42, group_by="iid", use_log_target=True
        )

