"""Evaluate already-trained standard-BBOB selector models on interpolation data."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

import selector as selector_module
from interpolation_data_collection import default_specs


INTERPOLATION_META_COLS = ["fid_a", "fid_b", "alpha"]


def internal_fid(fid_a: int, fid_b: int, alpha: float) -> int:
    """Recover the IOH-only ID; it is deliberately not stored in CSV files."""
    lookup = {
        (spec.fid_a, spec.fid_b, round(spec.alpha, 10)): spec.internal_fid
        for spec in default_specs()
    }
    return lookup[(fid_a, fid_b, round(alpha, 10))]


def evaluate_interpolations(dimension: int = 40, metric: str = "regret",
                            lookahead_count: int = 0, data_path: str = "data",
                            model_path: str = "models", results_path: str = "results") -> Path:
    selector_module.configure_experiment(dimension, metric)
    dynamic_selector = selector_module.DynamicSelector(
        load_models=True, data_path=data_path, model_path=model_path,
    )
    root = Path(data_path) / f"dim_{dimension}_interpolation"
    budgets = dynamic_selector.switching_budgets
    metric_name = f"achieved_{metric}"
    metrics = pd.concat([
        pd.read_csv(root / f"achieved_{metric}s" / f"achieved_{metric}s_{algorithm}_B{budget}_{dimension}D.csv")
        for algorithm in selector_module.SWITCH_ALGORITHMS
        for budget in budgets if budget != selector_module.TOTAL_BUDGET
    ] + [pd.read_csv(root / f"achieved_{metric}s" / f"achieved_{metric}s_Non-elitist_B{selector_module.TOTAL_BUDGET}_{dimension}D.csv")], ignore_index=True)
    # ``fid`` is an IOH implementation detail, not part of the interpolation
    # dataset schema.  Recreate it only for the selector's legacy lookup API.
    metrics["fid"] = metrics.apply(
        lambda row: internal_fid(row.fid_a, row.fid_b, row.alpha), axis=1
    )
    ela = pd.read_csv(root / "ela_features" / f"Non-elitist_B{selector_module.TOTAL_BUDGET}_{dimension}D" / "ELA_features.csv")
    output = Path(results_path) / f"dim_{dimension}_interpolation_models_default" / metric / f"lookahead_{lookahead_count}" / "selector_results.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for (fid_a, fid_b, alpha, iid, rep), group in ela.groupby([*INTERPOLATION_META_COLS, "iid", "rep"], sort=True):
        fid = internal_fid(fid_a, fid_b, alpha)
        metadata = group.iloc[0][INTERPOLATION_META_COLS].to_dict()
        # Match the training-time feature frame: interpolation and run
        # identifiers select a trajectory, but are not ELA model inputs.
        features = group.drop(
            columns=[
                "a1_budget", "a2_algorithm", *INTERPOLATION_META_COLS,
                "iid", "rep",
            ],
            errors="ignore",
        )
        result = dynamic_selector.simulate_single_run(fid, iid, rep, features, metrics, lookahead_count)
        for budget in budgets:
            if budget == selector_module.TOTAL_BUDGET:
                continue
            ela_row = features[features["ela_budget"] == budget].drop(columns=["ela_budget"])
            ela_row = selector_module.drop_all_nan_ela_columns(ela_row)
            scaled = pd.DataFrame(dynamic_selector.models[budget]["ela_scaler"].transform(ela_row), columns=ela_row.columns, index=[(fid, iid, rep)])
            prediction = dynamic_selector.models[budget]["selection_model"].predict(scaled)
            algorithm = list(prediction.values())[0][0][0]
            result[f"static_B{budget}"] = selector_module.get_metric_value(metrics, fid, iid, rep, algorithm, budget)
        result["no_switch"] = selector_module.get_metric_value(metrics, fid, iid, rep, selector_module.NO_SWITCH_ALGORITHM, selector_module.TOTAL_BUDGET)
        result.pop("fid")
        result.update(metadata)
        rows.append(result)
    results = pd.DataFrame(rows)
    front_columns = [*INTERPOLATION_META_COLS, "iid", "rep"]
    results = results[front_columns + [
        column for column in results.columns if column not in front_columns
    ]]
    results.to_csv(output, index=False)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dimension", type=int, default=40)
    parser.add_argument("--metric", choices=["regret", "auc"], default="regret")
    parser.add_argument("--lookahead-count", type=int, required=True)
    parser.add_argument("--data-path", default="data")
    parser.add_argument("--model-path", default="models")
    parser.add_argument("--results-path", default="results")
    args = parser.parse_args()
    print(evaluate_interpolations(**vars(args)))


if __name__ == "__main__":
    main()
