"""Train and evaluate six-choice static algorithm selectors.

At each candidate budget, the model selects among the five switch destinations
and the no-switch ``Non-elitist`` baseline.  Unlike :mod:`selector`, it never
trains a separate switching-time model.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tqdm.auto import tqdm

import interpolation_evaluate
import selector


ALGORITHMS = [*selector.A2_ALGORITHMS, selector.NO_SWITCH_ALGORITHM]


def _artifact_root(base_path: str, feature_study: str) -> Path:
    """Return an isolated path so these models cannot overwrite dynamic ones."""
    return Path(
        selector.study_scoped_path(
            selector.metric_scoped_path(
                selector.dimension_scoped_path(base_path)
            ),
            feature_study,
        )
    ) / "static_selection_models"


def _feature_columns(data: pd.DataFrame) -> list[str]:
    return [column for column in data.columns if column not in selector.META_COLS + ALGORITHMS]


def _normalise(data: pd.DataFrame) -> tuple[pd.DataFrame, MinMaxScaler]:
    """Match selector.py's per-function target and global feature scaling."""
    normalised = data.copy()
    for _, group in normalised.groupby("fid"):
        index = group.index
        values = normalised.loc[index, ALGORITHMS].to_numpy().reshape(-1, 1)
        normalised.loc[index, ALGORITHMS] = MinMaxScaler(
            feature_range=(1e-12, 1)
        ).fit_transform(values).reshape(len(index), len(ALGORITHMS))
    scaler = MinMaxScaler()
    features = _feature_columns(normalised)
    normalised[features] = scaler.fit_transform(normalised[features])
    return normalised, scaler


def create_training_data(run_data_path: str, budget: int, feature_study: str) -> tuple[pd.DataFrame, MinMaxScaler]:
    """Build one budget's training table with all six candidate performances."""
    switched = [
        pd.read_csv(selector.achieved_metric_path(run_data_path, algorithm, budget))
        for algorithm in selector.A2_ALGORITHMS
    ]
    no_switch = pd.read_csv(selector.achieved_metric_path(
        run_data_path, selector.NO_SWITCH_ALGORITHM, selector.TOTAL_BUDGET
    ))
    metrics = pd.concat([*switched, no_switch], ignore_index=True)
    metric_wide = (
        metrics.pivot(index=selector.KEY_COLS, columns="algname", values=selector.METRIC_COLUMN)
        .rename_axis(columns=None).reset_index()
    )
    missing = set(ALGORITHMS).difference(metric_wide.columns)
    if missing:
        raise ValueError(f"Missing candidate metrics at budget {budget}: {sorted(missing)}")
    features = selector.load_predictor_features(run_data_path, budget, feature_study)
    data = features.merge(metric_wide, on=selector.KEY_COLS, how="inner", validate="one_to_one")
    data = data[data["iid"].isin(selector.TRAIN_IIDS)].dropna(axis=1, how="all")
    return _normalise(data)


class StaticSelectionModel:
    def __init__(self, data_path: str = "data", model_path: str = "models", results_path: str = "results",
                 feature_study: str = "ela", interpolation: bool = False):
        self.feature_study = selector.validate_feature_study(feature_study)
        self.interpolation = interpolation
        if interpolation:
            self.run_data_path = str(
                Path(data_path) / f"dim_{selector.DIM}_interpolation" / "run_data"
            )
        else:
            self.run_data_path = selector.run_data_path(selector.dimension_scoped_path(data_path))
        self.model_root = _artifact_root(model_path, self.feature_study)
        result_dimension = (
            f"dim_{selector.DIM}_interpolation"
            if interpolation else selector.dimension_directory_name()
        )
        self.results_root = (
            Path(results_path) / result_dimension / "static"
            / selector.METRIC / self.feature_study
        )
        self.models: dict[int, dict] = {}

    def train(self) -> None:
        for budget in selector.TRAINING_SWITCHING_BUDGETS:
            data, scaler = create_training_data(self.run_data_path, budget, self.feature_study)
            features = _feature_columns(data)
            model = selector.make_default_performance_model()
            model.algorithms = ALGORITHMS
            model.fit(data[features], data[ALGORITHMS])
            folder = self.model_root / f"budget_{budget}"
            folder.mkdir(parents=True, exist_ok=True)
            joblib.dump(model, folder / "selection_model.joblib")
            joblib.dump(scaler, folder / "feature_scaler.joblib")
            joblib.dump(features, folder / "feature_columns.joblib")
            print(f"Trained six-choice static selector for B{budget}.")

    def load(self) -> None:
        for budget in selector.TRAINING_SWITCHING_BUDGETS:
            folder = self.model_root / f"budget_{budget}"
            paths = {name: folder / filename for name, filename in {
                "model": "selection_model.joblib", "scaler": "feature_scaler.joblib",
                "features": "feature_columns.joblib",
            }.items()}
            missing = [str(path) for path in paths.values() if not path.exists()]
            if missing:
                raise FileNotFoundError("Missing static-selector artifacts: " + ", ".join(missing))
            self.models[budget] = {name: joblib.load(path) for name, path in paths.items()}

    def evaluate(self) -> Path:
        if not self.models:
            self.load()
        if self.interpolation:
            return self._evaluate_interpolations()
        metrics = pd.concat([
            *(pd.read_csv(selector.achieved_metric_path(self.run_data_path, algorithm, budget))
              for budget in selector.TRAINING_SWITCHING_BUDGETS for algorithm in selector.A2_ALGORITHMS),
            pd.read_csv(selector.achieved_metric_path(
                self.run_data_path, selector.NO_SWITCH_ALGORITHM, selector.TOTAL_BUDGET
            )),
        ], ignore_index=True)
        rows = []
        for budget in selector.TRAINING_SWITCHING_BUDGETS:
            table = selector.load_predictor_features(self.run_data_path, budget, self.feature_study)
            table = table[table["iid"].isin(selector.TEST_IIDS)]
            artifacts = self.models[budget]
            feature_columns = artifacts["features"]
            evaluation_features = selector.clip_features_to_training_range(
                table[feature_columns], artifacts["scaler"],
            )
            scaled = artifacts["scaler"].transform(evaluation_features)
            prediction = artifacts["model"].predict(pd.DataFrame(scaled, columns=feature_columns, index=table.index))
            selected = [value[0][0] for value in prediction.values()]
            for (_, row), algorithm in zip(table.iterrows(), selected):
                fid, iid, rep = (int(row[key]) for key in selector.KEY_COLS)
                achieved = selector.get_metric_value(metrics, fid, iid, rep, algorithm, budget)
                candidates = [selector.get_metric_value(metrics, fid, iid, rep, candidate, budget)
                              for candidate in ALGORITHMS]
                rows.append({
                    "fid": fid, "iid": iid, "rep": rep, "budget": budget,
                    "selected_algorithm": algorithm, selector.METRIC_COLUMN: achieved,
                    selector.VBS_METRIC_COLUMN: min(candidates),
                })
        output = self.results_root / "selector_results.csv"
        output.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(output, index=False)
        print(f"Saved static-selector results to {output}")
        return output

    def _evaluate_interpolations(self) -> Path:
        """Evaluate standard-data static models on every interpolation run."""
        root = Path(self.run_data_path)
        budgets = selector.TRAINING_SWITCHING_BUDGETS
        metrics = pd.concat([
            pd.read_csv(
                root / f"achieved_{selector.METRIC}s" /
                f"achieved_{selector.METRIC}s_{algorithm}_B{budget}_{selector.DIM}D.csv"
            )
            for algorithm in selector.A2_ALGORITHMS
            for budget in budgets
        ] + [pd.read_csv(
            root / f"achieved_{selector.METRIC}s" /
            f"achieved_{selector.METRIC}s_{selector.NO_SWITCH_ALGORITHM}_B{selector.TOTAL_BUDGET}_{selector.DIM}D.csv"
        )], ignore_index=True)
        metrics["fid"] = metrics.apply(
            lambda row: interpolation_evaluate.internal_fid(
                row.fid_a, row.fid_b, row.alpha
            ),
            axis=1,
        )
        metrics = selector.metric_lookup(metrics)
        feature_tables = interpolation_evaluate.load_interpolation_predictor_features(
            root, self.feature_study
        )
        specs = {
            spec.internal_fid: spec
            for spec in interpolation_evaluate.default_specs()
        }
        rows = []
        progress = tqdm(
            total=sum(len(table) for table in feature_tables.values()),
            desc="Interpolation static evaluation", mininterval=5, file=sys.stdout,
        )
        for budget in budgets:
            table = feature_tables[budget]
            artifacts = self.models[budget]
            feature_columns = artifacts["features"]
            evaluation_features = selector.clip_features_to_training_range(
                table[feature_columns], artifacts["scaler"],
            )
            scaled = artifacts["scaler"].transform(evaluation_features)
            prediction = artifacts["model"].predict(
                pd.DataFrame(scaled, columns=feature_columns, index=table.index)
            )
            selected = [value[0][0] for value in prediction.values()]
            for (_, row), algorithm in zip(table.iterrows(), selected):
                fid, iid, rep = (int(row[key]) for key in selector.KEY_COLS)
                achieved = selector.get_metric_value(
                    metrics, fid, iid, rep, algorithm, budget
                )
                candidates = [
                    selector.get_metric_value(metrics, fid, iid, rep, candidate, budget)
                    for candidate in ALGORITHMS
                ]
                spec = specs[fid]
                rows.append({
                    "fid_a": spec.fid_a, "fid_b": spec.fid_b, "alpha": spec.alpha,
                    "iid": iid, "rep": rep, "budget": budget,
                    "selected_algorithm": algorithm,
                    selector.METRIC_COLUMN: achieved,
                    selector.VBS_METRIC_COLUMN: min(candidates),
                })
                progress.update()
        progress.close()
        output = self.results_root / "selector_results.csv"
        output.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(output, index=False)
        print(f"Saved interpolation static-selector results to {output}")
        return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["train", "evaluate", "train-evaluate"], required=True)
    parser.add_argument("--dimension", type=int, default=selector.DIM)
    parser.add_argument("--metric", choices=["regret", "auc"], default=selector.METRIC)
    parser.add_argument("--feature-study", choices=sorted(selector.FEATURE_STUDIES), default="ela")
    parser.add_argument("--data-path", default="data")
    parser.add_argument("--model-path", default="models")
    parser.add_argument("--results-path", default="results")
    parser.add_argument(
        "--interpolation", action="store_true",
        help="Evaluate standard-data static models on interpolation data.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selector.configure_experiment(args.dimension, args.metric)
    model = StaticSelectionModel(
        args.data_path, args.model_path, args.results_path, args.feature_study,
        args.interpolation,
    )
    if args.mode in {"train", "train-evaluate"}:
        if args.interpolation:
            raise ValueError("--interpolation supports evaluation only; train on standard data first")
        model.train()
    if args.mode in {"evaluate", "train-evaluate"}:
        model.evaluate()


if __name__ == "__main__":
    main()
