"""Evaluate already-trained standard-BBOB selector models on interpolation data."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd
from tsfresh import extract_features
from tsfresh.utilities.dataframe_functions import impute
from tqdm.auto import tqdm

import selector as selector_module
from data_collection.interpolation_data_collection import default_specs


INTERPOLATION_META_COLS = ["fid_a", "fid_b", "alpha"]
INTERNAL_STATE_CHANNELS = (
    "sigma", "d_norm", "d_mean", "ps_norm", "ps_mean", "pc_norm",
    "pc_mean", "mhl_norm", "mhl_mean", "loglikelihood",
)
# Match the Slurm allocation when available; local runs retain a modest default.
INTERNAL_STATE_N_JOBS = int(os.environ.get("SLURM_CPUS_PER_TASK", "8"))


def internal_fid(fid_a: int, fid_b: int, alpha: float) -> int:
    """Recover the IOH-only ID; it is deliberately not stored in CSV files."""
    lookup = {
        (spec.fid_a, spec.fid_b, round(spec.alpha, 10)): spec.internal_fid
        for spec in default_specs()
    }
    return lookup[(fid_a, fid_b, round(alpha, 10))]


def interpolation_internal_state_cache_path(root: Path, budget: int) -> Path:
    """Return the reusable selected-tsfresh matrix for an interpolation budget."""
    return (
        root.parent / "training_data" / "internal_state_features" / "feature_matrices"
        / f"selected_feature_matrix_B{budget}.csv"
    )


def interpolation_internal_state_features(
    root: Path, budget: int, n_jobs: int
) -> pd.DataFrame:
    """Calculate selected tsfresh features from interpolation trajectories.

    The retained column list comes from the matching normal experiment, so no
    selector is fit on interpolation functions.  The feature values themselves
    are calculated from interpolation CMA-ES trajectories in this evaluator.
    """
    states_path = (
        root / "internal_state"
        / f"internal_state_Non-elitist_B{selector_module.TOTAL_BUDGET}_"
        f"{selector_module.DIM}D.csv"
    )
    states = pd.read_csv(states_path)
    required = {
        "fid_a", "fid_b", "alpha", "iid", "rep", "iteration",
        "evaluations", *INTERNAL_STATE_CHANNELS,
    }
    missing = sorted(required - set(states.columns))
    if missing:
        raise ValueError(f"{states_path} is missing internal-state columns: {missing}")
    states["fid"] = states.apply(
        lambda row: internal_fid(row.fid_a, row.fid_b, row.alpha), axis=1
    )
    states["run_id"] = states[selector_module.KEY_COLS].astype(str).agg("_".join, axis=1)
    states = states.sort_values(["run_id", "evaluations", "iteration"], kind="stable")
    cutoffs = states.groupby("run_id", sort=False)["evaluations"].transform(
        lambda values: values[values >= budget].iloc[0]
        if (values >= budget).any() else values.iloc[-1]
    )
    states = states.loc[states["evaluations"] <= cutoffs].copy()
    run_lengths = states.groupby("run_id", sort=False).size()
    if (run_lengths < 2).any():
        raise ValueError("tsfresh requires at least two state observations per run")

    metadata = states.drop_duplicates("run_id").set_index("run_id")[selector_module.KEY_COLS]
    features = extract_features(
        states[["run_id", "evaluations", *INTERNAL_STATE_CHANNELS]],
        column_id="run_id", column_sort="evaluations", n_jobs=n_jobs,
    )
    impute(features)
    selected_path = (
        root.parent.parent / f"dim_{selector_module.DIM}" / "training_data"
        / "internal_state_features" / "selected_features"
        / f"selected_features_B{budget}.csv"
    )
    selected_names = pd.read_csv(selected_path)["feature"]
    unavailable = selected_names[~selected_names.isin(features.columns)]
    if not unavailable.empty:
        raise ValueError(
            f"{selected_path} contains tsfresh columns absent from interpolation data; "
            f"examples: {unavailable.iloc[:5].tolist()}"
        )
    selected = features.loc[:, selected_names].copy()
    metadata = metadata.loc[features.index]
    for column in reversed(selector_module.KEY_COLS):
        selected.insert(0, column, metadata[column].to_numpy())
    return selected


def prepare_interpolation_internal_state_features(
    dimension: int, data_path: str = "data", budget: int | None = None,
) -> list[Path]:
    """Extract and persist the interpolation tsfresh matrices needed by evaluation.

    These matrices are independent of the metric, feature study, and lookahead
    count. Preparing them once avoids every lookahead evaluation repeating the
    same tsfresh work.
    """
    selector_module.configure_experiment(dimension, "regret")
    root = Path(data_path) / f"dim_{dimension}_interpolation" / "run_data"
    budgets = selector_module.TRAINING_SWITCHING_BUDGETS
    if budget is not None:
        if budget not in budgets:
            raise ValueError(
                f"Budget {budget} is not a training switching budget: {budgets}"
            )
        budgets = [budget]

    cache_paths = []
    for feature_budget in budgets:
        cache_path = interpolation_internal_state_cache_path(root, feature_budget)
        if cache_path.exists():
            print(f"Using existing interpolation internal-state cache: {cache_path}")
            cache_paths.append(cache_path)
            continue

        print(f"Extracting interpolation internal-state features for budget {feature_budget}...")
        features = interpolation_internal_state_features(
            root, feature_budget, INTERNAL_STATE_N_JOBS
        )
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = cache_path.with_suffix(".csv.tmp")
        features.to_csv(temporary_path, index=False)
        temporary_path.replace(cache_path)
        cache_paths.append(cache_path)
    return cache_paths


def load_cached_interpolation_internal_state_features(
    root: Path, budget: int,
) -> pd.DataFrame:
    """Load a prepared matrix, refusing to perform tsfresh during evaluation."""
    cache_path = interpolation_internal_state_cache_path(root, budget)
    if not cache_path.exists():
        raise FileNotFoundError(
            f"Interpolation internal-state cache is missing: {cache_path}. "
            "Run interpolation_evaluate with --prepare-internal-state-features first."
        )
    return pd.read_csv(cache_path)


def load_interpolation_predictor_features(
    root: Path, feature_study: str,
) -> dict[int, pd.DataFrame]:
    """Load interpolation predictors in the schema expected by normal models."""
    feature_study = selector_module.validate_feature_study(feature_study)
    sources: dict[int, list[pd.DataFrame]] = {
        budget: [] for budget in selector_module.TRAINING_SWITCHING_BUDGETS
    }

    if feature_study in {"ela", "ela_internal_state"}:
        ela = pd.read_csv(
            root / "ela_features"
            / f"Non-elitist_B{selector_module.TOTAL_BUDGET}_{selector_module.DIM}D"
            / "ELA_features.csv"
        )
        ela["fid"] = ela.apply(
            lambda row: internal_fid(row.fid_a, row.fid_b, row.alpha), axis=1
        )
        for budget in sources:
            table = ela.loc[ela["ela_budget"] == budget].copy()
            table = table.drop(
                columns=[
                    "a1_budget", "ela_budget", "a2_algorithm", "high_level_category",
                    *INTERPOLATION_META_COLS,
                ],
                errors="ignore",
            )
            sources[budget].append(selector_module.drop_all_nan_ela_columns(table))

    if feature_study in {"internal_state", "ela_internal_state"}:
        for budget in sources:
            internal_state = load_cached_interpolation_internal_state_features(root, budget)
            internal_state = internal_state.drop(columns=["budget"], errors="ignore")
            internal_state = internal_state.rename(
                columns={
                    column: f"is__{column}"
                    for column in internal_state.columns
                    if column not in selector_module.KEY_COLS
                }
            )
            sources[budget].append(internal_state)

    return {
        budget: tables[0]
        if len(tables) == 1
        else tables[0].merge(tables[1], on=selector_module.KEY_COLS, how="inner", validate="one_to_one")
        for budget, tables in sources.items()
    }


def evaluate_interpolations(dimension: int = 40, metric: str = "regret",
                            lookahead_count: int = 0, data_path: str = "data",
                            model_path: str = "models", results_path: str = "results",
                            feature_study: str = "ela") -> Path:
    selector_module.configure_experiment(dimension, metric)
    feature_study = selector_module.validate_feature_study(feature_study)
    dynamic_selector = selector_module.DynamicSelector(
        load_models=True, data_path=data_path, model_path=model_path,
        feature_study=feature_study, lookahead_count=lookahead_count,
    )
    root = Path(data_path) / f"dim_{dimension}_interpolation" / "run_data"
    budgets = dynamic_selector.switching_budgets
    metric_name = f"achieved_{metric}"
    metrics = pd.concat([
        pd.read_csv(root / f"achieved_{metric}s" / f"achieved_{metric}s_{algorithm}_B{budget}_{dimension}D.csv")
        for algorithm in selector_module.A2_ALGORITHMS
        for budget in budgets if budget != selector_module.TOTAL_BUDGET
    ] + [pd.read_csv(root / f"achieved_{metric}s" / f"achieved_{metric}s_Non-elitist_B{selector_module.TOTAL_BUDGET}_{dimension}D.csv")], ignore_index=True)
    # ``fid`` is an IOH implementation detail, not part of the interpolation
    # dataset schema.  Recreate it only for the selector's legacy lookup API.
    metrics["fid"] = metrics.apply(
        lambda row: internal_fid(row.fid_a, row.fid_b, row.alpha), axis=1
    )
    metrics = selector_module.metric_lookup(metrics)
    feature_tables = {
        budget: table.set_index(selector_module.KEY_COLS, drop=False, verify_integrity=True)
        for budget, table in load_interpolation_predictor_features(root, feature_study).items()
    }
    output = (Path(results_path) / f"dim_{dimension}_interpolation" /
              metric / feature_study / f"lookahead_{lookahead_count}" /
              "selector_results.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    metadata_table = next(iter(feature_tables.values()))
    specs = {spec.internal_fid: spec for spec in default_specs()}
    run_keys = metadata_table.index.unique()
    for fid, iid, rep in tqdm(
        run_keys, total=len(run_keys), desc="Interpolation selector evaluation",
        mininterval=5, file=sys.stdout,
    ):
        spec = specs[fid]
        metadata = {"fid_a": spec.fid_a, "fid_b": spec.fid_b, "alpha": spec.alpha}
        result = dynamic_selector.simulate_single_run(
            fid, iid, rep, feature_tables, metrics, lookahead_count,
            clip_to_training_range=True,
        )
        for budget in budgets:
            if budget == selector_module.TOTAL_BUDGET:
                continue
            scaled = dynamic_selector._scaled_feature_row(
                budget, feature_tables[budget], fid, iid, rep,
                clip_to_training_range=True,
            )
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
    parser.add_argument("--lookahead-count", type=int)
    parser.add_argument("--data-path", default="data")
    parser.add_argument("--model-path", default="models")
    parser.add_argument("--results-path", default="results")
    parser.add_argument(
        "--feature-study", choices=sorted(selector_module.FEATURE_STUDIES), default="ela",
        help="Feature study whose normal dynamic-selector models to load.",
    )
    parser.add_argument(
        "--prepare-internal-state-features", action="store_true",
        help="Extract and cache interpolation internal-state matrices, then exit.",
    )
    parser.add_argument(
        "--internal-state-budget", type=int,
        help=("Prepare only this internal-state feature budget. Requires "
              "--prepare-internal-state-features."),
    )
    args = parser.parse_args()
    if args.prepare_internal_state_features:
        cache_paths = prepare_interpolation_internal_state_features(
            args.dimension, args.data_path, args.internal_state_budget,
        )
        print(f"Prepared {len(cache_paths)} interpolation internal-state feature matrices.")
        return
    if args.lookahead_count is None:
        parser.error("--lookahead-count is required unless preparing internal-state features")
    arguments = vars(args)
    if args.internal_state_budget is not None:
        parser.error("--internal-state-budget requires --prepare-internal-state-features")
    for name in ("prepare_internal_state_features", "internal_state_budget"):
        arguments.pop(name)
    print(evaluate_interpolations(**arguments))


if __name__ == "__main__":
    main()
