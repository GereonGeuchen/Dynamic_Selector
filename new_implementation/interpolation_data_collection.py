"""Collect selector-test data on two-function BBOB affine interpolations.

This is intentionally separate from :mod:`data_collection`: it reuses that
module's stable optimisation and ELA routines but writes exclusively below
``data/dim_<D>_interpolation``.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, fields
from pathlib import Path

import ioh
import numpy as np
import pandas as pd

from data_collection import (
    ALGORITHM_SPECS,
    DEFAULT_ALGORITHMS,
    TrackedParameters,
    TrackedParameters_switchAlgo,
    _record_performance,
    _run_switched_algorithm,
    calculate_ela_features,
    safe_df_to_csv,
)


PAIRS = ((2, 10), (24, 21), (21, 22), (24, 2))
ALPHAS = (0.9, 0.7, 0.5, 0.3, 0.1)
TEST_IIDS = (6, 7)


@dataclass(frozen=True)
class InterpolationSpec:
    fid_a: int
    fid_b: int
    alpha: float
    internal_fid: int

    @property
    def alpha_b(self) -> float:
        return 1.0 - self.alpha

    @property
    def experiment_id(self) -> str:
        return f"F{self.fid_a:02d}_F{self.fid_b:02d}_a{round(100 * self.alpha):03d}"

    def metadata(self) -> dict:
        return {
            "fid_a": self.fid_a,
            "fid_b": self.fid_b,
            "alpha": self.alpha,
        }


def default_specs() -> list[InterpolationSpec]:
    """Return the 20 requested pair/interpolation combinations."""
    return [
        InterpolationSpec(fid_a, fid_b, alpha, internal_fid)
        for internal_fid, (fid_a, fid_b, alpha) in enumerate(
            ((pair[0], pair[1], alpha) for pair in PAIRS for alpha in ALPHAS),
            start=1,
        )
    ]


def pair_weights(spec: InterpolationSpec) -> list[float]:
    weights = [0.0] * 24
    weights[spec.fid_a - 1] = spec.alpha
    weights[spec.fid_b - 1] = spec.alpha_b
    return weights


def create_interpolation_problem(spec: InterpolationSpec, iid: int, dim: int):
    """Create a mixture whose optimum location interpolates the endpoints."""
    endpoint_a = ioh.get_problem(spec.fid_a, iid, dim, ioh.ProblemClass.BBOB)
    endpoint_b = ioh.get_problem(spec.fid_b, iid, dim, ioh.ProblemClass.BBOB)
    xopt = (
        spec.alpha * np.asarray(endpoint_a.optimum.x)
        + spec.alpha_b * np.asarray(endpoint_b.optimum.x)
    )
    problem = ioh.problem.ManyAffine(
        xopt=list(xopt),
        weights=pair_weights(spec),
        instances=[iid] * 24,
        n_variables=dim,
    )
    # A unique ID prevents IOH logger collisions between interpolation levels.
    problem.set_id(spec.internal_fid)
    return problem


class InterpolationProblemWrapper:
    """IOH wrapper compatible with the established collection routines."""

    def __init__(self, problem):
        self.problem = problem
        self.function_evals = {}
        self.best_so_far_evals = {}
        self.best_eval_so_far = np.inf

    def __call__(self, x):
        y = self.problem(x)
        self.best_eval_so_far = min(self.best_eval_so_far, y)
        evaluation = self.problem.state.evaluations
        self.function_evals[evaluation] = (x, y)
        self.best_so_far_evals[evaluation] = (x, self.best_eval_so_far)
        return y

    def reset(self):
        self.problem.reset()
        self.function_evals = {}
        self.best_so_far_evals = {}
        self.best_eval_so_far = np.inf

    def __getattr__(self, name):
        return getattr(self.problem, name)


def _metrics_frame(values: dict, column: str, specs_by_fid: dict[int, InterpolationSpec]):
    rows = []
    for (internal_fid, iid, rep, a1_budget, algname), value in values.items():
        row = {
            "iid": iid, "rep": rep, "a1_budget": a1_budget,
            "algname": algname, column: value,
        }
        row.update(specs_by_fid[internal_fid].metadata())
        rows.append(row)
    return pd.DataFrame(rows)


def selected_specs(interpolation_ids=None) -> list[InterpolationSpec]:
    """Select fixed mixtures by their stable, one-based interpolation IDs."""
    all_specs = default_specs()
    if interpolation_ids is None:
        return all_specs
    ids = list(interpolation_ids)
    if not ids:
        raise ValueError("interpolation_ids must not be empty.")
    if any(not 1 <= index <= len(all_specs) for index in ids):
        raise ValueError(f"interpolation_ids must be between 1 and {len(all_specs)}.")
    return [all_specs[index - 1] for index in ids]


def collect_interpolations(a1_budget: int, dim: int, algs_to_run=None,
                           iids=TEST_IIDS, repetitions: int = 20,
                           total_budget: int = 1000, output_suffix: str = "",
                           interpolation_ids=None) -> Path:
    """Run selected fixed interpolations, normally IIDs 6 and 7."""
    if not 0 < a1_budget <= total_budget:
        raise ValueError("a1_budget must be in (0, total_budget].")
    specs = selected_specs(interpolation_ids)
    if len({spec.internal_fid for spec in specs}) != len(specs):
        raise ValueError("Each interpolation spec needs a unique internal_fid.")
    algorithms = DEFAULT_ALGORITHMS if algs_to_run is None else tuple(algs_to_run)
    root = Path("data") / f"dim_{dim}_interpolation"
    specs_by_fid = {spec.internal_fid: spec for spec in specs}
    trigger = ioh.logger.trigger.OnImprovement()

    for algorithm_class, algname in ALGORITHM_SPECS:
        if algname not in algorithms:
            continue
        if algname == "Non-elitist" and a1_budget != total_budget:
            continue
        if algname != "Non-elitist" and a1_budget == total_budget:
            continue

        run_tag = f"B{a1_budget}"
        regrets, aucs = {}, {}
        ela_rows = []
        ela_folder = root / "ela_features" / f"{algname}_{run_tag}_{dim}D"
        ela_filename = f"ELA_features{output_suffix}.csv"
        ela_path = ela_folder / ela_filename
        if algname == "Non-elitist" and ela_path.exists():
            os.remove(ela_path)

        for spec in specs:
            logger = ioh.logger.Analyzer(
                triggers=[trigger],
                folder_name=str(root / "raw_evaluations" / f"{algname}_{run_tag}_{dim}D" / spec.experiment_id),
                algorithm_name=algname,
                store_positions=True,
            )
            tracked = (TrackedParameters() if algname == "Non-elitist"
                       else TrackedParameters_switchAlgo())
            logger.watch(tracked, [field.name for field in fields(tracked)])
            for iid in iids:
                problem = InterpolationProblemWrapper(
                    create_interpolation_problem(spec, iid, dim)
                )
                problem.attach_logger(logger)
                for rep in range(repetitions):
                    tracked.rep, tracked.iid = rep, iid
                    print(f"Running {spec.experiment_id}: iid {iid}, rep {rep}, {algname}, B{a1_budget}")
                    np.random.seed(rep)
                    _run_switched_algorithm(problem, algorithm_class, algname, a1_budget, dim, total_budget, tracked)
                    if algname == "Non-elitist":
                        for budget in range(50, total_budget + 1, 50):
                            evaluations = {key: value for key, value in problem.function_evals.items() if key <= budget}
                            row = calculate_ela_features(evaluations, spec.internal_fid, iid, rep, a1_budget, dim, algname)
                            row.pop("fid")
                            row.pop("high_level_category")
                            row.update(spec.metadata())
                            ela_rows.append(row)
                    _record_performance(problem, spec.internal_fid, iid, rep, a1_budget, algname, total_budget, regrets, aucs)
                    problem.reset()
                problem.detach_logger()

        if ela_rows:
            frame = pd.DataFrame(ela_rows)
            if dim == 40:
                frame = frame.drop(columns=["ela_meta.quad_simple.cond"], errors="ignore")
            safe_df_to_csv(str(ela_folder), ela_filename, frame, append=True)
        for values, metric in ((regrets, "regret"), (aucs, "auc")):
            frame = _metrics_frame(values, f"achieved_{metric}", specs_by_fid)
            safe_df_to_csv(str(root / f"achieved_{metric}s"), f"achieved_{metric}s_{algname}_{run_tag}_{dim}D{output_suffix}.csv", frame)
    return root


def merge_parts(dim: int, a1_budget: int, algorithms: list[str], suffixes: list[str]) -> Path:
    """Merge parallel-job CSV parts into the filenames consumed by the evaluator.

    ``suffixes`` must name every parallel part exactly once (for example,
    ``.part-0,.part-1,...``).  This operation only creates the unsuffixed
    combined files; individual part files are retained.
    """
    if not suffixes or any(not suffix for suffix in suffixes):
        raise ValueError("Provide one non-empty suffix for every part to merge.")
    root = Path("data") / f"dim_{dim}_interpolation"
    run_tag = f"B{a1_budget}"
    for algorithm in algorithms:
        paths = []
        for suffix in suffixes:
            if algorithm == "Non-elitist":
                path = root / "ela_features" / f"{algorithm}_{run_tag}_{dim}D" / f"ELA_features{suffix}.csv"
                paths.append(path)
            for metric in ("regret", "auc"):
                path = root / f"achieved_{metric}s" / f"achieved_{metric}s_{algorithm}_{run_tag}_{dim}D{suffix}.csv"
                paths.append(path)
        for part_path in paths:
            if not part_path.exists():
                raise FileNotFoundError(f"Missing parallel part: {part_path}")
        grouped = {}
        for path in paths:
            target = path.with_name(path.name.replace(next(s for s in suffixes if path.name.endswith(f"{s}.csv")), ""))
            grouped.setdefault(target, []).append(path)
        for target, part_paths in grouped.items():
            merged = pd.concat([pd.read_csv(path) for path in part_paths], ignore_index=True)
            if target.name == "ELA_features.csv":
                # Keep interpolation identifiers and run identifiers together
                # at the start of the schema, independent of their insertion
                # order while individual ELA rows were assembled.
                front_columns = ["fid_a", "fid_b", "alpha", "iid", "rep"]
                missing_columns = set(front_columns).difference(merged.columns)
                if missing_columns:
                    raise ValueError(
                        f"ELA part files are missing required columns: "
                        f"{', '.join(sorted(missing_columns))}"
                    )
                merged = merged[front_columns + [
                    column for column in merged.columns if column not in front_columns
                ]]
            merged.to_csv(target, index=False)
    return root


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("a1_budget", type=int, nargs="?", help="A1 switch budget to collect or merge.")
    parser.add_argument("--dimension", type=int, default=40)
    parser.add_argument("--algorithms", default=",".join(DEFAULT_ALGORITHMS))
    parser.add_argument("--repetitions", type=int, default=20)
    parser.add_argument("--total-budget", type=int, default=1000)
    parser.add_argument("--suffix", default="")
    parser.add_argument(
        "--interpolation-ids", default="1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20",
        help="Comma-separated one-based IDs from the fixed 20-mixture grid.",
    )
    parser.add_argument("--merge-suffixes", help="Comma-separated part suffixes to merge instead of collecting.")
    args = parser.parse_args()
    if args.a1_budget is None:
        parser.error("a1_budget is required")
    if args.merge_suffixes:
        print(merge_parts(args.dimension, args.a1_budget, args.algorithms.split(","), args.merge_suffixes.split(",")))
        return
    collect_interpolations(args.a1_budget, args.dimension, args.algorithms.split(","),
                           repetitions=args.repetitions, total_budget=args.total_budget,
                           output_suffix=args.suffix,
                           interpolation_ids=[int(index) for index in args.interpolation_ids.split(",")])


if __name__ == "__main__":
    main()
