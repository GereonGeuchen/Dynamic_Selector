"""Create function-wise boxplots from DynamicSelector evaluation results.

The script compares the selector with the virtual best solver (VBS) and the
no-switch baseline separately for every BBOB function.  It can also include
static selection models for selected switching budgets.
"""

import argparse
from math import ceil
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import pandas as pd


METHOD_LABELS = {
    "vbs": "VBS",
    "selector": "Selector",
    "no_switch": "No switch",
}
METHOD_COLOURS = {
    "vbs": "#4C78A8",
    "selector": "#F58518",
    "no_switch": "#54A24B",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create function-wise selector-performance boxplots."
    )
    parser.add_argument("--metric", choices=("regret", "auc"), default="regret")
    parser.add_argument("--lookahead-count", type=int, default=0)
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Evaluation CSV. Defaults to results/<metric>/lookahead_<n>/selector_results.csv.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output image. Defaults beside the input CSV as function_wise_boxplots.png.",
    )
    parser.add_argument(
        "--static-budgets",
        type=int,
        nargs="*",
        default=[],
        metavar="BUDGET",
        help="Also plot static selection models, e.g. --static-budgets 50 300 650.",
    )
    parser.add_argument("--columns", type=int, default=4, help="Number of subplot columns.")
    return parser.parse_args()


def metric_columns(metric: str, static_budgets: list[int]) -> tuple[list[str], dict[str, str]]:
    columns = [f"vbs_{metric}", f"achieved_{metric}", "no_switch"]
    labels = {
        f"vbs_{metric}": METHOD_LABELS["vbs"],
        f"achieved_{metric}": METHOD_LABELS["selector"],
        "no_switch": METHOD_LABELS["no_switch"],
    }
    for budget in static_budgets:
        column = f"static_B{budget}"
        columns.append(column)
        labels[column] = f"Static B{budget}"
    return columns, labels


def plot_function_wise_boxplots(
    results: pd.DataFrame,
    metric: str,
    output_path: Path,
    static_budgets: list[int],
    columns: int,
) -> None:
    """Save one boxplot panel per function, with one box per comparison method."""
    value_columns, labels = metric_columns(metric, static_budgets)
    missing = [column for column in value_columns if column not in results.columns]
    if missing:
        raise ValueError(f"The input CSV is missing required columns: {', '.join(missing)}")
    if columns < 1:
        raise ValueError("--columns must be at least 1.")

    functions = sorted(results["fid"].unique())
    if not functions:
        raise ValueError("The input CSV does not contain any function IDs.")

    rows = ceil(len(functions) / columns)
    figure, axes = plt.subplots(rows, columns, figsize=(4.4 * columns, 3.5 * rows), squeeze=False)
    colours = [METHOD_COLOURS.get(column, "#B279A2") for column in value_columns]

    for axis, fid in zip(axes.flat, functions):
        function_results = results.loc[results["fid"] == fid, value_columns]
        data = [function_results[column].dropna().to_numpy() for column in value_columns]
        boxes = axis.boxplot(data, patch_artist=True, showfliers=False, medianprops={"color": "black"})
        for box, colour in zip(boxes["boxes"], colours):
            box.set_facecolor(colour)
            box.set_alpha(0.85)
        axis.set_title(f"Function f{fid} (n={len(function_results)})")
        axis.set_xticks(range(1, len(value_columns) + 1), [labels[column] for column in value_columns], rotation=35, ha="right")
        axis.set_ylabel(metric.capitalize())
        axis.grid(axis="y", alpha=0.25)

    for axis in list(axes.flat)[len(functions):]:
        axis.set_visible(False)

    legend = [Patch(facecolor=colour, label=labels[column], alpha=0.85) for column, colour in zip(value_columns, colours)]
    figure.legend(handles=legend, loc="upper center", ncol=min(len(legend), 5), frameon=False)
    figure.suptitle(f"{metric.capitalize()} by BBOB function", y=0.995)
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    input_path = args.input or Path("results") / args.metric / f"lookahead_{args.lookahead_count}" / "selector_results.csv"
    output_path = args.output or input_path.with_name("function_wise_boxplots.png")
    if not input_path.is_file():
        raise FileNotFoundError(f"Evaluation results not found: {input_path}")

    results = pd.read_csv(input_path)
    if "fid" not in results.columns:
        raise ValueError("The input CSV must contain a 'fid' column.")
    plot_function_wise_boxplots(results, args.metric, output_path, args.static_budgets, args.columns)
    print(f"Saved function-wise boxplots to {output_path}")


if __name__ == "__main__":
    main()
