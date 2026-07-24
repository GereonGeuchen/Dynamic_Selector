"""Plot achieved selector performance for every BBOB function.

Set ``METRIC`` below to either ``"regret"`` or ``"auc"`` and run this file.
For each FID, the plot contains one box for every evaluated number of
lookahead models, plus the VBS and the static selection model at B=150.
"""

from math import ceil
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Change this to "auc" to plot the AUC results instead.
METRIC = "regret"

RESULTS_DIR = Path("results")
PLOT_DIR = Path("plots")
STATIC_BUDGET = 150
OUTPUT_FILENAME = "function_wise_lookahead_boxplots.pdf"
PLOT_COLUMNS = 4
KEY_COLUMNS = ["fid", "iid", "rep"]
TAB20_COLOURS = [
    "#1F77B4", "#AEC7E8", "#FF7F0E", "#FFBB78", "#2CA02C", "#98DF8A",
    "#D62728", "#FF9896", "#9467BD", "#C5B0D5", "#8C564B", "#C49C94",
    "#E377C2", "#F7B6D2", "#7F7F7F", "#C7C7C7", "#BCBD22", "#DBDB8D",
    "#17BECF", "#9EDAE5",
]


def lookahead_result_paths(metric_dir: Path) -> list[tuple[int, Path]]:
    """Return available result CSVs ordered by their lookahead count."""
    paths = []
    for directory in metric_dir.glob("lookahead_*"):
        try:
            count = int(directory.name.removeprefix("lookahead_"))
        except ValueError:
            continue
        result_path = directory / "selector_results.csv"
        if result_path.is_file():
            paths.append((count, result_path))
    return sorted(paths)


def load_plot_data(metric: str) -> tuple[pd.DataFrame, list[str]]:
    """Load selector values for all lookahead variants and the shared baselines."""
    metric_dir = RESULTS_DIR / metric
    result_paths = lookahead_result_paths(metric_dir)
    if not result_paths:
        raise FileNotFoundError(f"No selector results found in {metric_dir}/lookahead_*/")

    achieved_column = f"achieved_{metric}"
    vbs_column = f"vbs_{metric}"
    static_column = f"static_B{STATIC_BUDGET}"

    selector_frames = []
    baseline = None
    required_columns = [*KEY_COLUMNS, achieved_column, vbs_column, static_column]

    for lookahead_count, result_path in result_paths:
        results = pd.read_csv(result_path)
        missing = [column for column in required_columns if column not in results.columns]
        if missing:
            raise ValueError(f"{result_path} is missing required columns: {', '.join(missing)}")

        label = f"Lookahead {lookahead_count}"
        selector_frames.append(
            results.loc[:, [*KEY_COLUMNS, achieved_column]].rename(columns={achieved_column: label})
        )
        if baseline is None:
            baseline = results.loc[:, [*KEY_COLUMNS, vbs_column, static_column]].rename(
                columns={vbs_column: "VBS", static_column: f"Static B{STATIC_BUDGET}"}
            )

    assert baseline is not None
    plot_data = baseline
    lookahead_labels = []
    for selector_frame in selector_frames:
        label = selector_frame.columns[-1]
        lookahead_labels.append(label)
        plot_data = plot_data.merge(selector_frame, on=KEY_COLUMNS, how="inner", validate="one_to_one")

    plot_data.to_csv(metric_dir / "selector_results_merged.csv", index=False)

    return plot_data, ["VBS", *lookahead_labels, f"Static B{STATIC_BUDGET}"]


def save_figure_as_pdf(figure: go.Figure, output_path: Path) -> None:
    """Save one-page vector PDF using Plotly's native Kaleido export."""
    try:
        figure.write_image(output_path, format="pdf", width=2200, height=2520, scale=1)
    except ValueError as error:
        raise RuntimeError(
            "Plotly PDF export requires Kaleido. Install it with: python -m pip install kaleido"
        ) from error


def plot_function_wise_boxplots(
    results: pd.DataFrame, value_columns: list[str], metric: str
) -> Path:
    """Save one Plotly-rendered PDF containing a boxplot panel for every FID."""
    fids = sorted(results["fid"].unique())
    if not fids:
        raise ValueError("The result files do not contain any function IDs.")

    colours = [TAB20_COLOURS[index % len(TAB20_COLOURS)] for index in range(len(value_columns))]
    output_directory = RESULTS_DIR / metric / PLOT_DIR
    output_directory.mkdir(parents=True, exist_ok=True)
    rows = ceil(len(fids) / PLOT_COLUMNS)
    subplot_titles = [f"Function f{fid} (n={sum(results['fid'] == fid)})" for fid in fids]
    subplot_titles.extend([""] * (rows * PLOT_COLUMNS - len(fids)))
    figure = make_subplots(
        rows=rows,
        cols=PLOT_COLUMNS,
        subplot_titles=subplot_titles,
        vertical_spacing=0.04,
        horizontal_spacing=0.04,
    )

    for index, fid in enumerate(fids):
        row = index // PLOT_COLUMNS + 1
        column = index % PLOT_COLUMNS + 1
        function_results = results.loc[results["fid"] == fid, value_columns]
        for value_column, colour in zip(value_columns, colours):
            figure.add_trace(
                go.Box(
                    y=function_results[value_column],
                    name=value_column,
                    marker_color=colour,
                    line={"color": "#333333", "width": 1},
                    fillcolor=colour,
                    opacity=0.85,
                    boxpoints=False,
                    showlegend=False,
                    hovertemplate=(
                        f"{value_column}<br>{metric.upper()}: %{{y}}<extra>f{fid}</extra>"
                    ),
                ),
                row=row,
                col=column,
            )
        figure.update_xaxes(
            categoryorder="array",
            categoryarray=value_columns,
            tickangle=35,
            tickfont={"size": 8},
            showline=True,
            linewidth=1,
            linecolor="#808080",
            ticks="",
            showgrid=False,
            zeroline=False,
            row=row,
            col=column,
        )
        figure.update_yaxes(
            title_text=metric.upper(),
            title_font={"size": 13},
            tickfont={"size": 10},
            type="log",
            showline=True,
            linewidth=1,
            linecolor="#808080",
            ticks="",
            showgrid=True,
            gridcolor="rgba(0, 0, 0, 0.25)",
            gridwidth=1,
            zeroline=False,
            row=row,
            col=column,
        )

    figure.update_layout(
        title=f"Achieved {metric.upper()} by BBOB function",
        title_x=0.5,
        title_font={"size": 24},
        font={"size": 12},
        paper_bgcolor="white",
        plot_bgcolor="white",
        height=420 * rows,
        width=2200,
        margin={"l": 80, "r": 50, "t": 110, "b": 50},
        showlegend=False,
    )
    output_path = output_directory / OUTPUT_FILENAME
    save_figure_as_pdf(figure, output_path)
    return output_path


def main() -> None:
    if METRIC not in {"regret", "auc"}:
        raise ValueError('METRIC must be either "regret" or "auc".')
    results, value_columns = load_plot_data(METRIC)
    output_path = plot_function_wise_boxplots(results, value_columns, METRIC)
    print(f"Saved function-wise boxplots to {output_path}")


if __name__ == "__main__":
    main()
