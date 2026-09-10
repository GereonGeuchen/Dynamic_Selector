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
from tqdm.auto import tqdm

import plotly.io as pio

# Change this to "auc" to plot the AUC results instead.
METRIC = "auc"
DIM = 40
# None includes all available values; otherwise use lists such as [1, 5, 10].
LOOKAHEAD_COUNTS = [0,10,20]

RESULTS_DIR = Path("results") / f"dim_{DIM}"
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
    results: pd.DataFrame, value_columns: list[str], metric: str,
    lookahead_counts: list[int] | None = None,
) -> Path:
    """Plot selected lookahead counts for every FID, retaining VBS and static B150.

    None selects all available values. Explicit lists preserve their order;
    an empty lookahead_counts list plots only the two baselines.
    """
    fids = sorted(results["fid"].unique())
    if not fids:
        raise ValueError("The result files do not contain any function IDs.")

    if lookahead_counts is not None:
        value_columns = [
            "VBS",
            *(f"Lookahead {count}" for count in dict.fromkeys(lookahead_counts)),
            f"Static B{STATIC_BUDGET}",
        ]
        missing_columns = [column for column in value_columns if column not in results.columns]
        if missing_columns:
            raise ValueError(f"Requested plot columns not present in results: {missing_columns}")

    if pio.kaleido.scope is not None:
        pio.kaleido.scope.mathjax = None

    colours = [TAB20_COLOURS[index % len(TAB20_COLOURS)] for index in range(len(value_columns))]
    output_directory = RESULTS_DIR / metric / PLOT_DIR
    output_directory.mkdir(parents=True, exist_ok=True)
    rows = ceil(len(fids) / PLOT_COLUMNS)
    subplot_titles = [f"Function f{fid} (dim={DIM})" for fid in fids]
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
    # Add considered lookaheads to filename
    OUTPUT_FILENAME = f"function_wise_boxplots_{metric}_{','.join(str(l) for l in lookahead_counts)}.pdf"
    output_path = output_directory / OUTPUT_FILENAME
    save_figure_as_pdf(figure, output_path)
    return output_path


def plot_run_wise_switching_regrets(
    iids: list[int] | None = None,
    *,
    metric: str = "regret",
    dim: int = DIM,
    lookahead_counts: list[int] | None = None,
    selector_directory: Path | None = None,
    data_directory: Path | None = None,
    output_directory: Path | None = None,
) -> list[Path]:
    """Save one PDF plot per (fid, iid, rep), without averaging.

    ``iids=None`` includes all instances. ``data_directory`` points directly to
    an achieved_regrets or achieved_aucs folder, matching ``metric``. Lines connect recorded a1_budget values;
    Non-elitist's B1000 value is a horizontal no-switch reference.
    Merged CSVs take precedence over their .part-* shards.

    ``lookahead_counts=None`` overlays all available selector variants; []
    disables overlays. ``selector_directory`` points to results/dim_N/metric.
    Selector choices are matched by (fid, iid, rep), with algorithm-colored
    switch lines and markers at their recorded achieved metric values.

    Example: plot_run_wise_switching_regrets(iids=[1, 2], metric="auc", dim=40)
    """
    if metric not in {"regret", "auc"}:
        raise ValueError('metric must be either "regret" or "auc".')
    value_column = f"achieved_{metric}"
    metric_label = "AUC" if metric == "auc" else "regret"
    root = Path(__file__).resolve().parent
    data_directory = (
        Path(data_directory) if data_directory is not None
        else root / "data" / f"dim_{dim}" / f"achieved_{metric}s"
    )
    output_directory = (
        Path(output_directory) if output_directory is not None
        else root / "plots" / f"dim_{dim}" / f"switching_{metric}s"
    )
    paths = sorted(data_directory.glob(f"achieved_{metric}s_*_{dim}D*.csv"))
    paths = [
        path for path in paths
        if ".part-" not in path.name
        or not path.with_name(path.name.split(".part-")[0] + ".csv").exists()
    ]
    if not paths:
        raise FileNotFoundError(f"No achieved-{metric} CSVs found in {data_directory}")
    columns = [*KEY_COLUMNS, "a1_budget", "algname", value_column]
    frames = []
    for path in paths:
        frame = pd.read_csv(path, usecols=columns)
        if iids is not None:
            frame = frame.loc[frame["iid"].isin(iids)]
        frames.append(frame)
    results = pd.concat(frames, ignore_index=True).drop_duplicates()
    if results.empty:
        raise ValueError(f"No achieved-{metric} runs found for the requested iids.")
    if iids is not None:
        missing = sorted(set(iids) - set(results["iid"]))
        if missing:
            raise ValueError(f"Instance IDs not present in results: {missing}")
    if results.duplicated([*KEY_COLUMNS, "algname", "a1_budget"]).any():
        raise ValueError(f"Conflicting {metric} values for the same run, algorithm and budget.")

    algorithms = ["Elitist", "PSO", "DE", "BFGS", "MLSL", "Non-elitist"]
    algorithm_colours = {name: TAB20_COLOURS[2 * i] for i, name in enumerate(algorithms)}
    selector_directory = (
        Path(selector_directory) if selector_directory is not None
        else root / "results" / f"dim_{dim}" / metric
    )
    available_paths = dict(lookahead_result_paths(selector_directory))
    counts = sorted(available_paths) if lookahead_counts is None else list(dict.fromkeys(lookahead_counts))
    missing_counts = [count for count in counts if count not in available_paths]
    if missing_counts or (lookahead_counts is None and not available_paths):
        raise FileNotFoundError(
            f"No selector results for lookahead counts {missing_counts or 'any'} in {selector_directory}"
        )
    selector_results = {}
    for count in counts:
        choices = pd.read_csv(
            available_paths[count],
            usecols=[*KEY_COLUMNS, "switch_budget", "selected_algorithm", value_column],
        )
        if iids is not None:
            choices = choices.loc[choices["iid"].isin(iids)]
        if choices.duplicated(KEY_COLUMNS).any():
            raise ValueError(f"Duplicate selector runs for lookahead {count}.")
        if choices[["switch_budget", "selected_algorithm", value_column]].isna().any().any():
            raise ValueError(f"Incomplete selector choices for lookahead {count}.")
        unknown = set(choices["selected_algorithm"]) - set(algorithms)
        if unknown:
            raise ValueError(f"Unknown selected algorithms for lookahead {count}: {sorted(unknown)}")
        selector_results[count] = choices.set_index(KEY_COLUMNS)

    output_directory.mkdir(parents=True, exist_ok=True)
    output_paths = []
    runs = results.groupby(KEY_COLUMNS, sort=True)
    for (fid, iid, rep), run in tqdm(
        runs, total=runs.ngroups, desc="Saving run plots", unit="run"
    ):
        figure = go.Figure()
        for index, algorithm in enumerate(algorithms):
            values = run.loc[run["algname"] == algorithm].sort_values("a1_budget")
            x = values["a1_budget"].tolist()
            y = values[value_column].tolist()
            baseline = algorithm == "Non-elitist" and x == [1000]
            if baseline:
                x, y = [0, 1000], [y[0], y[0]]
            figure.add_trace(go.Scatter(
                x=x, y=y, name=algorithm,
                mode="lines" if baseline else "lines+markers",
                line={"color": TAB20_COLOURS[2 * index],
                      "dash": "dash" if baseline else "solid"},
                connectgaps=False,
                hovertemplate=(
                    f"{algorithm}<br>Switching budget: %{{x}}"
                    f"<br>Achieved {metric_label}: %{{y}}<extra></extra>"
                ),
            ))
        for budget in sorted(run["a1_budget"].dropna().unique()):
            figure.add_vline(
                x=budget, line_width=1, line_color="rgba(0, 0, 0, 0.15)",
                layer="below",
            )
        symbols = ["diamond", "square", "star", "triangle-up", "cross", "x"]
        for index, count in enumerate(counts):
            choices = selector_results[count]
            if (fid, iid, rep) not in choices.index:
                figure.add_trace(go.Scatter(
                    x=[None], y=[None], mode="markers",
                    marker={"color": "gray"},
                    name=f"Lookahead {count}: no result", showlegend=True,
                ))
                continue
            choice = choices.loc[(fid, iid, rep)]
            budget = choice["switch_budget"]
            algorithm = choice["selected_algorithm"]
            colour = algorithm_colours[algorithm]
            figure.add_vline(x=budget, line_color=colour, line_width=2, line_dash="dot")
            figure.add_trace(go.Scatter(
                x=[budget], y=[choice[value_column]], mode="markers",
                name=f"Lookahead {count}: {algorithm}, B={budget:g}",
                marker={"color": colour, "size": 14,
                        "symbol": symbols[index % len(symbols)],
                        "line": {"color": "black", "width": 1}},
                hovertemplate=(
                    f"Lookahead {count}: {algorithm}<br>Switching budget: %{{x}}"
                    f"<br>Achieved {metric_label}: %{{y}}<extra></extra>"
                ),
            ))
        figure.update_layout(
            title=f"Achieved {metric_label} — f{fid}, iid {iid}, rep {rep} ({dim}D)",
            xaxis={
                "title": "Candidate switching budget", "range": [0, 1000],
                "tickmode": "linear", "tick0": 0, "dtick": 100,
                "showgrid": False,
            },
            yaxis={"title": f"Achieved {metric_label}", "type": "linear"},
            legend_title_text="A2 algorithm",
            template="plotly_white", width=1100, height=650,
        )
        output_path = output_directory / f"switching_{metric}_f{fid}_iid{iid}_rep{rep}.pdf"
        figure.write_image(output_path, format="pdf", width=1100, height=650, scale=1)
        output_paths.append(output_path)
    return output_paths


def main() -> None:
    if METRIC not in {"regret", "auc"}:
        raise ValueError('METRIC must be either "regret" or "auc".')
    # results, value_columns = load_plot_data(METRIC)
    # output_path = plot_function_wise_boxplots(
    #     results, value_columns, METRIC, lookahead_counts=LOOKAHEAD_COUNTS
    # )
    # print(f"Saved function-wise boxplots to {output_path}")
    plot_run_wise_switching_regrets(
        iids=[6,7], metric=METRIC, dim=DIM, lookahead_counts=LOOKAHEAD_COUNTS
    )


if __name__ == "__main__":
    main()
