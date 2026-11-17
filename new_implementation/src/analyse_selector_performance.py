"""Plot achieved selector performance for every BBOB function.

With ``STANDALONE_BOXPLOTS = True``, run this file to plot all six standalone
algorithms for both metrics in every data/dim_*_standalone dataset. Zeros
are displayed at an annotated positive floor on the logarithmic axis.
Set it to False to run the selector analysis instead.
For each FID, the plot contains one box for every evaluated number of
lookahead models, plus VBS, the configured static model, and Non-elitist.
"""

from math import ceil
from pathlib import Path
import re

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from tqdm.auto import tqdm

import plotly.io as pio

# Change this to "auc" to plot the AUC results instead.
METRIC = "auc"
DIM = 5
TEST_IIDS = [6, 7]
# None includes all available values; otherwise use lists such as [1, 5, 10].
LOOKAHEAD_COUNTS = [0,10,20]
MA = True
# Use default-dataset models for MA results; False uses matching-dataset models.
models_default = True
STANDALONE_BOXPLOTS = False
STANDALONE_ALGORITHMS = ["DE", "MLSL", "PSO", "BFGS", "Non-elitist", "Elitist"]


def result_directory_name(dim: int, ma: bool, models_default: bool) -> str:
    """Match the selector's dataset and model-dataset directory convention."""
    directory = f"dim_{dim}_ma" if ma else f"dim_{dim}"
    if ma and models_default:
        directory += "_models_default"
    return directory


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results" / result_directory_name(DIM, MA, models_default)
PLOT_DIR = PROJECT_ROOT / "plots"
STATIC_BUDGET = 150
OUTPUT_FILENAME = "function_wise_lookahead_boxplots_600_static.pdf"
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


def load_plot_data(
    metric: str, *, metric_directory: Path | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Load selector values for all lookahead variants and the shared baselines.

    ``metric_directory`` supports result layouts with extra feature directories,
    e.g. ``results/dim_5/auc/ela``. When omitted, the configured selector
    result directory is used for backward compatibility.
    """
    metric_dir = RESULTS_DIR / metric if metric_directory is None else metric_directory
    result_paths = lookahead_result_paths(metric_dir)
    if not result_paths:
        raise FileNotFoundError(f"No selector results found in {metric_dir}/lookahead_*/")

    achieved_column = f"achieved_{metric}"
    vbs_column = f"vbs_{metric}"
    static_column = f"static_B{STATIC_BUDGET}"

    selector_frames = []
    baseline = None
    required_columns = [*KEY_COLUMNS, achieved_column, vbs_column, static_column, "no_switch"]

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
            baseline["Non-elitist"] = results["no_switch"]

    assert baseline is not None
    plot_data = baseline
    lookahead_labels = []
    for selector_frame in selector_frames:
        label = selector_frame.columns[-1]
        lookahead_labels.append(label)
        plot_data = plot_data.merge(selector_frame, on=KEY_COLUMNS, how="inner", validate="one_to_one")

    plot_data.to_csv(metric_dir / "selector_results_merged.csv", index=False)

    return plot_data, ["VBS", *lookahead_labels, f"Static B{STATIC_BUDGET}", "Non-elitist"]


def plot_all_selector_boxplots() -> None:
    """Plot every selector-result combination, mirroring ``results`` in ``plots``.

    A combination is any directory containing
    ``lookahead_*/selector_results.csv``. This includes dimension, metric,
    and optional feature directories without maintaining a hard-coded list.
    """
    results_root = PROJECT_ROOT / "results"
    metric_directories = sorted({path.parent.parent for path in results_root.glob(
        "**/lookahead_*/selector_results.csv"
    )})
    if not metric_directories:
        raise FileNotFoundError(f"No selector results found below {results_root}")

    for metric_directory in metric_directories:
        relative_directory = metric_directory.relative_to(results_root)
        metric_candidates = [part for part in relative_directory.parts if part in {"auc", "regret"}]
        dimension_candidates = [
            int(match.group(1))
            for part in relative_directory.parts
            if (match := re.fullmatch(r"dim_(\d+)(?:_.*)?", part)) is not None
        ]
        if len(metric_candidates) != 1 or len(dimension_candidates) != 1:
            print(f"Skipping unrecognised result directory: {metric_directory}", flush=True)
            continue

        metric = metric_candidates[0]
        dim = dimension_candidates[0]
        results, value_columns = load_plot_data(metric, metric_directory=metric_directory)
        output_path = plot_function_wise_boxplots(
            results,
            value_columns,
            metric,
            lookahead_counts=LOOKAHEAD_COUNTS,
            dim=dim,
            output_directory=PLOT_DIR / relative_directory / "log_scale_boxplots",
        )
        print(f"Saved {output_path}", flush=True)


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
    *,
    dim: int | None = None,
    output_directory: Path | None = None,
    output_filename: str | None = None,
    zero_floor: float | None = None,
) -> Path:
    """Plot selected lookahead counts for every FID with all three baselines.

    None selects all available values. Explicit lists preserve their order;
    an empty lookahead_counts list plots only the three baselines.
    """
    fids = sorted(results["fid"].unique())
    if not fids:
        raise ValueError("The result files do not contain any function IDs.")

    if lookahead_counts is not None:
        value_columns = [
            "VBS",
            *(f"Lookahead {count}" for count in dict.fromkeys(lookahead_counts)),
            f"Static B{STATIC_BUDGET}",
            "Non-elitist",
        ]
        missing_columns = [column for column in value_columns if column not in results.columns]
        if missing_columns:
            raise ValueError(f"Requested plot columns not present in results: {missing_columns}")

    # Plotly < 6 exposed the Kaleido scope here. Plotly 6 removed that public
    # attribute, so only disable MathJax when the legacy scope is available.
    kaleido_scope = getattr(pio.kaleido, "scope", None)
    if kaleido_scope is not None:
        kaleido_scope.mathjax = None

    colours = [TAB20_COLOURS[index % len(TAB20_COLOURS)] for index in range(len(value_columns))]
    dim = DIM if dim is None else dim
    if output_directory is None:
        dimension_directory = result_directory_name(DIM, MA, models_default)
        output_directory = PLOT_DIR / dimension_directory / metric / "log_scale_boxplots"
    output_directory.mkdir(parents=True, exist_ok=True)
    rows = ceil(len(fids) / PLOT_COLUMNS)
    subplot_titles = [f"Function f{fid} (dim={dim})" for fid in fids]
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
    if zero_floor is not None:
        figure.add_annotation(
            text=f"Zero values displayed at {zero_floor:.3g} on the logarithmic scale",
            xref="paper", yref="paper", x=0.5, y=1.035,
            showarrow=False, font={"size": 13},
        )
    # Add considered lookaheads to filename
    if output_filename is None:
        counts = "all" if lookahead_counts is None else ','.join(map(str, lookahead_counts))
        output_filename = f"function_wise_boxplots_{metric}_{counts}.pdf"
    output_path = output_directory / output_filename
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
    root = PROJECT_ROOT
    dimension_directory = f"dim_{dim}_ma" if MA else f"dim_{dim}"
    result_directory = result_directory_name(dim, MA, models_default)
    data_directory = (
        Path(data_directory) if data_directory is not None
        else root / "data" / dimension_directory / "run_data" / f"achieved_{metric}s"
    )
    output_directory = (
        Path(output_directory) if output_directory is not None
        else PLOT_DIR / result_directory / f"switching_{metric}s"
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
        else root / "results" / result_directory / metric
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
            yaxis={"title": f"Achieved {metric_label}", "type": "log"},
            legend_title_text="A2 algorithm",
            template="plotly_white", width=1100, height=650,
        )
        output_path = output_directory / f"switching_{metric}_f{fid}_iid{iid}_rep{rep}.pdf"
        figure.write_image(output_path, format="pdf", width=1100, height=650, scale=1)
        output_paths.append(output_path)
    return output_paths


def plot_a2_distribution_and_switching(
    lookahead_counts: list[int] | None = None,
    *,
    metric: str | None = None,
    dim: int | None = None,
    iids: list[int] | None = None,
    metric_directory: Path | None = None,
    output_directory: Path | None = None,
    title_directory: str | None = None,
) -> list[Path]:
    """Save A2 proportions and switching-budget boxes per function/lookahead.

    Defaults use the current METRIC, DIM, MA and models_default settings.
    ``metric_directory`` and ``output_directory`` allow the function to be
    used for any discovered results combination. None selects all available
    lookaheads; [] produces no plots. All instances and repetitions are pooled
    per function unless ``iids`` filters instances. Budgets are shown as
    recorded, including Non-elitist selections.

    Example: plot_a2_distribution_and_switching(LOOKAHEAD_COUNTS)
    """
    metric = METRIC if metric is None else metric
    dim = DIM if dim is None else dim
    if metric not in {"auc", "regret"}:
        raise ValueError('metric must be either "regret" or "auc".')
    root = PROJECT_ROOT
    directory = result_directory_name(dim, MA, models_default)
    metric_directory = (
        root / "results" / directory / metric
        if metric_directory is None else Path(metric_directory)
    )
    title_directory = directory if title_directory is None else title_directory
    available = dict(lookahead_result_paths(metric_directory))
    counts = sorted(available) if lookahead_counts is None else list(dict.fromkeys(lookahead_counts))
    missing = [count for count in counts if count not in available]
    if missing or (lookahead_counts is None and not available):
        raise FileNotFoundError(
            f"No selector results for lookahead counts {missing or 'any'} in {metric_directory}"
        )
    output_directory = (
        PLOT_DIR / directory / metric / "a2_distribution_and_switching"
        if output_directory is None else Path(output_directory)
    )
    colours = {
        "BFGS": "#1f77b4", "Non-elitist": "#ff7f0e", "DE": "#2ca02c",
        "PSO": "#d62728", "MLSL": "#9467bd", "Elitist": "#8c564b",
    }
    display_names = {
        "Elitist": "CMA-ES, elitist", "Non-elitist": "CMA-ES, non-elitist",
    }
    output_paths = []
    for count in counts:
        results = pd.read_csv(
            available[count], usecols=[*KEY_COLUMNS, "selected_algorithm", "switch_budget"]
        )
        if iids is not None:
            results = results.loc[results["iid"].isin(iids)]
        if results.empty:
            raise ValueError(f"No selector runs for lookahead {count} and requested instances.")
        if results.isna().any().any() or results.duplicated(KEY_COLUMNS).any():
            raise ValueError(f"Incomplete or duplicate selector runs for lookahead {count}.")
        unknown = set(results["selected_algorithm"]) - set(colours)
        if unknown:
            raise ValueError(f"Unknown selected algorithms: {sorted(unknown)}")
        results["switch_budget"] = pd.to_numeric(results["switch_budget"], errors="raise")
        fids = sorted(results["fid"].unique())
        frequencies = pd.crosstab(results["fid"], results["selected_algorithm"])
        proportions = frequencies.div(frequencies.sum(axis=1), axis=0).reindex(fids)
        figure = make_subplots(
            rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.13,
            subplot_titles=["A2 algorithm distribution", "Recorded switching budget"],
        )
        for algorithm, colour in colours.items():
            if algorithm not in proportions.columns:
                continue
            figure.add_trace(go.Bar(
                x=fids, y=proportions[algorithm],
                name=display_names.get(algorithm, algorithm), marker_color=colour,
                width=0.86,
                hovertemplate="f%{x}<br>Proportion: %{y:.1%}<extra>%{fullData.name}</extra>",
            ), row=1, col=1)
        for fid in fids:
            budgets = results.loc[results["fid"] == fid, "switch_budget"]
            figure.add_trace(go.Box(
                x=[fid] * len(budgets), y=budgets, name=f"f{fid}",
                boxpoints=False, width=0.55, fillcolor="royalblue",
                line={"color": "black", "width": 1.5}, showlegend=False,
            ), row=2, col=1)
        figure.update_xaxes(
            tickmode="array", tickvals=fids, range=[min(fids) - 0.5, max(fids) + 0.5],
            showticklabels=True, showline=True, linecolor="black", zeroline=False,
        )
        figure.update_xaxes(title_text="BBOB function", row=2, col=1)
        figure.update_yaxes(title_text="Proportion", range=[0, 1], tickformat=".0%", row=1, col=1)
        figure.update_yaxes(title_text="Switching budget", rangemode="tozero", row=2, col=1)
        figure.update_layout(
            title=f"{metric.upper()} — {title_directory}, lookahead {count}",
            barmode="stack", template="plotly_white", width=1200, height=800,
            font={"size": 14}, margin={"l": 80, "r": 30, "t": 130, "b": 60},
            legend={"orientation": "h", "x": 0.5, "xanchor": "center", "y": 1.14},
        )
        instance_suffix = "" if iids is None else "_iids_" + "-".join(map(str, sorted(set(iids))))
        output_directory.mkdir(parents=True, exist_ok=True)
        output_path = output_directory / f"a2_distribution_and_switching_lookahead_{count}{instance_suffix}.pdf"
        figure.write_image(output_path, format="pdf", width=1200, height=800, scale=1)
        output_paths.append(output_path)
    return output_paths


def plot_all_a2_distributions_and_switching(
    lookahead_counts: list[int] | None = None,
) -> list[Path]:
    """Plot A2 distributions and switching budgets for every result combination.

    Each directory below ``results`` containing
    ``lookahead_*/selector_results.csv`` is treated as one combination. The
    matching output is written below ``plots`` with the same relative
    directory. By default all available lookahead values are plotted; pass
    ``[0, 10, 20]`` to create only those three variants.
    """
    results_root = PROJECT_ROOT / "results"
    metric_directories = sorted({
        path.parent.parent
        for path in results_root.glob("**/lookahead_*/selector_results.csv")
    })
    if not metric_directories:
        raise FileNotFoundError(f"No selector results found below {results_root}")

    output_paths = []
    for metric_directory in metric_directories:
        relative_directory = metric_directory.relative_to(results_root)
        metric_candidates = [part for part in relative_directory.parts if part in {"auc", "regret"}]
        dimension_candidates = [
            int(match.group(1))
            for part in relative_directory.parts
            if (match := re.fullmatch(r"dim_(\d+)(?:_.*)?", part)) is not None
        ]
        if len(metric_candidates) != 1 or len(dimension_candidates) != 1:
            print(f"Skipping unrecognised result directory: {metric_directory}", flush=True)
            continue

        sample_paths = lookahead_result_paths(metric_directory)
        sample_columns = set(pd.read_csv(sample_paths[0][1], nrows=0).columns)
        required_columns = {*KEY_COLUMNS, "selected_algorithm", "switch_budget"}
        missing_columns = required_columns - sample_columns
        if missing_columns:
            print(
                "Skipping A2 distribution and switching plots for "
                f"{relative_directory}; unsupported result schema (missing "
                f"{', '.join(sorted(missing_columns))})",
                flush=True,
            )
            continue

        print(f"Creating A2 distribution and switching plots for {relative_directory}", flush=True)
        paths = plot_a2_distribution_and_switching(
            lookahead_counts,
            metric=metric_candidates[0],
            dim=dimension_candidates[0],
            metric_directory=metric_directory,
            output_directory=(PLOT_DIR / relative_directory / "a2_distribution_and_switching"),
            title_directory=str(relative_directory),
        )
        output_paths.extend(paths)
        for path in paths:
            print(f"Saved {path}", flush=True)
    return output_paths


FEATURE_COMPARISON_DIRECTORIES = {
    "ela": "ELA",
    "ela_internal_state": "ELA + internal state",
    "internal_state": "Internal state",
}


def plot_feature_comparison_boxplots(
    lookahead_counts: list[int] | None = None,
) -> list[Path]:
    """Compare feature sets at a fixed lookahead in function-wise log boxplots.

    Standard results are expected at
    ``results/dim_N/<metric>/<feature_set>/lookahead_K/selector_results.csv``.
    One PDF is produced per dimension, metric, and requested lookahead. Each
    function panel compares the achieved selector value for ELA, ELA plus
    internal state, and internal state alone.
    """
    results_root = PROJECT_ROOT / "results"
    output_paths = []
    dimension_directories = sorted(
        path for path in results_root.glob("dim_*")
        if path.is_dir() and re.fullmatch(r"dim_\d+", path.name)
    )
    for dimension_directory in dimension_directories:
        dim = int(dimension_directory.name.removeprefix("dim_"))
        for metric in ("auc", "regret"):
            feature_directories = {
                directory: dimension_directory / metric / directory
                for directory in FEATURE_COMPARISON_DIRECTORIES
            }
            missing_features = [
                label for label, directory in feature_directories.items()
                if not directory.is_dir()
            ]
            if missing_features:
                print(
                    f"Skipping feature comparison for {dimension_directory.name}/{metric}; "
                    f"missing feature results: {', '.join(missing_features)}",
                    flush=True,
                )
                continue

            available_counts = set.intersection(*[
                {count for count, _ in lookahead_result_paths(directory)}
                for directory in feature_directories.values()
            ])
            counts = (
                sorted(available_counts) if lookahead_counts is None
                else list(dict.fromkeys(lookahead_counts))
            )
            missing_counts = [count for count in counts if count not in available_counts]
            if missing_counts:
                print(
                    f"Skipping unavailable lookaheads for {dimension_directory.name}/{metric}: "
                    f"{missing_counts}",
                    flush=True,
                )
            for count in counts:
                if count not in available_counts:
                    continue
                print(
                    f"Creating feature-comparison boxplot for "
                    f"{dimension_directory.name}/{metric}, lookahead {count}",
                    flush=True,
                )
                frames = []
                achieved_column = f"achieved_{metric}"
                for feature_directory, label in FEATURE_COMPARISON_DIRECTORIES.items():
                    result_path = (
                        feature_directories[feature_directory]
                        / f"lookahead_{count}" / "selector_results.csv"
                    )
                    results = pd.read_csv(result_path)
                    required_columns = [*KEY_COLUMNS, achieved_column]
                    missing_columns = set(required_columns) - set(results.columns)
                    if missing_columns:
                        raise ValueError(
                            f"{result_path} is missing required columns: "
                            f"{', '.join(sorted(missing_columns))}"
                        )
                    frames.append(results.loc[:, required_columns].rename(
                        columns={achieved_column: label}
                    ))

                comparison = frames[0]
                for frame in frames[1:]:
                    comparison = comparison.merge(
                        frame, on=KEY_COLUMNS, how="inner", validate="one_to_one"
                    )
                if comparison.empty:
                    raise ValueError(
                        f"No shared selector runs for {dimension_directory.name}/{metric}, "
                        f"lookahead {count}"
                    )
                labels = list(FEATURE_COMPARISON_DIRECTORIES.values())
                output_path = plot_function_wise_boxplots(
                    comparison,
                    labels,
                    metric,
                    lookahead_counts=None,
                    dim=dim,
                    output_directory=(
                        PLOT_DIR / dimension_directory.name / metric
                        / "feature_comparison_log_scale_boxplots"
                    ),
                    output_filename=(
                        f"function_wise_feature_comparison_{metric}_lookahead_{count}.pdf"
                    ),
                )
                output_paths.append(output_path)
                print(f"Saved {output_path}", flush=True)
    return output_paths


def load_standalone_plot_data(directory: Path, metric: str, dim: int) -> pd.DataFrame:
    """Load B0 standalone runs, requiring matching run keys across algorithms."""
    column = f"achieved_{metric}"
    frames = []
    for algorithm in STANDALONE_ALGORITHMS:
        path = directory / f"achieved_{metric}s" / f"achieved_{metric}s_{algorithm}_B0_{dim}D.csv"
        frame = pd.read_csv(path)
        required = [*KEY_COLUMNS, "a1_budget", "algname", column]
        missing = set(required) - set(frame.columns)
        if missing:
            raise ValueError(f"{path}: missing columns {sorted(missing)}")
        if frame.empty or frame[KEY_COLUMNS].isna().any().any():
            raise ValueError(f"{path}: empty data or missing run keys")
        if not frame["a1_budget"].eq(0).all() or not frame["algname"].eq(algorithm).all():
            raise ValueError(f"{path}: expected standalone B0 runs for {algorithm}")
        values = pd.to_numeric(frame[column], errors="raise")
        if not np.isfinite(values).all() or values.lt(0).any():
            raise ValueError(f"{path}: expected finite, nonnegative values")
        series = frame.set_index(KEY_COLUMNS)[column].rename(algorithm)
        if not series.index.is_unique:
            raise ValueError(f"{path}: duplicate run keys")
        if frames and (len(series) != len(frames[0]) or not series.index.isin(frames[0].index).all()):
            raise ValueError(f"{path}: run keys differ between algorithms")
        frames.append(series)
    return pd.concat(frames, axis=1).reset_index()


def plot_standalone_boxplots() -> None:
    """Save both metrics for every standalone dataset using the selector layout."""
    directories = sorted((PROJECT_ROOT / "data").glob("dim_*_standalone"))
    if not directories:
        raise FileNotFoundError(f"No standalone datasets found in {PROJECT_ROOT / 'data'}")
    for directory in directories:
        match = re.fullmatch(r"dim_(\d+)(?:_ma)?_standalone", directory.name)
        if match is None:
            continue
        dim = int(match.group(1))
        for metric in ["auc", "regret"]:
            results = load_standalone_plot_data(directory, metric, dim)
            values = results[STANDALONE_ALGORITHMS]
            zero_floor = None
            if values.eq(0).any().any():
                positive_min = values.where(values.gt(0)).min().min()
                if pd.isna(positive_min):
                    raise ValueError(f"{directory}: no positive {metric} values for a log scale")
                zero_floor = positive_min / 10
                results[STANDALONE_ALGORITHMS] = values.mask(values.eq(0), zero_floor)
            output_path = plot_function_wise_boxplots(
                results, STANDALONE_ALGORITHMS, metric, dim=dim,
                zero_floor=zero_floor,
                output_directory=PROJECT_ROOT / "plots" / directory.name / metric / "log_scale_boxplots",
                output_filename=f"function_wise_boxplots_{metric}_standalone.pdf",
            )
            print(f"Saved {output_path} ({len(results)} runs per algorithm)", flush=True)

def calculate_vbs_choices(
    *,
    metric: str,
    dim: int,
    iids: list[int] | None = None,
    data_directory: Path | None = None,
) -> pd.DataFrame:
    """Return the VBS algorithm and budget for every recorded run.

    The VBS is computed from the same raw achieved-metric files as the
    selector: the lowest metric across all A2 algorithm/budget combinations
    and the no-switch baseline. Equal metric values use the latest budget,
    then alphabetical algorithm order, to yield one reproducible choice.
    """
    if metric not in {"auc", "regret"}:
        raise ValueError('metric must be either "auc" or "regret".')

    root = PROJECT_ROOT
    dataset_directory = f"dim_{dim}_ma" if MA else f"dim_{dim}"
    metric_directory = (
        Path(data_directory)
        if data_directory is not None
        else root / "data" / dataset_directory / "run_data" / f"achieved_{metric}s"
    )
    metric_paths = sorted(metric_directory.glob(f"achieved_{metric}s_*_{dim}D.csv"))
    if not metric_paths:
        raise FileNotFoundError(f"No achieved-{metric} files found in {metric_directory}")

    metric_column = f"achieved_{metric}"
    choices = pd.concat(
        [
            pd.read_csv(
                path,
                usecols=[*KEY_COLUMNS, "a1_budget", "algname", metric_column],
            )
            for path in metric_paths
        ],
        ignore_index=True,
    )
    if iids is not None:
        choices = choices.loc[choices["iid"].isin(iids)]
    if choices.empty:
        raise ValueError("No achieved-metric runs found for the requested instances.")
    if choices.duplicated([*KEY_COLUMNS, "algname", "a1_budget"]).any():
        raise ValueError("Conflicting metric values for the same run, algorithm and budget.")

    choices = choices.sort_values(
        [*KEY_COLUMNS, metric_column, "a1_budget", "algname"],
        ascending=[True, True, True, True, False, True],
        kind="stable",
    )
    return (
        choices.drop_duplicates(KEY_COLUMNS)
        .rename(
            columns={
                "algname": "selected_algorithm",
                "a1_budget": "switch_budget",
                metric_column: f"vbs_{metric}",
            }
        )
        .reset_index(drop=True)
    )


def plot_vbs_distribution_and_switching(
    *,
    metric: str | None = None,
    dim: int | None = None,
    iids: list[int] | None = None,
    data_directory: Path | None = None,
    output_directory: Path | None = None,
    results_directory: Path | None = None,
    title_directory: str | None = None,
) -> Path:
    """Save the VBS counterpart of the A2 distribution and switching plot.

    By default, only the selector test instances (IIDs 6 and 7) are shown.
    The optional directory arguments support batch plotting of result layouts
    beyond the globally configured dataset.
    """
    metric = METRIC if metric is None else metric
    dim = DIM if dim is None else dim
    iids = TEST_IIDS if iids is None else iids
    choices = calculate_vbs_choices(
        metric=metric,
        dim=dim,
        iids=iids,
        data_directory=data_directory,
    )

    # VBS comes directly from raw outcomes, so model-training provenance is irrelevant.
    directory = f"dim_{dim}_ma" if MA else f"dim_{dim}"
    title_directory = directory if title_directory is None else title_directory
    vbs_results_directory = (
        PROJECT_ROOT / "results" / directory / metric
        if results_directory is None else Path(results_directory)
    )
    vbs_results_directory.mkdir(parents=True, exist_ok=True)
    choices.to_csv(
        vbs_results_directory / "vbs_choices.csv", index=False,
        columns=[*KEY_COLUMNS, "selected_algorithm", "switch_budget", f"vbs_{metric}"],
    )

    output_directory = (
        PLOT_DIR / directory / metric / "vbs_distribution_and_switching"
        if output_directory is None else Path(output_directory)
    )
    colours = {
        "BFGS": "#1f77b4", "Non-elitist": "#ff7f0e", "DE": "#2ca02c",
        "PSO": "#d62728", "MLSL": "#9467bd", "Elitist": "#8c564b",
    }
    display_names = {
        "Elitist": "CMA-ES, elitist", "Non-elitist": "CMA-ES, non-elitist",
    }
    unknown = set(choices["selected_algorithm"]) - set(colours)
    if unknown:
        raise ValueError(f"Unknown VBS algorithms: {sorted(unknown)}")

    fids = sorted(choices["fid"].unique())
    frequencies = pd.crosstab(choices["fid"], choices["selected_algorithm"])
    proportions = frequencies.div(frequencies.sum(axis=1), axis=0).reindex(fids)
    figure = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.13,
        subplot_titles=["VBS algorithm distribution", "VBS switching budget"],
    )
    for algorithm, colour in colours.items():
        if algorithm not in proportions.columns:
            continue
        figure.add_trace(go.Bar(
            x=fids, y=proportions[algorithm],
            name=display_names.get(algorithm, algorithm), marker_color=colour,
            width=0.86,
            hovertemplate="f%{x}<br>Proportion: %{y:.1%}<extra>%{fullData.name}</extra>",
        ), row=1, col=1)
    for fid in fids:
        budgets = choices.loc[choices["fid"] == fid, "switch_budget"]
        figure.add_trace(go.Box(
            x=[fid] * len(budgets), y=budgets, name=f"f{fid}",
            boxpoints=False, width=0.55, fillcolor="royalblue",
            line={"color": "black", "width": 1.5}, showlegend=False,
        ), row=2, col=1)
    figure.update_xaxes(
        tickmode="array", tickvals=fids, range=[min(fids) - 0.5, max(fids) + 0.5],
        showticklabels=True, showline=True, linecolor="black", zeroline=False,
    )
    figure.update_xaxes(title_text="BBOB function", row=2, col=1)
    figure.update_yaxes(title_text="Proportion", range=[0, 1], tickformat=".0%", row=1, col=1)
    figure.update_yaxes(title_text="Switching budget", rangemode="tozero", row=2, col=1)
    figure.update_layout(
        title=f"{metric.upper()} - VBS ({title_directory})",
        barmode="stack", template="plotly_white", width=1200, height=800,
        font={"size": 14}, margin={"l": 80, "r": 30, "t": 130, "b": 60},
        legend={"orientation": "h", "x": 0.5, "xanchor": "center", "y": 1.14},
    )
    instance_suffix = "" if iids is None else "_iids_" + "-".join(map(str, sorted(set(iids))))
    output_directory.mkdir(parents=True, exist_ok=True)
    output_path = output_directory / f"vbs_distribution_and_switching{instance_suffix}.pdf"
    figure.write_image(output_path, format="pdf", width=1200, height=800, scale=1)
    return output_path


def plot_all_vbs_distributions_and_switching() -> list[Path]:
    """Plot VBS distributions for every selector result combination.

    VBS choices are recomputed from each combination's raw ``run_data``. A
    combination is skipped (with a message) when its raw data are unavailable,
    which lets selector-only result archives remain usable.
    """
    results_root = PROJECT_ROOT / "results"
    data_root = PROJECT_ROOT / "data"
    metric_directories = sorted({
        path.parent.parent
        for path in results_root.glob("**/lookahead_*/selector_results.csv")
    })
    if not metric_directories:
        raise FileNotFoundError(f"No selector results found below {results_root}")

    output_paths = []
    for metric_directory in metric_directories:
        relative_directory = metric_directory.relative_to(results_root)
        metric_candidates = [part for part in relative_directory.parts if part in {"auc", "regret"}]
        dimension_matches = [
            (index, part, match)
            for index, part in enumerate(relative_directory.parts)
            if (match := re.fullmatch(r"dim_(\d+)(?:_.*)?", part)) is not None
        ]
        if len(metric_candidates) != 1 or len(dimension_matches) != 1:
            print(f"Skipping unrecognised result directory: {metric_directory}", flush=True)
            continue

        dimension_index, dimension_directory, match = dimension_matches[0]
        dim = int(match.group(1))
        # Prefer a data layout that preserves any prefix such as
        # ``old_switching_logic``; fall back to the standard dimension dataset.
        prefixed_data_directory = (
            data_root / Path(*relative_directory.parts[:dimension_index + 1]) / "run_data"
        )
        standard_data_directory = data_root / dimension_directory / "run_data"
        data_directory = next(
            (path for path in (prefixed_data_directory, standard_data_directory) if path.is_dir()),
            None,
        )
        if data_directory is None:
            print(f"Skipping VBS plot; raw run data not found for {metric_directory}", flush=True)
            continue

        print(f"Creating VBS distribution and switching plot for {relative_directory}", flush=True)
        output_path = plot_vbs_distribution_and_switching(
            metric=metric_candidates[0],
            dim=dim,
            data_directory=data_directory / f"achieved_{metric_candidates[0]}s",
            results_directory=metric_directory,
            output_directory=(PLOT_DIR / relative_directory / "vbs_distribution_and_switching"),
            title_directory=str(relative_directory),
        )
        output_paths.append(output_path)
        print(f"Saved {output_path}", flush=True)
    return output_paths

def main() -> None:
    if STANDALONE_BOXPLOTS:
        plot_standalone_boxplots()
    else:
        # plot_all_selector_boxplots()
        # plot_all_a2_distributions_and_switching(LOOKAHEAD_COUNTS)
        # plot_all_vbs_distributions_and_switching()
        plot_feature_comparison_boxplots(LOOKAHEAD_COUNTS)


if __name__ == "__main__":
    main()
