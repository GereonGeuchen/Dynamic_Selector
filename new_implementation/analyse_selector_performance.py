"""Plot achieved selector performance for every BBOB function.

Set ``METRIC`` below to either ``"regret"`` or ``"auc"`` and run this file.
For each FID, the plot contains one box for every evaluated number of
lookahead models, plus VBS, the configured static model, and Non-elitist.
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
DIM = 5
# None includes all available values; otherwise use lists such as [1, 5, 10].
LOOKAHEAD_COUNTS = [0,10,20]
MA = True
# Use default-dataset models for MA results; False uses matching-dataset models.
models_default = True


def result_directory_name(dim: int, ma: bool, models_default: bool) -> str:
    """Match the selector's dataset and model-dataset directory convention."""
    directory = f"dim_{dim}_ma" if ma else f"dim_{dim}"
    if ma and models_default:
        directory += "_models_default"
    return directory


RESULTS_DIR = Path("results") / result_directory_name(DIM, MA, models_default)
PLOT_DIR = Path(__file__).resolve().parent / "plots"
STATIC_BUDGET = 150
OUTPUT_FILENAME = "function_wise_lookahead_boxplots_600_static.pdf"
PLOT_COLUMNS = 4
KEY_COLUMNS = ["fid", "iid", "rep"]
TEST_IIDS = [6, 7]
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

    if pio.kaleido.scope is not None:
        pio.kaleido.scope.mathjax = None

    colours = [TAB20_COLOURS[index % len(TAB20_COLOURS)] for index in range(len(value_columns))]
    dimension_directory = result_directory_name(DIM, MA, models_default)
    output_directory = PLOT_DIR / dimension_directory / metric / "log_scale_boxplots"
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
    dimension_directory = f"dim_{dim}_ma" if MA else f"dim_{dim}"
    result_directory = result_directory_name(dim, MA, models_default)
    data_directory = (
        Path(data_directory) if data_directory is not None
        else root / "data" / dimension_directory / f"achieved_{metric}s"
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
) -> list[Path]:
    """Save A2 proportions and switching-budget boxes per function/lookahead.

    Defaults use the current METRIC, DIM, MA and models_default settings.
    None selects all available lookaheads; [] produces no plots. All instances
    and repetitions are pooled per function unless ``iids`` filters instances.
    Budgets are shown as recorded, including Non-elitist selections.

    Example: plot_a2_distribution_and_switching(LOOKAHEAD_COUNTS)
    """
    metric = METRIC if metric is None else metric
    dim = DIM if dim is None else dim
    if metric not in {"auc", "regret"}:
        raise ValueError('metric must be either "regret" or "auc".')
    root = Path(__file__).resolve().parent
    directory = result_directory_name(dim, MA, models_default)
    metric_directory = root / "results" / directory / metric
    available = dict(lookahead_result_paths(metric_directory))
    counts = sorted(available) if lookahead_counts is None else list(dict.fromkeys(lookahead_counts))
    missing = [count for count in counts if count not in available]
    if missing or (lookahead_counts is None and not available):
        raise FileNotFoundError(
            f"No selector results for lookahead counts {missing or 'any'} in {metric_directory}"
        )
    output_directory = PLOT_DIR / directory / metric / "a2_distribution_and_switching"
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
            title=f"{metric.upper()} — {directory}, lookahead {count}",
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

    root = Path(__file__).resolve().parent
    dataset_directory = f"dim_{dim}_ma" if MA else f"dim_{dim}"
    metric_directory = (
        Path(data_directory)
        if data_directory is not None
        else root / "data" / dataset_directory / f"achieved_{metric}s"
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
) -> Path:
    """Save the VBS counterpart of the A2 distribution and switching plot.

    By default, only the selector test instances (IIDs 6 and 7) are shown.
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
    vbs_results_directory = Path("results") / directory / metric
    vbs_results_directory.mkdir(parents=True, exist_ok=True)
    choices.to_csv(
        vbs_results_directory / "vbs_choices.csv", index=False,
        columns=[*KEY_COLUMNS, "selected_algorithm", "switch_budget", f"vbs_{metric}"],
    )

    output_directory = PLOT_DIR / directory / metric / "vbs_distribution_and_switching"
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
        title=f"{metric.upper()} - VBS ({directory})",
        barmode="stack", template="plotly_white", width=1200, height=800,
        font={"size": 14}, margin={"l": 80, "r": 30, "t": 130, "b": 60},
        legend={"orientation": "h", "x": 0.5, "xanchor": "center", "y": 1.14},
    )
    instance_suffix = "" if iids is None else "_iids_" + "-".join(map(str, sorted(set(iids))))
    output_directory.mkdir(parents=True, exist_ok=True)
    output_path = output_directory / f"vbs_distribution_and_switching{instance_suffix}.pdf"
    figure.write_image(output_path, format="pdf", width=1200, height=800, scale=1)
    return output_path


def main() -> None:
    global METRIC, DIM, MA, models_default, RESULTS_DIR

    for DIM in [5, 40]:
        for MA in [False, True]:
            # for models_default in ([False, True] if MA else [False]):
            #     RESULTS_DIR = Path("results") / result_directory_name(DIM, MA, models_default)
            #     for METRIC in ["auc", "regret"]:
            #         # results, value_columns = load_plot_data(METRIC)
            #         # output_path = plot_function_wise_boxplots(
            #         #     results,
            #         #     value_columns,
            #         #     METRIC,
            #         #     lookahead_counts=LOOKAHEAD_COUNTS,
            #         # )
            #         # print(f"Saved function-wise boxplots to {output_path}")
            #         for output_path in plot_a2_distribution_and_switching(LOOKAHEAD_COUNTS):
            #             print(f"Saved A2 distribution and switching plots to {output_path}")
            plot_vbs_distribution_and_switching(metric=METRIC, dim=DIM)
            directory = f"dim_{DIM}_ma" if MA else f"dim_{DIM}"
            print(
                f"Saved VBS distribution and switching plots for {METRIC}, "
                f"dim={DIM}, MA={MA} to "
                f"{PLOT_DIR / directory / METRIC / 'vbs_distribution_and_switching'}"
            )

if __name__ == "__main__":
    main()
