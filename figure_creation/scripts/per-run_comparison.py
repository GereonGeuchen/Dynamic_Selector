import os
import plotly
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go


BUDGETS = [50*i for i in range(1, 21)]
ALGORITHMS = ["BFGS", "PSO", "DE", "MLSL", "Elitist", "Non-elitist"]
SBS_prec = 1836.78
OBS_prec = 1254.1
VBS_prec = 767.08
SELECTOR_prec = 1681.49
per_instance_prec = 2571.83

df_prec_train = pd.read_csv("../data/A2_precisions.csv")
df_prec_test = pd.read_csv("../data/A2_precisions_test.csv")
df_results = pd.read_csv("../data/selector_performance_data/selector_results_with_lookahead_all_epms_10_sbs.csv")
# df_results_per_instance = pd.read_csv("../data/per_instance_selector_results_150_all_reps.csv")

results = {
    "emps": {
        -1: {"closed_gap": 0.04495549290750028, "significant": False},
        0: {"closed_gap": 0.05816047314019725, "significant": False},
        1: {"closed_gap": 0.062209060143000976, "significant": False},
        2: {"closed_gap": 0.07023722275921616, "significant": False},
        3: {"closed_gap": 0.07337476377253006, "significant": False},
        4: {"closed_gap": 0.11292963039563086, "significant": False},
        5: {"closed_gap": 0.06655703883664607, "significant": False},
        6: {"closed_gap": 0.0802166214165512, "significant": False},
        7: {"closed_gap": 0.13616605788977787, "significant": True},
        8: {"closed_gap": 0.10417639233344733, "significant": False},
        9: {"closed_gap": 0.11552442954180812, "significant": False},
        10: {"closed_gap": 0.1451659661251559, "significant": True},
        11: {"closed_gap": 0.11188335929315751, "significant": False},
        12: {"closed_gap": 0.11725847959750967, "significant": False},
        13: {"closed_gap": 0.1099433013635228, "significant": False},
        14: {"closed_gap": 0.08867990277205436, "significant": False},
        15: {"closed_gap": 0.10340198885285182, "significant": False},
        16: {"closed_gap": 0.06705093346587275, "significant": False},
        17: {"closed_gap": 0.09958541939316963, "significant": False},
        18: {"closed_gap": 0.0751802400436478, "significant": False},
        19: {"closed_gap": 0.10164928830691504, "significant": False},
    },

    "algo_features_only": {
        "closed_gap": 0.08352047060255603,
        "significant": False
    },

    "emps_plus_algo": {
        1: {"closed_gap": 0.07506784093645857, "significant": False},
        2: {"closed_gap": 0.047778072846652864, "significant": False},
        3: {"closed_gap": 0.1197082548567849, "significant": False},
        4: {"closed_gap": 0.060728722491474464, "significant": False},
        5: {"closed_gap": 0.04336280985701626, "significant": False},
        6: {"closed_gap": 0.1035194011106506, "significant": False},
        7: {"closed_gap": 0.08601587742496528, "significant": False},
        8: {"closed_gap": 0.061549169613701736, "significant": False},
        9: {"closed_gap": 0.096547372139595, "significant": False},
        10: {"closed_gap": 0.11109342461921129, "significant": False},
        11: {"closed_gap": 0.11544345180082553, "significant": False},
        12: {"closed_gap": 0.09633602479524607, "significant": False},
        13: {"closed_gap": 0.0877931977773681, "significant": False},
        14: {"closed_gap": 0.11010635563510239, "significant": False},
        15: {"closed_gap": 0.1016440625530801, "significant": False},
        16: {"closed_gap": 0.059044485163181566, "significant": False},
        17: {"closed_gap": 0.09509560610463214, "significant": False},
        18: {"closed_gap": 0.09509792740255028, "significant": False},
        19: {"closed_gap": 0.09471776076320705, "significant": False},
    }
}

def find_sbs():
    # Find best (budget, algorithm) in the training set
    best_combination = None
    best_precision = np.inf
    best_precision_test = np.inf
    for budget in BUDGETS:
        for algorithm in ALGORITHMS:
            df_subset = df_prec_train[(df_prec_train["budget"] == budget) & (df_prec_train["algorithm"] == algorithm)]
            df_subset_test = df_prec_test[(df_prec_test["budget"] == budget) & (df_prec_test["algorithm"] == algorithm)]
            if df_subset["precision"].sum() < best_precision:
                best_precision = df_subset["precision"].sum()
                best_precision_test = df_subset_test["precision"].sum()
                best_combination = (budget, algorithm)

    print(f"Best combination in training set: Budget = {best_combination[0]}, Algorithm = {best_combination[1]}, Precision = {best_precision}, Test Precision = {best_precision_test}")

def plot_closed_gap_static_selector():

    pio.kaleido.scope.mathjax = None
    vbs_prec = df_results["vbs_precisions"].sum()
    closed_gaps = {}

    for budget in BUDGETS:
        regret_sum = df_results[f"static_B{budget}"].sum()
        gap = (SBS_prec - regret_sum) / (SBS_prec - vbs_prec)
        closed_gaps[budget] = gap

    x_vals = list(closed_gaps.keys())
    y_vals = list(closed_gaps.values())

    fig = px.bar(
        x=x_vals,
        y=y_vals,
        labels={"x": "A1 budget", "y": "Closed gap"},
    )

    fig.update_traces(marker_color="royalblue",
                      marker_line_color="black",
        marker_line_width=1
    )

    # Axis styling (force black text)
    fig.update_xaxes(
        range=[0, 1050],
        tickmode="array",
        tickvals=x_vals,
        ticktext=[f"{b}" for b in x_vals],
        showline=True,
        linewidth=2,
        linecolor="black",
        tickfont=dict(color="black"),
        title_font=dict(color="black"),
        title_standoff=1.5
    )

    fig.update_yaxes(
        range=[-0.35, 0.1],
        tickfont=dict(color="black"),
        title_font=dict(color="black"),
        title_standoff = 1.5,
        showline=True,
        linewidth=2,
        linecolor="black"

    )

    # Horizontal lines
    fig.add_hline(y=0, line_color="black", line_width=1.5)

    # Vertical subtle lines
    for b in x_vals:
        fig.add_vline(
            x=b,
            line_color="rgba(120,120,120,0.4)",
            line_width=1,
            layer="below"
        )

    # --- Manual annotations ---
    special_budgets = {50, 950, 1000}

    for b, v in zip(x_vals, y_vals):
        if b in special_budgets:
            # place just above x-axis
            fig.add_annotation(
                x=b,
                y=0.01,
                text=f"{v:.3f}",
                showarrow=False,
                font=dict(size=12),
            )
        else:
            if v > 0:
                # place above bar
                fig.add_annotation(
                    x=b,
                    y=v + 0.01,
                    text=f"{v:.3f}",
                    showarrow=False,
                    font=dict(size=12),
                )
            else:
                # place above bar
                fig.add_annotation(
                    x=b,
                    y=v - 0.01,
                    text=f"{v:.3f}",
                    showarrow=False,
                    font=dict(size=12),
                )

    # Tight layout & black title
    fig.update_layout(
        plot_bgcolor="rgb(230,230,230)",
        paper_bgcolor="white",
        bargap=0.15,
        font=dict(
            family="Latin Modern Roman",
            color="black"
        ),
        title_font=dict(
            family="Latin Modern Roman",
            color="black"
        ),
        margin=dict(l=10, r=10, t=10, b=10),
        width=800,
        height=450,
    )

    # fig.update_xaxes(title=None)

    if not os.path.exists("../figures"):
        os.makedirs("../figures")

    fig.write_image("../figures/closed_gaps_static_selectors.pdf")

def plot_closed_gap_selector_b650_obs():
    pio.kaleido.scope.mathjax = None

    # Baseline values
    b650_prec = df_results["static_B650"].sum()
    denom = (SBS_prec - VBS_prec)

    # Closed gaps (same normalization as first plot)
    closed_gap_650 = (SBS_prec - b650_prec) / denom
    closed_gap_selector = (SBS_prec - SELECTOR_prec) / denom
    closed_gap_obs = (SBS_prec - OBS_prec) / denom

    x_vals = ["B650", "D10", "SBO"]
    y_vals = [closed_gap_650, closed_gap_selector, closed_gap_obs]

    fig = px.bar(
        x=x_vals,
        y=y_vals,
        labels={"x": "Method", "y": "Closed Gap"},
    )

    fig.update_traces(marker_color="royalblue",
        marker_line_color="black",
        marker_line_width=1
    )

    # Axis styling
    fig.update_xaxes(
        tickmode="array",
        tickvals=x_vals,
        ticktext=x_vals,
        showline=True,
        linewidth=2,
        linecolor="black",
        tickfont=dict(color="black"),
        title_font=dict(color="black"),
        title_standoff=1.5
    )

    fig.update_yaxes(
        range=[0, 0.7],
        tickfont=dict(color="black"),
        title_font=dict(color="black"),
        title_standoff=1.5,
        showline=True,
        linewidth=2,
        linecolor="black",

    )

    # Horizontal zero line
    fig.add_hline(y=0, line_color="black", line_width=1.5)

    # Subtle vertical lines behind bars
    for x in x_vals:
        fig.add_vline(
            x=x,
            line_color="rgba(120,120,120,0.4)",
            line_width=1,
            layer="below"
        )

    # Value annotations
    for x, v in zip(x_vals, y_vals):
        fig.add_annotation(
            x=x,
            y=(v + 0.015) if v >= 0 else (v - 0.015),
            text=f"{v:.2f}",
            showarrow=False,
            font=dict(size=12),
        )

    # Layout (identical style)
    fig.update_layout(
        width = 600,
        height = 300,
        plot_bgcolor="rgb(245,245,245)",
        paper_bgcolor="white",
        bargap=0.35,
        font=dict(
            family="Latin Modern Roman",
            color="black"
        ),
        title_font=dict(
            family="Latin Modern Roman",
            color="black"
        ),
        margin=dict(l=10, r=10, t=10, b=10),
    )

    if not os.path.exists("../figures"):
        os.makedirs("../figures")

    fig.write_image("../figures/closed_gap_selector_b650_obs.pdf")

def plot_switch_frequency_dynamic_selector():
    pio.kaleido.scope.mathjax = None

    
    total_runs = len(df_results)
    switch_counts = df_results["selector_switch_budget"].value_counts().to_dict()

    switch_freq = {}
    for budget in BUDGETS:
        switch_freq[budget] = switch_counts.get(budget, 0) / total_runs

    x_vals = list(switch_freq.keys())
    y_vals = list(switch_freq.values())

    # --- Assign colors per bar ---
    colors = [
        "firebrick" if (b <= 300 or 750 <= b <= 1000)
        else "royalblue"
        for b in x_vals
    ]

    # --- Create figure ---
    fig = go.Figure()

    # Main bar trace
    fig.add_trace(
        go.Bar(
            x=x_vals,
            y=y_vals,
            marker_color=colors,
            showlegend=False,
        )
    )

    # --- Legend-only dummy traces ---
    fig.add_trace(
        go.Scatter(
            x=[None], y=[None],
            mode="markers",
            marker=dict(size=12, color="firebrick", symbol="square"),
            name="Negative closed gap",
            hoverinfo="skip",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=[None], y=[None],
            mode="markers",
            marker=dict(size=12, color="royalblue", symbol="square"),
            name="Positive closed gap",
            hoverinfo="skip",
        )
    )

    # --- Axes styling ---
    fig.update_xaxes(
        title="Budget",
        range=[0, 1050],
        tickmode="array",
        tickvals=x_vals,
        ticktext=[str(b) for b in x_vals],
        showline=True,
        linewidth=2,
        linecolor="black",
        tickfont=dict(color="black"),
        title_font=dict(color="black"),
    )

    y_max = max(y_vals) if len(y_vals) > 0 else 0.0
    fig.update_yaxes(
        title="Switch Frequency",
        range=[0, 0.15],
        tickfont=dict(color="black"),
        title_font=dict(color="black"),
        title_standoff=1.5,
        tickformat=".0%",   # 0.2 -> 20%
    )

    fig.add_hline(y=0, line_color="black", line_width=1.5)

    for b in x_vals:
        fig.add_vline(
            x=b,
            line_color="rgba(120,120,120,0.4)",
            line_width=1,
            layer="below"
        )

    # --- Value labels above bars ---
    for b, v in zip(x_vals, y_vals):
        fig.add_annotation(
            x=b,
            y=v + 0.005,
            text=f"{100*v:.1f}%",
            showarrow=False,
            font=dict(size=12),
        )

    # --- Layout ---
    fig.update_layout(
        width=600,
        height=300,
        plot_bgcolor="rgb(245,245,245)",
        paper_bgcolor="white",
        bargap=0.15,
        font=dict(family="Latin Modern Roman", color="black"),
        title_font=dict(family="Latin Modern Roman", color="black"),
        margin=dict(l=10, r=10, t=10, b=10),
        legend=dict(
            x=0.01,
            y=0.99,
            xanchor="left",
            yanchor="top",
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="black",
            borderwidth=1,
        ),
    )

    if not os.path.exists("../figures"):
        os.makedirs("../figures")

    fig.write_image("../figures/switch_frequency_dynamic_selector.pdf")

def plot_algo_distribution_by_fid(
    df: pd.DataFrame,
    save_pdf: str | None = None,
    width: int = 880,
    height: int = 480,
    font_family: str = "Latin Modern Roman",
    # algorithm -> color
    algo_colors: dict[str, str] | None = None,
    bar_opacity: float = 0.8,
    bar_width: float = 0.86,
    title: str = "Algorithm Distribution by fid",
    legend_position: str = "top",         # "top" or "bottom"
):
   
    pio.kaleido.scope.mathjax = None
    # default palette
    if algo_colors is None:
        algo_colors = {
            "BFGS": "#1f77b4",
            "Non-elitist": "#ff7f0e",  # CMA-ES non-elitist
            "DE": "#2ca02c",
            "PSO": "#d62728",
            "MLSL": "#9467bd",
            "Elitist": "#8c564b",        # CMA-ES elitist
        }

    # mapping for display names in legend (data keys stay as-is)
    display_name = {
        "Elitist": "CMA-ES, elitist",
        "Non-elitist": "CMA-ES, non-elitist",
    }

    # prep
    df = df.copy()
    df["fid"] = pd.to_numeric(df["fid"], errors="coerce").astype(int)

    # proportions per fid
    counts = df.groupby(["fid", "selector_algorithm"]).size().rename("n").reset_index()
    totals = counts.groupby("fid")["n"].transform("sum")
    counts["prop"] = counts["n"] / totals

    # order: use color dict order but only keep present algos
    present = set(df["selector_algorithm"].unique())
    algos_in_data = [a for a in algo_colors.keys() if a in present]

    fids = sorted(counts["fid"].unique().tolist())
    wide = (
        counts.pivot(index="fid", columns="selector_algorithm", values="prop")
        .reindex(index=fids, columns=algos_in_data)
        .fillna(0.0)
    )

    # build stacked bars
    fig = go.Figure()
    for algo in algos_in_data:
        fig.add_trace(
            go.Bar(
                x=fids,
                y=wide[algo].to_list(),
                name=display_name.get(algo, algo),  # legend label
                width=bar_width,
                marker=dict(color=algo_colors[algo]),
                opacity=bar_opacity,
                hovertemplate=f"fid=%{{x}}<br>{display_name.get(algo, algo)} share=%{{y:.2f}}<extra></extra>",
            )
        )

    # legend placement
    if legend_position.lower() == "bottom":
        legend_y = -0.1
        margins = dict(l=20, r=20, t=20, b=20)
    else:  # "top"
        legend_y = 1.04
        margins = dict(l=20, r=20, t=20, b=20)

    # layout / style
    fig.update_layout(
        #title=dict(text=title, x=0.5, xanchor="center", y=0.95, yanchor="top"),
        width=width, height=height,
        font=dict(family=font_family, size=16, color="black"),
        plot_bgcolor="rgb(230,230,230)",
        paper_bgcolor="white",
        margin=margins,
        barmode="stack",
        bargap=0.1,
        legend=dict(
            orientation="h",
            x=0.5, xanchor="center",
            y=legend_y, yanchor="bottom" if legend_y >= 1 else "top",
            font=dict(size=16, color="black"),
            traceorder="normal",
        ),
    )

    # axes
    fig.update_xaxes(
        title=dict(text="fid", standoff=12),
        tickmode="array",
        tickvals=fids,
        dtick=1,
        range=[min(fids) - 0.5, max(fids) + 0.5],
        showline=True, linecolor="black", linewidth=1,
        showgrid=True, gridcolor="black", gridwidth=0.5,
        zeroline=False, color="black",
        tickfont=dict(size=16),
    )

    fig.update_yaxes(
        title=dict(text="Proportion", standoff=16),
        range=[0, 1],
        showline=True, linecolor="black", linewidth=1,
        showgrid=True, gridcolor="black", gridwidth=0.5,
        zeroline=True, zerolinecolor="black", zerolinewidth=2,
        color="black",
        tickfont=dict(size=16),
    )

    if save_pdf:
        outdir = os.path.dirname(save_pdf)
        if outdir:
            os.makedirs(outdir, exist_ok=True)
        fig.write_image(save_pdf, format="pdf")

    return fig, wide

def plot_gap_closed_by_fid(
    df: pd.DataFrame,
    static_prefix: str = "static_",
    save_pdf: str | None = None,
    width: int = 880,
    height: int = 480,
    font_family: str = "Latin Modern Roman",
    fid_groups: dict[str, list[int]] | None = None,  # NEW: map label -> list of fids
):

    # Which methods to show
    methods = ["selector_precision", "static_B650"]
    pio.kaleido.scope.mathjax = None

    # Helpers
    def _scale_y(val: float) -> float:
        if pd.isna(val):
            return val
        return val if val >= 0 else 0.5 * val

    def _scale_array(vals):
        return [(_scale_y(v) if pd.notna(v) else v) for v in vals]

    # Compute scores (original, unscaled)
    recs = []

    if fid_groups is None:
        # Per-fid mode (original behavior)
        for fid, sub in df.groupby("fid", dropna=False):
            vbs_sum = sub["vbs_precisions"].sum()
            sbs_sum = sub["sbs_precision"].sum()
            den = sbs_sum - vbs_sum
            for col in methods:
                msum = sub[col].sum()
                num = sbs_sum - msum
                if num == den:
                    score = 1.0
                elif den == 0:
                    score = 0.0
                else:
                    score = num / den
                    if fid == 2 and col == "selector_precision":
                        print(f"fid={fid}, num={num}, den={den}, score={score}")

                # Pretty names
                if col == "selector_precision":
                    name = "Dynamic selector"
                elif col == "Non-elitist":
                    name = "CMA-ES, non-elitist"
                elif col == "static_B650":
                    name = "B650"
                else:
                    name = col

                recs.append({"x": fid, "method": name, "fraction_closed": score})

        x_title = "fid"
        # preserve natural numeric ordering for fids
        x_order = list(range(1, int(df["fid"].max()) + 1))

    else:
        # Grouped mode
        # Keep given order of groups
        x_order = list(fid_groups.keys())
        for label, fid_list in fid_groups.items():
            sub = df[df["fid"].isin(fid_list)]
            vbs_sum = sub["vbs_precisions"].sum()
            sbs_sum = sub["sbs_precision"].sum()
            den = sbs_sum - vbs_sum
            for col in methods:
                msum = sub[col].sum()
                num = sbs_sum - msum
                if num == den:
                    score = 1.0
                elif den == 0:
                    score = 0.0
                else:
                    score = num / den

                if col == "selector_precision":
                    name = "Dynamic selector"
                elif col == "Non-elitist":
                    name = "CMA-ES, non-elitist"
                elif col == "static_B650":
                    name = "B650"
                else:
                    name = col

                recs.append({"x": label, "method": name, "fraction_closed": score})

        x_title = "fid group"

    scores = pd.DataFrame(recs).sort_values(["method"], kind="stable").reset_index(drop=True)
    scores["fraction_closed_scaled"] = _scale_array(scores["fraction_closed"])

    # Figure
    fig = go.Figure()
    for method, subm in scores.groupby("method", sort=False):
        # Ensure x follows x_order
        if isinstance(x_order[0], str):
            subm = subm.set_index("x").reindex(x_order).reset_index()
        else:
            subm = subm.sort_values("x")
        fig.add_trace(
            go.Scatter(
                x=subm["x"],
                y=subm["fraction_closed_scaled"],
                customdata=subm["fraction_closed"],
                mode="lines+markers",
                name=method,
                marker=dict(size=7, line=dict(width=0.5, color="black")),
                line=dict(width=2),
                hovertemplate=f"{x_title}=%{{x}}<br>fraction=%{{customdata:.4f}}<extra>{method}</extra>",
            )
        )

    # Reference lines
    fig.add_hline(y=0, line_dash="solid", line_width=1.2, line_color="black")
    fig.add_hline(y=1, line_dash="dot", line_width=1, line_color="black")

    # Y-axis ticks
    original_tick_vals = [-5, -4, -3, -2, -1, 0, 0.5, 1.0]
    scaled_tick_vals = [_scale_y(v) for v in original_tick_vals]
    tick_text = [str(v).rstrip("0").rstrip(".") if isinstance(v, float) else str(v) for v in original_tick_vals]

    # Layout
    fig.update_layout(
        # title=dict(
        #     text="Fraction of the Gap Closed" + ("" if fid_groups is None else " (grouped)"),
        #     x=0.5, xanchor="center", y=0.95, yanchor="top"
        # ),
        width=width,
        height=height,
        font=dict(family=font_family, size=16, color="black"),
        plot_bgcolor="rgb(230,230,230)",
        paper_bgcolor="white",
        margin=dict(l=60, r=30, t=60, b=50),
        legend=dict(
            orientation="h",
            x=0.98, xanchor="right",
            y=0.02, yanchor="bottom",  
            bgcolor="rgba(255,255,255,0.6)",  
            bordercolor="black",
            borderwidth=1,
            font=dict(size=16, color="black"),
        ),
    )

    # X-axis (handle numeric fids or categorical group labels)
    if fid_groups is None:
        fig.update_xaxes(
            title=dict(text=x_title, standoff=12),
            tickmode="array",
            tickvals=x_order,
            range=[min(x_order) - 0.5, max(x_order) + 0.5],
            dtick=1,
            showline=True, linecolor="black", linewidth=1,
            showgrid=True, gridcolor="black", gridwidth=0.5,
            zeroline=False, color="black",
            tickfont=dict(size=16),
        )
    else:
        fig.update_xaxes(
            title=dict(text=x_title, standoff=12),
            type="category",
            categoryorder="array",
            categoryarray=x_order,
            showline=True, linecolor="black", linewidth=1,
            showgrid=True, gridcolor="black", gridwidth=0.5,
            zeroline=False, color="black",
            tickfont=dict(size=16),
        )

    # Y-axis
    y_scaled_min = _scale_y(-5)
    y_scaled_max = 1.1
    fig.update_yaxes(
        title=dict(text="Fraction of the gap closed", standoff=16),
        range=[y_scaled_min, y_scaled_max],
        tickmode="array",
        tickvals=scaled_tick_vals,
        ticktext=tick_text,
        showline=True, linecolor="black", linewidth=1,
        showgrid=True, gridcolor="black", gridwidth=0.5,
        zeroline=True, zerolinecolor="black", zerolinewidth=2,
        color="black",
        tickfont=dict(size=16),
    )

    # --- Out-of-bounds indicator for fid=2, Dynamic selector ---
    x_oob = 2
    y_oob = y_scaled_min + 0.08  # a bit above the bottom edge (tune if you want)

    # small rectangle
    fig.add_shape(
        type="rect",
        xref="x",
        yref="y",
        x0=x_oob - 0.45,
        x1=x_oob + 0.45,
        y0=y_oob - 0.08,
        y1=y_oob + 0.08,
        line=dict(color="black", width=1),
        fillcolor="rgba(255,255,255,0.9)",
        layer="above",
    )

    # text inside
    fig.add_annotation(
        x=x_oob,
        y=y_oob,
        text="-117",
        showarrow=False,
        font=dict(size=14, color="black"),
        xanchor="center",
        yanchor="middle",
    )

    # Enforce legend order
    fig.update_traces(selector=dict(name="Dynamic selector"), legendrank=1)
    fig.update_traces(selector=dict(name="CMA-ES, non-elitist"), legendrank=2)
    fig.update_traces(selector=dict(name="B650"), legendrank=3)

    if save_pdf:
        outdir = os.path.dirname(save_pdf)
        if outdir:
            os.makedirs(outdir, exist_ok=True)
        fig.write_image(save_pdf, format="pdf")

    return fig, scores

def plot_closed_gap_emps(results, out_path="../figures/closed_gaps_epms.pdf"):
    pio.kaleido.scope.mathjax = None

    emps_data = results["emps"]

    x_vals = sorted(emps_data.keys())  # 0 ... 19
    y_vals = [emps_data[t]["closed_gap"] for t in x_vals]

    # Color first column in firebrich, else royalblue
    colors = ["firebrick"] + ["royalblue" for _ in range(1, len(x_vals))]

    fig = px.bar(
        x=x_vals,
        y=y_vals,
        labels={"x": "Lookahead horizon", "y": "Closed gap"},
    )

    fig.update_traces(
        marker_color=colors,
        marker_line_color="black",
        marker_line_width=1
    )

    fig.update_xaxes(
        range=[-1.5, 19.5],
        tickmode="array",
        tickvals=x_vals,
        ticktext = [
            "∅" if x < 0 else f"{x}"
            for x in x_vals
        ],
        showline=True,
        linewidth=2,
        linecolor="black",
        tickfont=dict(color="black"),
        title_font=dict(color="black"),
        title_standoff=1.5
    )

    fig.update_yaxes(
        range=[0, 0.16],
        tickfont=dict(color="black"),
        title_font=dict(color="black"),
        title_standoff=1.5,
        showline=True,
        linewidth=2,
        linecolor="black",
    )

    fig.add_hline(y=0, line_color="black", line_width=1.5)

    for x in x_vals:
        fig.add_vline(
            x=x,
            line_color="rgba(120,120,120,0.4)",
            line_width=1,
            layer="below"
        )

    for x, y in zip(x_vals, y_vals):
        if y >= 0:
            ann_y = y + 0.004
        else:
            ann_y = y - 0.015

        fig.add_annotation(
            x=x,
            y=ann_y,
            text=f"{y:.3f}",
            showarrow=False,
            font=dict(size=12),
        )

    fig.update_layout(
        plot_bgcolor="rgb(230,230,230)",
        paper_bgcolor="white",
        bargap=0.15,
        font=dict(
            family="Latin Modern Roman",
            color="black"
        ),
        title_font=dict(
            family="Latin Modern Roman",
            color="black"
        ),
        margin=dict(l=10, r=10, t=10, b=10),
        width = 800,
        height = 450,
    )
   

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    print(f"Saving closed gap EMPs figure to {out_path}")
    fig.write_image(out_path)


def plot_closed_gap_emps_plus_algo(results, out_path="../figures/closed_gaps_epms_plus_algo.jpg"):
    pio.kaleido.scope.mathjax = None

    plus_data = results["emps_plus_algo"]
    algo_only = results["algo_features_only"]

    x_vals = ["just_algo"] + list(sorted(plus_data.keys()))   # ["just_algo", 1, 2, ..., 19]
    y_vals = [algo_only["closed_gap"]] + [plus_data[t]["closed_gap"] for t in sorted(plus_data.keys())]
    significant = [algo_only["significant"]] + [plus_data[t]["significant"] for t in sorted(plus_data.keys())]

    colors = ["firebrick" if sig else "royalblue" for sig in significant]

    # use numeric positions so categorical spacing is fully controlled
    positions = list(range(len(x_vals)))

    fig = px.bar(
        x=positions,
        y=y_vals,
        labels={"x": "Lookahead horizon", "y": "Closed Gap"},
    )

    fig.update_traces(marker_color=colors)

    fig.update_xaxes(
        range=[-0.5, len(x_vals) - 0.5],
        tickmode="array",
        tickvals=positions,
        ticktext=[str(x) for x in x_vals],
        showline=True,
        linewidth=2,
        linecolor="black",
        tickfont=dict(color="black"),
        title_font=dict(color="black"),
    )

    fig.update_yaxes(
        range = [0, 0.15],
        tickfont=dict(color="black"),
        title_font=dict(color="black"),
        title_standoff=1.5,
    )

    fig.add_hline(y=0, line_color="black", line_width=1.5)

    for pos in positions:
        fig.add_vline(
            x=pos,
            line_color="rgba(120,120,120,0.4)",
            line_width=1,
            layer="below"
        )

    for pos, y in zip(positions, y_vals):
        if y >= 0:
            ann_y = y + 0.0075
        else:
            ann_y = y - 0.015

        fig.add_annotation(
            x=pos,
            y=ann_y,
            text=f"{y:.3f}",
            showarrow=False,
            font=dict(size=12),
        )

    fig.update_layout(
        plot_bgcolor="rgb(245,245,245)",
        paper_bgcolor="white",
        bargap=0.15,
        font=dict(
            family="Latin Modern Roman",
            color="black"
        ),
        title_font=dict(
            family="Latin Modern Roman",
            color="black"
        ),
        margin=dict(l=10, r=10, t=10, b=10),
    )

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.write_image(out_path)

def plot_closed_gap_d10_b650_sbs_per_instance():
    pio.kaleido.scope.mathjax = None

    # Baseline values
    b650_prec = df_results["static_B650"].sum()
    b150_prec = df_results["static_B150"].sum()
    d10_prec = SELECTOR_prec          
    sbs_prec = SBS_prec             
    per_instance = per_instance_prec 

    denom = (SBS_prec - VBS_prec)

    # Closed gaps
    closed_gap_d10 = (SBS_prec - d10_prec) / denom
    closed_gap_b650 = (SBS_prec - b650_prec) / denom
    closed_gap_b150 = (SBS_prec - b150_prec) / denom
    closed_gap_sbs = (SBS_prec - sbs_prec) / denom
    closed_gap_per_instance = (SBS_prec - per_instance) / denom

    print(
        closed_gap_d10,
        closed_gap_b650,
        closed_gap_b150,
        closed_gap_sbs,
        closed_gap_per_instance
    )

    x_vals = ["Dynamic Sel. (ours)", "B650 (Best static)", "B150 (Kostovska et al.)", "Per Instance"]
    y_vals = [
        closed_gap_d10,
        closed_gap_b650,
        closed_gap_b150,
        closed_gap_per_instance,
    ]

    fig = px.bar(
        x=x_vals,
        y=y_vals,
        labels={"x": "", "y": "Closed gap"},
    )

    fig.update_traces(
        marker_color="royalblue",
        marker_line_color="black",
        marker_line_width=1
    )

    # Axis styling
    fig.update_xaxes(
        tickmode="array",
        tickvals=x_vals,
        ticktext=[
            "Dynamic Selector<br><span style='line-height:0.6'> (Ours)</span>",
            "Best static",
            "Kostovska et al.",
            "Per Instance"
        ],
        showline=True,
        linewidth=2,
        linecolor="black",
        tickfont=dict(color="black"),
        title_font=dict(color="black"),
        title_standoff=1.5
    )

    fig.update_yaxes(
        range=[-0.1, 0.2],
        tickfont=dict(color="black"),
        title_font=dict(color="black"),
        title_standoff=1.5,
        showline=True,
        linewidth=2,
        linecolor="black",
    )

    fig.update_xaxes(title=None)

    # Horizontal zero line
    fig.add_hline(y=0, line_color="black", line_width=1.5)

    # Subtle vertical lines behind bars
    for x in x_vals:
        fig.add_vline(
            x=x,
            line_color="rgba(120,120,120,0.4)",
            line_width=1,
            layer="below"
        )

    # Value annotations (3 decimals + special placement for Per Instance)
    for x, v in zip(x_vals, y_vals):

        if x == "Per Instance":
            y_pos = 0.01  # slightly above x-axis
        else:
            y_pos = (v + 0.015) if v >= 0 else (v - 0.015)

        fig.add_annotation(
            x=x,
            y=y_pos,
            text=f"{v:.3f}",
            showarrow=False,
            font=dict(size=12),
        )

    # Layout
    fig.update_layout(
        width=600,
        height=300,
        plot_bgcolor="rgb(230,230,230)",
        paper_bgcolor="white",
        bargap=0.35,
        font=dict(
            family="Latin Modern Roman",
            color="black"
        ),
        title_font=dict(
            family="Latin Modern Roman",
            color="black"
        ),
        margin=dict(l=10, r=10, t=10, b=10),
    )

    if not os.path.exists("../figures"):
        os.makedirs("../figures")

    fig.write_image("../figures/closed_gap_d10_b650_sbs_per_instance.pdf")

if __name__ == "__main__":
    # df = pd.read_csv("../data/selector_results_with_lookahead_all_epms_10.csv")
    # print(df["selector_precision"].sum())

    # df = pd.read_csv("../data/selector_results_with_lookahead_all_epms_10_sbs.csv")

    # for fid in sorted(df["fid"].unique()):
    #     sub = df[df["fid"] == fid]
    #     print(f"fid={fid}, SBS sum={sub['sbs_precision'].sum():.16f}, VBS sum={sub['vbs_precisions'].sum():.16f}, Selector sum={sub['selector_precision'].sum():.16f}, Best static sum={sub['static_B650'].sum():.16f}")

    # print(f"Overall SBS sum={df['sbs_precision'].sum():.16f}, VBS sum={df['vbs_precisions'].sum():.16f}, Selector sum={df['selector_precision'].sum():.16f}, Best static sum={df['static_B650'].sum():.16f}")

    plot_closed_gap_emps(results)