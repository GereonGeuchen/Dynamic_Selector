import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
import os
from scipy.stats import permutation_test

def plot_selector_dashboard(
    df: pd.DataFrame,
    df_algos: pd.DataFrame,
    save_pdf: str | None = "selector_dashboard.pdf",
    width: int = 980,
    height: int = 980,
    font_family: str = "Latin Modern Roman",
    row_heights: tuple[float, float, float] = (0.25, 0.33, 0.33),
    vertical_spacing: float = 0.06,
):
    import os
    import pandas as pd
    import plotly.graph_objects as go
    import plotly.io as pio
    from plotly.subplots import make_subplots

    pio.kaleido.scope.mathjax = None
    fids = sorted(df["fid"].dropna().astype(int).unique().tolist())

    # Top subplot: fraction of the gap closed
    methods = ["selector_precision", "static_B650"]

    def _scale_y(v: float) -> float:
        # if pd.isna(v):
        #     return v
        # return v if v >= 0 else 0.5 * v
        return v

    recs = []
    for fid, sub in df.groupby("fid", dropna=False):
        vbs_sum = sub["vbs_precisions"].sum()
        sbs_sum = sub["sbs_precision"].sum()
        den = sbs_sum - vbs_sum
        for col in methods:
            if col not in sub.columns:
                continue
            msum = sub[col].sum()
            num = sbs_sum - msum
            score = 1.0 if num == den else (0.0 if den == 0 else num / den)
            name = "Dynamic selector (ours)" if col == "selector_precision" else "Best static"
            recs.append({"fid": int(fid), "method": name, "fraction": float(score)})

    top_df = pd.DataFrame(recs).sort_values(["method", "fid"])
    print(top_df)
    top_df["fraction_scaled"] = top_df["fraction"].map(_scale_y)

    # Middle subplot: algorithm distribution
    algo_colors = {
        "BFGS": "#1f77b4",
        "Non-elitist": "#ff7f0e",
        "DE": "#2ca02c",
        "PSO": "#d62728",
        "MLSL": "#9467bd",
        "Elitist": "#8c564b",
    }
    display_name = {
        "Elitist": "CMA-ES, elitist",
        "Non-elitist": "CMA-ES, non-elitist",
    }

    df_alg = df_algos.copy()
    df_alg["fid"] = pd.to_numeric(df_alg["fid"], errors="coerce").astype(int)

    counts = df_alg.groupby(["fid", "selector_algorithm"]).size().rename("n").reset_index()
    totals = counts.groupby("fid")["n"].transform("sum")
    counts["prop"] = counts["n"] / totals

    algos_in_data = [a for a in algo_colors if a in set(counts["selector_algorithm"].unique())]
    mid_wide = (
        counts.pivot(index="fid", columns="selector_algorithm", values="prop")
        .reindex(index=fids, columns=algos_in_data)
        .fillna(0.0)
    )

    # Bottom subplot: switching budget
    stats = (
        df.groupby("fid", as_index=False)["selector_switch_budget"]
        .agg(mean="mean", std="std")
        .sort_values("fid")
    )
    stats["fid"] = stats["fid"].astype(int)
    stats = stats.set_index("fid").reindex(fids).reset_index()
    stats["std"] = stats["std"].fillna(0.0)

    mean_y = stats["mean"].to_numpy(dtype=float)
    std_y = stats["std"].to_numpy(dtype=float)
    lower = (mean_y - std_y).tolist()
    upper = (mean_y + std_y).tolist()

    # Build combined figure
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=vertical_spacing,
        row_heights=list(row_heights),
    )

    # Top axis setup
    top_tick_orig = [-0.5, 0, 0.5, 1.0]
    top_tick_scaled = top_tick_orig
    top_tick_text = ["-0.5", "0", "0.5", "1"]

    top_ymin = -0.5
    top_ymax = 1.1

    # Top traces: keep true line, do NOT clip
    for method, subm in top_df.groupby("method", sort=False):
        subm = subm.set_index("fid").reindex(fids).reset_index()

        fig.add_trace(
            go.Scatter(
                x=subm["fid"],
                y=subm["fraction_scaled"],
                customdata=subm["fraction"],
                mode="markers",
                name=method,
                marker=dict(size=11, line=dict(width=0.8, color="black")),
                hovertemplate="fid=%{x}<br>fraction=%{customdata:.4f}<extra>" + method + "</extra>",
                legendgroup="top",
                legend="legend",
                cliponaxis=True,
            ),
            row=1,
            col=1,
        )

    # Add annotation boxes for out-of-bounds points in top plot
    for method, subm in top_df.groupby("method", sort=False):
        subm = subm.set_index("fid").reindex(fids).reset_index()
        xshift = -18 if method == "Dynamic selector" else 18

        for _, r in subm.iterrows():
            true_val = r["fraction"]
            scaled_val = r["fraction_scaled"]

            if pd.isna(true_val) or pd.isna(scaled_val):
                continue

            label = f"{true_val:.0f}" if abs(true_val) >= 10 else f"{true_val:.2f}"

            if scaled_val < top_ymin:
                if scaled_val < top_ymin:
                    fig.add_annotation(
                        x=r["fid"],
                        y=top_ymin,
                        text=label,
                        xanchor="center",
                        yanchor="top",
                        xshift=xshift + -19,   # ← move left
                        yshift=10,             # ← move up
                        showarrow=True,
                        arrowhead=0,
                        arrowsize=1,
                        arrowwidth=1,
                        arrowcolor="black",
                        ax=0,
                        ay=-12,
                        font=dict(size=12, color="#EF553B"),
                        bgcolor="white",
                        bordercolor="black",
                        borderwidth=1,
                        row=1,
                        col=1,
                    )
            elif scaled_val > top_ymax:
                fig.add_annotation(
                    x=r["fid"],
                    y=top_ymax,
                    text=label,
                    xanchor="center",
                    yanchor="bottom",
                    xshift=xshift+1,
                    yshift=0,
                    showarrow=True,
                    arrowhead=0,
                    arrowsize=1,
                    arrowwidth=1,
                    arrowcolor="black",
                    ax=0,
                    ay=12,
                    font=dict(size=12, color="#EF553B"),
                    bgcolor="white",
                    bordercolor="black",
                    borderwidth=1,
                    row=1,
                    col=1,
                )

    fig.add_hline(
        y=0,
        line_dash="solid",
        line_width=2.0,
        line_color="black",
        row=1,
        col=1,
        layer="below",
    )
    # fig.add_hline(
    #     y=1,
    #     line_dash="dot",
    #     line_width=1.0,
    #     line_color="black",
    #     row=1,
    #     col=1,
    # )

    fig.update_yaxes(
        title_text="Closed gap",
        tickmode="array",
        tickvals=top_tick_scaled,
        ticktext=top_tick_text,
        range=[top_ymin, top_ymax],
        row=1,
        col=1,
        gridcolor="black",
    )
    fig.update_yaxes(zeroline=False, row=1, col=1)

    # Middle traces
    for algo in algos_in_data:
        fig.add_trace(
            go.Bar(
                x=fids,
                y=mid_wide[algo].tolist(),
                name=display_name.get(algo, algo),
                marker=dict(color=algo_colors[algo]),
                width=0.86,
                opacity=0.9,
                legendgroup="middle",
                legend="legend2",
            ),
            row=2,
            col=1,
        )
    fig.update_yaxes(
        title_text="Proportion of selected algorithms",
        range=[0, 1],
        row=2,
        col=1,
        gridcolor="black",
    )

    # Bottom traces
    # fig.add_trace(
    #     go.Scatter(
    #         x=fids + fids[::-1],
    #         y=upper + lower[::-1],
    #         fill="toself",
    #         fillcolor="rgba(65,105,225,0.35)",
    #         line=dict(color="rgba(0,0,0,0)"),
    #         hoverinfo="skip",
    #         name="±1σ",
    #         legendgroup="bottom",
    #         legend="legend3",
    #     ),
    #     row=3,
    #     col=1,
    # )
    fig.add_trace(
        go.Scatter(
            x=fids,
            y=mean_y,
            mode="markers",
            name="Mean switch budget",
            marker=dict(
                size=12,
                color="royalblue",
                line=dict(width=0.9, color="black")
            ),
            error_y=dict(
                type="data",
                array=std_y,          # +σ
                arrayminus=std_y,     # -σ (symmetric)
                visible=True,
                thickness=2.5,       # vertical line thickness
                width=6,             # ← THIS controls cap width (horizontal lines)
                color="black",
            ),
            hovertemplate="fid=%{x}<br>mean=%{y:.2f}<br>std=%{error_y.array:.2f}<extra></extra>",
            legendgroup="bottom",
            legend="legend3",
        ),
        row=3,
        col=1,
    )
    
    fig.update_yaxes(
        title_text="Switching budget",
        row=3,
        col=1,
        gridcolor="black",
        zerolinecolor="black",
        zerolinewidth=1.5,
        range=[0, 1050],
        title_standoff=5
    )

    # X-axes: ticks on all rows, label only bottom
    common_x = dict(
        tickmode="array",
        tickvals=fids,
        dtick=1,
        range=[min(fids) - 0.5, max(fids) + 0.5],
        showline=True,
        linecolor="black",
        linewidth=1,
        showgrid=True,
        gridcolor="black",
        gridwidth=0.5,
        zeroline=False,
    )
    fig.update_xaxes(row=1, col=1, **{**common_x, "showticklabels": True, "title_text": None})
    fig.update_xaxes(row=2, col=1, **{**common_x, "showticklabels": True, "title_text": None})
    fig.update_xaxes(
        row=3,
        col=1,
        **common_x,
        title_text="BBOB function",
        title_standoff=12,
        showticklabels=True,
    )

    # Legend positioning helpers
    h1, h2, h3 = row_heights
    s = vertical_spacing
    top_bottom = 1 - h1
    mid_top = top_bottom - s
    gap_mid = (top_bottom + mid_top) / 2.0 - 0.09

    fig.update_layout(
        width=width,
        height=height,
        font=dict(family=font_family, size=16, color="black"),
        paper_bgcolor="white",
        plot_bgcolor="rgb(230,230,230)",
        margin=dict(l=12, r=12, t=12, b=12),
        barmode="stack",

        legend=dict(
            x=0.98,
            y=top_bottom + 0.015,
            xanchor="right",
            yanchor="bottom",
            orientation="h",
            bgcolor="rgba(255,255,255,0.6)",
            bordercolor="black",
            borderwidth=1,
            font=dict(size=14),
            traceorder="normal",
        ),

        legend2=dict(
            x=0.5,
            y=gap_mid + 0.017,
            xanchor="center",
            yanchor="middle",
            orientation="h",
            bgcolor="rgba(255,255,255,0.4)",
            bordercolor="black",
            borderwidth=1,
            font=dict(size=14),
            traceorder="normal",
        ),

        legend3=dict(
            x=0.02,
            y=0.05,
            xanchor="left",
            yanchor="top",
            orientation="v",
            bgcolor="rgba(255,255,255,0.6)",
            bordercolor="black",
            borderwidth=1,
            font=dict(size=14),
            traceorder="normal",
        ),
    )

    if save_pdf:
        outdir = os.path.dirname(save_pdf)
        if outdir:
            os.makedirs(outdir, exist_ok=True)
        fig.write_image(save_pdf, format="pdf")

    return fig

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
import os

def plot_selector_dashboard_with_convergence(
    df: pd.DataFrame,
    df_algos: pd.DataFrame,
    top_fids: tuple[int, int] = (1, 12),
    save_pdf: str | None = "selector_dashboard_with_convergence.pdf",
    width: int = 1200,
    height: int = 980,
    font_family: str = "Latin Modern Roman",
    row_heights: tuple[float, float, float] = (0.33, 0.29, 0.29),
    vertical_spacing: float = 0.06,
    convergence_base_path: str = "../data/convergence_plot_data",
    selector_results_path: str = "../data/selector_performance_data/selector_results_with_lookahead_all_epms_10.csv",
):
    pio.kaleido.scope.mathjax = None

    # ----------------------------
    # Inputs / validation
    # ----------------------------
    if len(top_fids) != 2:
        raise ValueError("top_fids must contain exactly two function ids, e.g. (3, 12).")

    top_fid_left, top_fid_right = int(top_fids[0]), int(top_fids[1])
    fids = sorted(df["fid"].dropna().astype(int).unique().tolist())

    # ----------------------------
    # Load data
    # ----------------------------
    conv_df = pd.read_csv(f"{convergence_base_path}/Selector_mean_per_fid.csv")
    conv_a1 = pd.read_csv(f"{convergence_base_path}/A1_mean_per_fid.csv")
    conv_sbs = pd.read_csv(f"{convergence_base_path}/SBS_mean_per_fid.csv")
    conv_b150 = pd.read_csv(f"{convergence_base_path}/B150_mean_per_fid.csv")
    df_res = pd.read_csv(selector_results_path)

    for _df in (conv_df, conv_a1, conv_sbs, conv_b150):
        _df["eval"] = _df["eval"].astype(int)

    # ----------------------------
    # Helpers for convergence
    # ----------------------------
    def get_y_range(fid: int):
        sub_list = [
            conv_df[conv_df["fid"] == fid],
            conv_a1[conv_a1["fid"] == fid],
            conv_sbs[conv_sbs["fid"] == fid],
            conv_b150[conv_b150["fid"] == fid]
        ]
        upper_candidates = []
        lower_candidates = []
        for s in sub_list:
            mask = (s["mean_raw_y"] > 0)
            if fid != 12:
                mask &= (s["eval"] >= 150)
            upper_candidates.extend(s.loc[mask, "mean_raw_y"].tolist())
            lower_candidates.extend(s.loc[s["mean_raw_y"] > 0, "mean_raw_y"].tolist())
        
        if not upper_candidates or not lower_candidates: return None
        y_max, y_min = max(upper_candidates), min(lower_candidates)
        if y_min >= y_max: y_min = y_max / 10.0
        return [np.log10(y_min), np.log10(y_max)]

    def get_switch_stats(fid: int):
        vals = df_res.loc[df_res["fid"] == fid, "selector_switch_budget"].dropna()
        if len(vals) == 0: return None, None, None
        return vals.mean(), vals.quantile(0.25), vals.quantile(0.75)

    def add_convergence_subplot(fig, fid: int, row: int, col: int, showlegend: bool):
        sub_list = [
            (conv_df[conv_df["fid"] == fid], "Dynamic selector", "royalblue"),
            (conv_a1[conv_a1["fid"] == fid], "A1", "firebrick"),
            (conv_sbs[conv_sbs["fid"] == fid], "SBS", "darkgreen"),
            (conv_b150[conv_b150["fid"] == fid], "Kostovska et al.", "darkorange")
        ]

        for d, name, color in sub_list:
            fig.add_trace(
                go.Scatter(
                    x=d["eval"], y=d["mean_raw_y"], mode="lines",
                    line=dict(color=color), name=name,
                    showlegend=showlegend, legendgroup="conv", legend="legend"
                ),
                row=row, col=col
            )

        m, q1, q3 = get_switch_stats(fid)
        if m is not None:
            fig.add_vline(x=m, line=dict(color="black", width=2), row=row, col=col)
            for q in [q1, q3]:
                fig.add_vline(x=q, line=dict(color="black", width=2, dash="dot"), row=row, col=col)

        fig.update_xaxes(
            range=[0, 1000] if fid == 12 else [150, 1000], showline=True, linewidth=1.2,
            linecolor="black", ticks="outside", showgrid=True, gridcolor="white",
            tickfont=dict(size=14, family=font_family, color="black"), row=row, col=col
        )
        fig.update_yaxes(
            type="log", range=get_y_range(fid), showline=True, linewidth=1.2,
            linecolor="black", ticks="outside", tickformat=".0e", showgrid=True,
            gridcolor="white", tickfont=dict(size=14, family=font_family, color="black"),
            nticks=5, row=row, col=col
        )

    # ----------------------------
    # Middle/Bottom Data Prep
    # ----------------------------
    algo_colors = {"BFGS": "#1f77b4", "Non-elitist": "#ff7f0e", "DE": "#2ca02c", 
                   "PSO": "#d62728", "MLSL": "#9467bd", "Elitist": "#8c564b"}
    display_name = {"Elitist": "CMA-ES, elitist", "Non-elitist": "CMA-ES, non-elitist"}

    df_alg = df_algos.copy()
    df_alg["fid"] = pd.to_numeric(df_alg["fid"], errors="coerce").astype(int)
    counts = df_alg.groupby(["fid", "selector_algorithm"]).size().rename("n").reset_index()
    mid_wide = counts.pivot(index="fid", columns="selector_algorithm", values="n").fillna(0)
    mid_wide = mid_wide.div(mid_wide.sum(axis=1), axis=0).reindex(index=fids).fillna(0)

    # ----------------------------
    # Build Figure
    # ----------------------------
    fig = make_subplots(
        rows=3, cols=2,
        specs=[[{}, {}], [{"colspan": 2}, None], [{"colspan": 2}, None]],
        row_heights=list(row_heights), vertical_spacing=vertical_spacing,
        horizontal_spacing=0.08, subplot_titles=[f"f{top_fid_left}", f"f{top_fid_right}", "", ""]
    )

    # Row 1
    add_convergence_subplot(fig, top_fid_left, 1, 1, True)
    add_convergence_subplot(fig, top_fid_right, 1, 2, False)

    # Row 2
    for algo in [a for a in algo_colors if a in mid_wide.columns]:
        fig.add_trace(
            go.Bar(
                x=fids, y=mid_wide[algo], name=display_name.get(algo, algo),
                marker=dict(color=algo_colors[algo]), width=0.86, opacity=0.9,
                legendgroup="middle", legend="legend2"
            ),
            row=2, col=1
        )

    # Row 3 (Boxplots)
    for i, fid in enumerate(fids):
        vals = df.loc[df["fid"] == fid, "selector_switch_budget"].dropna()
        fig.add_trace(
            go.Box(
                y=vals, x=[fid]*len(vals), name="Switching Budget",
                boxpoints=False, width=0.55, line=dict(color="black", width=2.5),
                fillcolor="royalblue", opacity=0.75, marker=dict(color="royalblue"),
                showlegend=False,
                hovertemplate=f"fid={fid}<br>switch=%{{y:.2f}}<extra></extra>"
            ),
            row=3, col=1
        )

    # ----------------------------
    # Final Layout & Styling
    # ----------------------------
    common_x = dict(tickmode="array", tickvals=fids, range=[min(fids)-0.5, max(fids)+0.5], 
                    showline=True, linecolor="black", linewidth=1, showgrid=True, 
                    gridcolor="black", gridwidth=0.5, zeroline=False)

    fig.update_xaxes(row=2, col=1, **{**common_x, "showticklabels": True})
    fig.update_xaxes(row=3, col=1, title_text="BBOB function", title_standoff=12, **common_x)
    
    fig.update_yaxes(title_text="Proportion of selected algorithms", range=[0, 1], gridcolor="black", row=2, col=1)
    fig.update_yaxes(title_text="Switching budget", range=[0, 1050], gridcolor="black", zerolinecolor="black", row=3, col=1)

    # Annotations
    fig.add_annotation(text="Evaluations", x=0.205, y=0.772, xref="paper", yref="paper", showarrow=False, font=dict(size=18, family=font_family))
    fig.add_annotation(text="Evaluations", x=0.825, y=0.772, xref="paper", yref="paper", showarrow=False, font=dict(size=18, family=font_family))
    fig.add_annotation(text="Mean regret", x=-0.005, y=0.95, xref="paper", yref="paper", xshift=-55, textangle=-90, showarrow=False, font=dict(size=18, family=font_family))

    fig.update_layout(
        width=width, height=height, barmode="stack",
        font=dict(family=font_family, size=16, color="black"),
        paper_bgcolor="white", plot_bgcolor="rgb(230,230,230)",
        margin=dict(l=70, r=20, t=40, b=40),
        legend=dict(x=0.98, y=0.98, xanchor="right", orientation="h", bgcolor="rgba(255,255,255,0.7)", bordercolor="black", borderwidth=1, font=dict(size=14)),
        legend2=dict(x=0.4575, y=0.7, xanchor="center", orientation="h", bgcolor="rgba(255,255,255,0.3)", bordercolor="black", borderwidth=1, font=dict(size=14)),
        #legend3=dict(x=0.02, y=0.06, xanchor="left", orientation="v", bgcolor="rgba(255,255,255,0.7)", bordercolor="black", borderwidth=1, font=dict(size=14))
    )

    if save_pdf:
        fig.write_image(save_pdf, format="pdf")

    return fig

if __name__ == "__main__":
    df = pd.read_csv("../data/selector_performance_data/selector_results_with_lookahead_all_epms_10_sbs.csv")
    df_algos = pd.read_csv("../data/selector_performance_data/selector_results_with_lookahead_all_epms_10_sbs.csv")
    df_prec = pd.read_csv("../data/A2_precisions_test.csv")
    plot_selector_dashboard_with_convergence(df, df_algos, top_fids=(11, 15), row_heights=(0.2,0.33,0.33), height=1000, vertical_spacing=0.06)

    # add sbs_precision column to df_algos
    # for each (fid,iid,rep), it is the precision of (BFGS, 650)
    # df_algos["sbs_precision"] = np.nan
    # for (fid, iid, rep), sub in df_prec.groupby(["fid", "iid", "rep"]):
    #     df_algos.loc[(df_algos["fid"] == fid) & (df_algos["iid"] == iid) & (df_algos["rep"] == rep), "sbs_precision"] = sub.loc[(sub["algorithm"] == "BFGS") & (sub["budget"] == 650), "precision"].values[0]

    #df_algos.to_csv("../data/selector_results_with_lookahead_all_epms_10_sbs.csv", index=False)

    # print(pd.read_csv("../data/selector_results_with_lookahead_all_epms_10_sbs.csv")["sbs_precision"].sum())