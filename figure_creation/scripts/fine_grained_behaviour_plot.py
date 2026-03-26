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

if __name__ == "__main__":
    df = pd.read_csv("../data/selector_results_with_lookahead_all_epms_10_sbs.csv")
    df_algos = pd.read_csv("../data/selector_results_with_lookahead_all_epms_10_sbs.csv")
    df_prec = pd.read_csv("../data/A2_precisions_test.csv")
    plot_selector_dashboard(df, df_algos, save_pdf="../figures/selector_dashboard.pdf")

    # add sbs_precision column to df_algos
    # for each (fid,iid,rep), it is the precision of (BFGS, 650)
    # df_algos["sbs_precision"] = np.nan
    # for (fid, iid, rep), sub in df_prec.groupby(["fid", "iid", "rep"]):
    #     df_algos.loc[(df_algos["fid"] == fid) & (df_algos["iid"] == iid) & (df_algos["rep"] == rep), "sbs_precision"] = sub.loc[(sub["algorithm"] == "BFGS") & (sub["budget"] == 650), "precision"].values[0]

    #df_algos.to_csv("../data/selector_results_with_lookahead_all_epms_10_sbs.csv", index=False)

    # print(pd.read_csv("../data/selector_results_with_lookahead_all_epms_10_sbs.csv")["sbs_precision"].sum())