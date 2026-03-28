import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
import os
import math

import pandas as pd

def aggregate_runs():
    # Load file
    df = pd.read_csv("../data/A2_run_data_test/A2_BFGS_B650_5D.csv")

    # Make sure eval is integer
    df["eval"] = df["eval"].astype(int)

    # Keep only relevant columns
    df = df[["fid", "iid", "rep", "eval", "raw_y"]].copy()

    # Sort properly
    df = df.sort_values(["fid", "iid", "rep", "eval"])

    all_evals = pd.Index(range(1, 1001), name="eval")

    dfs = []

    # Process each (fid, iid, rep) trajectory separately
    for (fid, iid, rep), g in df.groupby(["fid", "iid", "rep"], sort=False):
        g = g.set_index("eval").sort_index()

        # Reindex to eval 1..1000 and forward-fill previous best known value
        g = g.reindex(all_evals).ffill()

        # Restore identifiers
        g["fid"] = fid
        g["iid"] = iid
        g["rep"] = rep
        g["eval"] = g.index

        dfs.append(g.reset_index(drop=True))

    expanded = pd.concat(dfs, ignore_index=True)

    # expanded.to_csv("../data/expanded_convergence_data.csv", index=False)

    # Mean across iid and rep for each fid and eval
    result = (
        expanded.groupby(["fid", "eval"], as_index=False)["raw_y"]
        .mean()
        .rename(columns={"raw_y": "mean_raw_y"})
    )

    result.to_csv("../data/A1_mean_per_fid_eval_sbs.csv", index=False)

def create_convergence_file():
    df_selector = pd.read_csv("../data/B150_selector_choices.csv")


    # res is empty dataframe with columns (fid,iid,rep,eval,raw_y)
    res = pd.DataFrame(columns=["fid", "iid", "rep", "eval", "raw_y"])

    for _, row in df_selector.iterrows():
        fid = row["fid"]
        iid = row["iid"]
        rep = row["rep"]
        switch_budget = 150 # row["selector_switch_budget"]
        algo = row["selector_algorithm"]
        run_df = pd.read_csv(f"../data/A2_run_data_test/A2_{algo}_B{switch_budget}_5D.csv")
        run_df = run_df[run_df["fid"] == fid]
        run_df = run_df[run_df["iid"] == iid]
        run_df = run_df[run_df["rep"] == rep]
        run_df = run_df.sort_values("evaluations")
        res = pd.concat([res, run_df[["fid", "iid", "rep", "evaluations", "raw_y"]].rename(columns={"evaluations": "eval"})], ignore_index=True)
        print(f"Processed fid={fid}, iid={iid}, rep={rep}")

    res["eval"] = res["eval"].astype(int)
    res["raw_y"] = res["raw_y"].astype(float)
    res.to_csv("../data/convergence_data_B150.csv", index=False)

def plot_convergence_data(
    save_dir: str = "../figures/convergence_plots",
    combined: bool = False,
    combined_path: str = "../figures/convergence_plots/convergence_all.pdf",
):

    pio.kaleido.scope.mathjax = None
    os.makedirs(save_dir, exist_ok=True)

    # Load data
    base_path = "../data/convergence_plot_data"
    df = pd.read_csv(f"{base_path}/Selector_mean_per_fid.csv")
    df_a1 = pd.read_csv(f"{base_path}/A1_mean_per_fid.csv")
    df_res = pd.read_csv("../data/selector_performance_data/selector_results_with_lookahead_all_epms_10.csv")
    df_sbs = pd.read_csv(f"{base_path}/SBS_mean_per_fid.csv")
    df_b150 = pd.read_csv(f"{base_path}/B150_mean_per_fid.csv")

    # Ensure correct types and ordering
    df["eval"] = df["eval"].astype(int)
    df_a1["eval"] = df_a1["eval"].astype(int)
    df_sbs["eval"] = df_sbs["eval"].astype(int)
    df_b150["eval"] = df_b150["eval"].astype(int)

    df = df.sort_values(["fid", "eval"])
    df_a1 = df_a1.sort_values(["fid", "eval"])
    df_sbs = df_sbs.sort_values(["fid", "eval"])
    df_b150 = df_b150.sort_values(["fid", "eval"])

    fids = sorted(df["fid"].unique())

    def get_y_range(fid):
        """
        Upper bound:
            largest y-value reached by any curve at eval >= 150
        Lower bound:
            smallest positive y-value reached by any curve over the full run
        Returned in log10 space for Plotly's log axis.
        """
        sub = df[df["fid"] == fid]
        sub_a1 = df_a1[df_a1["fid"] == fid]
        sub_sbs = df_sbs[df_sbs["fid"] == fid]
        sub_b150 = df_b150[df_b150["fid"] == fid]

        upper_candidates = []
        for s in (sub, sub_a1, sub_sbs, sub_b150):
            # if fid=12, then include all evals, otherwise only evals >= 150
            if fid == 12:
                vals = s.loc[s["mean_raw_y"] > 0, "mean_raw_y"]
            else:
                vals = s.loc[(s["eval"] >= 150) & (s["mean_raw_y"] > 0), "mean_raw_y"]
                upper_candidates.extend(vals.tolist())

        lower_candidates = []
        for s in (sub, sub_a1, sub_sbs, sub_b150):
            vals = s.loc[s["mean_raw_y"] > 0, "mean_raw_y"]
            lower_candidates.extend(vals.tolist())

        if not upper_candidates or not lower_candidates:
            return None

        y_max = max(upper_candidates)
        y_min = min(lower_candidates)

        if y_min <= 0 or y_max <= 0:
            return None
        if y_min >= y_max:
            y_min = y_max / 10.0

        return [np.log10(y_min), np.log10(y_max)]

    def get_switch_stats(fid):
        vals = df_res.loc[df_res["fid"] == fid, "selector_switch_budget"].dropna()
        if len(vals) == 0:
            return None, None, None
        return vals.mean(), vals.quantile(0.25), vals.quantile(0.75)

    def add_switch_lines(fig, mean_switch, q1, q3, row=None, col=None):
        if mean_switch is None:
            return

        line_kwargs = {}
        if row is not None and col is not None:
            line_kwargs = {"row": row, "col": col}

        # Mean: solid
        fig.add_vline(
            x=mean_switch,
            line=dict(color="black", width=2),
            **line_kwargs,
        )

        # Quartiles: dotted
        for q in (q1, q3):
            if q is not None:
                fig.add_vline(
                    x=q,
                    line=dict(color="black", width=2, dash="dot"),
                    **line_kwargs,
                )

    def add_traces_and_vline(fig, fid, row=None, col=None, showlegend=True):
        sub = df[df["fid"] == fid]
        sub_a1 = df_a1[df_a1["fid"] == fid]
        sub_sbs = df_sbs[df_sbs["fid"] == fid]
        sub_b150 = df_b150[df_b150["fid"] == fid]

        mean_switch, q1, q3 = get_switch_stats(fid)

        trace_kwargs = {}
        if row is not None and col is not None:
            trace_kwargs = {"row": row, "col": col}

        fig.add_trace(
            go.Scatter(
                x=sub["eval"],
                y=sub["mean_raw_y"],
                mode="lines",
                line=dict(color="royalblue"),
                name="Dynamic selector",
                showlegend=showlegend,
            ),
            **trace_kwargs,
        )

        fig.add_trace(
            go.Scatter(
                x=sub_a1["eval"],
                y=sub_a1["mean_raw_y"],
                mode="lines",
                line=dict(color="firebrick"),
                name="A1",
                showlegend=showlegend,
            ),
            **trace_kwargs,
        )

        fig.add_trace(
            go.Scatter(
                x=sub_sbs["eval"],
                y=sub_sbs["mean_raw_y"],
                mode="lines",
                line=dict(color="darkgreen"),
                name="SBS",
                showlegend=showlegend,
            ),
            **trace_kwargs,
        )

        fig.add_trace(
            go.Scatter(
                x=sub_b150["eval"],
                y=sub_b150["mean_raw_y"],
                mode="lines",
                line=dict(color="darkorange"),
                name="Kostovska et al.",
                showlegend=showlegend,
            ),
            **trace_kwargs,
        )

        add_switch_lines(fig, mean_switch, q1, q3, row=row, col=col)

    if not combined:
        for fid in fids:
            sub = df[df["fid"] == fid]
            sub_a1 = df_a1[df_a1["fid"] == fid]
            sub_sbs = df_sbs[df_sbs["fid"] == fid]
            sub_b150 = df_b150[df_b150["fid"] == fid]

            mean_switch, q1, q3 = get_switch_stats(fid)
            y_range = get_y_range(fid)



            fig = go.Figure()

            fig.add_trace(
                go.Scatter(
                    x=sub["eval"],
                    y=sub["mean_raw_y"],
                    mode="lines",
                    line=dict(color="royalblue"),
                    name="Dynamic selector",
                    showlegend=True,
                )
            )

            fig.add_trace(
                go.Scatter(
                    x=sub_a1["eval"],
                    y=sub_a1["mean_raw_y"],
                    mode="lines",
                    line=dict(color="firebrick"),
                    name=r"$\mathcal{A}1$",
                    showlegend=True,
                )
            )

            fig.add_trace(
                go.Scatter(
                    x=sub_sbs["eval"],
                    y=sub_sbs["mean_raw_y"],
                    mode="lines",
                    line=dict(color="darkgreen"),
                    name="SBS",
                    showlegend=True,
                )
            )

            fig.add_trace(
                go.Scatter(
                    x=sub_b150["eval"],
                    y=sub_b150["mean_raw_y"],
                    mode="lines",
                    line=dict(color="darkorange"),
                    name="Kostovska et al.",
                    showlegend=True,
                )
            )

            add_switch_lines(fig, mean_switch, q1, q3)

            if fid == 12:
                x_range = [0, 1000]
            else:
                x_range = [150, 1000]

            fig.update_layout(
                width=700,
                height=450,
                font=dict(family="Latin Modern Roman", size=14, color="black"),
                margin=dict(l=80, r=30, t=40, b=60),
                plot_bgcolor="rgb(230,230,230)",
                paper_bgcolor="white",
                legend=dict(
                    x=0.98,
                    y=0.98,
                    xanchor="right",
                    yanchor="top",
                    bgcolor="rgba(255,255,255,0.8)",
                    bordercolor="black",
                    borderwidth=1,
                ),
            )

            fig.update_xaxes(
                title="Evaluations",
                range=x_range,
                showline=True,
                linewidth=2,
                linecolor="black",
                ticks="outside",
                tickcolor="black",
                showgrid=True,
                gridcolor="white",
                font=dict(size=24, family="Latin Modern Roman", color="black"),
            )

            fig.update_yaxes(
                title="Mean regret",
                showline=True,
                linewidth=2,
                linecolor="black",
                ticks="outside",
                tickcolor="black",
                tickformat=".0e",
                showgrid=True,
                gridcolor="white",
                type="log",
                range=y_range,
                font=dict(size=24, family="Latin Modern Roman", color="black"),
            )

            out_path = os.path.join(save_dir, f"convergence_f{fid}.pdf")
            fig.write_image(out_path)

    else:
        ncols = 4
        nrows = math.ceil(len(fids) / ncols)

        subplot_titles = [f"f{int(fid)}" for fid in fids]

        fig = make_subplots(
            rows=nrows,
            cols=ncols,
            subplot_titles=subplot_titles,
            horizontal_spacing=0.05,
            vertical_spacing=0.03,
        )

        fig.update_layout(
            annotations=[
                dict(font=dict(size=20))
                for _ in fig.layout.annotations
            ]
        )

        first = True
        for i, fid in enumerate(fids):
            row = i // ncols + 1
            col = i % ncols + 1

            add_traces_and_vline(fig, fid, row=row, col=col, showlegend=first)
            first = False

            y_range = get_y_range(fid)

            if fid == 12:
                x_range = [0, 1000]
            else:
                x_range = [150, 1000]

            fig.update_xaxes(
                range=x_range,
                showline=True,
                linewidth=1.5,
                linecolor="black",
                ticks="outside",
                tickcolor="black",
                showgrid=True,
                gridcolor="white",
                row=row,
                col=col,
            )
            fig.update_yaxes(
                type="log",
                range=y_range,
                showline=True,
                linewidth=1.5,
                linecolor="black",
                ticks="outside",
                tickcolor="black",
                tickformat=".0e",
                showgrid=True,
                gridcolor="white",
                row=row,
                col=col,
            )

            fig.update_xaxes(
                tickfont=dict(size=20),
                row=row,
                col=col,
            )

            fig.update_yaxes(
                tickfont=dict(size=20),
                row=row,
                col=col,
            )

            fig.update_yaxes(
                nticks=5,
                row=row,
                col=col,
            )

        fig.update_layout(
            width=1800,
            height=0.85 * 400 * nrows,
            font=dict(family="Latin Modern Roman", size=12, color="black"),
            margin=dict(l=100, r=0, t=25, b=140),
            plot_bgcolor="rgb(233,233,233)",
            paper_bgcolor="white",
            legend=dict(
                orientation="h",
                x=1.01,
                y=-0.055,
                xanchor="right",
                yanchor="bottom",
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor="black",
                borderwidth=1,
                font=dict(size=30, family="Latin Modern Roman", color="black"),
            ),
        )

        fig.add_annotation(
            text="Evaluations",
            x=0.5,
            y=0,
            xref="paper",
            yref="paper",
            yshift=-90,
            showarrow=False,
            font=dict(size=30, family="Latin Modern Roman", color="black"),
            standoff=30,
        )

        fig.add_annotation(
            text="Mean regret",
            x=-0.01,
            y=0.5,
            xref="paper",
            yref="paper",
            xshift=-90,
            textangle=-90,
            showarrow=False,
            font=dict(size=30, family="Latin Modern Roman", color="black"),
        )

        fig.write_image(combined_path)


if __name__ == "__main__":
    # # plot_convergence_data()
    # df = pd.read_csv("../data/A1_mean_per_fid_eval_sbs.csv")
    # # For each fid in df, check if the final mean_raw_y at eval=1000 is more than at eval=999, if yes then replace it with the value at eval=999
    # for fid in df["fid"].unique():
    #     sub = df[df["fid"] == fid]
    #     if sub[sub["eval"] == 1000]["mean_raw_y"].values[0] > sub[sub["eval"] == 999]["mean_raw_y"].values[0]:
    #         df.loc[(df["fid"] == fid) & (df["eval"] == 1000), "mean_raw_y"] = sub[sub["eval"] == 999]["mean_raw_y"].values[0]
    # df.to_csv("../data/A1_mean_per_fid_eval_sbs.csv", index=False)

    plot_convergence_data(combined=True)

    # aggregate_runs()
    # create_convergence_file()