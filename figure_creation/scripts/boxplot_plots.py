import pandas as pd

import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

# Boxplot interpretation (Plotly default / Tukey definition):
# - The horizontal line inside each box is the median (50th percentile).
# - The box spans from Q1 (25th percentile) to Q3 (75th percentile),
#   i.e., it contains the middle 50% of the data (IQR = Q3 - Q1).
# - The whiskers extend to the most extreme data points that are still within
#   [Q1 - 1.5 * IQR, Q3 + 1.5 * IQR].
# - Data points outside this range are shown individually as outliers.

def plot_precision_boxplots_per_fid(
    input_path: str,
    output_dir: str = "../figures/precision_boxplots",
):
    pio.kaleido.scope.mathjax = None

    df = pd.read_csv(input_path)

    os.makedirs(output_dir, exist_ok=True)

    # Ensure numeric columns
    cols = ["selector_precision", "static_B150", "SBS_precision", "vbs_precisions"]
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    fids = sorted(df["fid"].dropna().astype(int).unique().tolist())

    label_map = {
        "selector_precision": "Dynamic Selector",
        "static_B150": "Kostovska et al.",
        "SBS_precision": "SBS",
        "vbs_precisions": "VBS",
    }

    def make_long_df(sub: pd.DataFrame, use_log: bool) -> pd.DataFrame:
        data = []
        for col in cols:
            vals = sub[col].dropna().to_numpy(dtype=float)

            if use_log:
                vals = np.where(vals <= 0, 1e-12, vals)
                vals = np.log10(vals)

            for v in vals:
                data.append(
                    {
                        "method_col": col,
                        "method_label": label_map[col],
                        "value": v,
                    }
                )

        return pd.DataFrame(data)

    def build_fig(plot_df: pd.DataFrame, fid: int, use_log: bool) -> go.Figure:
        fig = go.Figure()

        order = ["selector_precision", "static_B150", "SBS_precision", "vbs_precisions"]

        for col in order:
            vals = plot_df.loc[plot_df["method_col"] == col, "value"]

            fig.add_trace(
                go.Box(
                    y=vals,
                    name=label_map[col],
                    boxpoints="outliers",
                    marker=dict(
                        color="royalblue",
                        line=dict(color="black", width=1),
                    ),
                    line=dict(color="black", width=1.5),
                    fillcolor="royalblue",
                    opacity=0.9,
                    width=0.55,
                    showlegend=False,
                )
            )

        y_title = "log10 regret" if use_log else "regret"

        fig.update_xaxes(
            tickmode="array",
            tickvals=[
                "Dynamic Selector",
                "Kostovska et al.",
                "SBS",
                "VBS",
            ],
            ticktext=[
                "Dynamic Selector",
                "Kostovska et al.",
                "SBS",
                "VBS",
            ],
            showline=True,
            linewidth=2,
            linecolor="black",
            tickfont=dict(color="black", size=16, family="Latin Modern Roman"),
            title_font=dict(color="black", size=18, family="Latin Modern Roman"),
            title=None,
        )

        fig.update_yaxes(
            title=y_title,
            tickfont=dict(color="black", size=16, family="Latin Modern Roman"),
            title_font=dict(color="black", size=18, family="Latin Modern Roman"),
            title_standoff=1.5,
            showline=True,
            linewidth=2,
            linecolor="black",
            zeroline=False,
            tickformat=".0e",   # <-- this is the key
        )

        # Subtle vertical guides behind boxes
        for x in ["Dynamic Selector", "Kostovska et al.", "SBS", "VBS"]:
            fig.add_vline(
                x=x,
                line_color="rgba(120,120,120,0.4)",
                line_width=1,
                layer="below",
            )

        fig.update_layout(
            width=700,
            height=350,
            plot_bgcolor="rgb(230,230,230)",
            paper_bgcolor="white",
            font=dict(
                family="Latin Modern Roman",
                color="black",
                size=16,
            ),
            # title=dict(
            #     text=f"f{fid}",
            #     x=0.5,
            #     xanchor="center",
            #     font=dict(
            #         family="Latin Modern Roman",
            #         color="black",
            #         size=20,
            #     ),
            # ),
            margin=dict(l=20, r=20, t=40, b=20),
        )

        return fig

    for fid in fids:
        sub = df[df["fid"] == fid].copy()

        if sub.empty:
            continue

        # Raw plot
        raw_df = make_long_df(sub, use_log=False)
        fig_raw = build_fig(raw_df, fid=fid, use_log=False)
        fig_raw.write_image(os.path.join(output_dir, f"precision_boxplot_f{fid}.pdf"))

        # Log plot
        log_df = make_long_df(sub, use_log=True)
        fig_log = build_fig(log_df, fid=fid, use_log=True)
        fig_log.write_image(os.path.join(output_dir, f"precision_boxplot_log_f{fid}.pdf"))

if __name__ == "__main__":
    for i in range(-1, 20):
        print(f"Processing {i} lookahead EPMs...")
        plot_precision_boxplots_per_fid(
            input_path=f"../data/selector_performance_data/highest_tiebreak/selector_results_with_lookahead_all_epms_{i}_sbs.csv",
            output_dir=f"../figures/precision_boxplots/highest_tiebreak/{i}_lookahead_epms",
        )