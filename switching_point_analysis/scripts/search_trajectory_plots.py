import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

from sklearn.decomposition import PCA




def plot_search_trajectories_pca(csv_path, output_folder):
    
    pio.kaleido.scope.mathjax = None
    os.makedirs(output_folder, exist_ok=True)

    df = pd.read_csv(csv_path)

    # Types
    df["fid"] = df["fid"].astype(int)
    df["iid"] = df["iid"].astype(int)
    df["rep"] = df["rep"].astype(int)
    df["evaluations"] = df["evaluations"].astype(int)

    if df["is_optimal"].dtype != bool:
        df["is_optimal"] = df["is_optimal"].astype(str).str.lower().isin(["true", "1"])

    x_cols = [c for c in df.columns if c.startswith("x")]
    if len(x_cols) < 2:
        raise ValueError(f"Need at least 2 x-columns, found {x_cols}")

    # PCA (correct version with centering)
    # def pca_2d(X):
    #     X = np.asarray(X, dtype=float)
    #     Xc = X - X.mean(axis=0, keepdims=True)
    #     cov = np.cov(Xc, rowvar=False)
    #     eigvals, eigvecs = np.linalg.eigh(cov)
    #     idx = np.argsort(eigvals)[::-1]
    #     return Xc @ eigvecs[:, idx[:2]]

    for (fid, iid, rep), grp in df.groupby(["fid", "iid", "rep"], sort=False):
        if iid != 1:
            continue

        grp = grp.sort_values("evaluations").copy()

        if len(grp) < 2:
            continue

        X = grp[x_cols].to_numpy(dtype=float)

        # sklearn PCA
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)

        grp["pc1"] = X_pca[:, 0]
        grp["pc2"] = X_pca[:, 1]

        fid_folder = os.path.join(output_folder, f"fid_{fid}")
        os.makedirs(fid_folder, exist_ok=True)

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=grp["pc1"],
                y=grp["pc2"],
                mode="markers", 
                marker=dict(
                    size=6,
                    color=grp["evaluations"],
                    colorscale="Viridis",
                    showscale=True,
                    colorbar=dict(
                        title="Evaluations",
                        tickfont=dict(size=16, family="Latin Modern Roman"),
                    ),
                ),
                name="Trajectory",
            )
        )

        # Optimal points
        grp_opt = grp[grp["is_optimal"]]
        if not grp_opt.empty:
            fig.add_trace(
                go.Scatter(
                    x=grp_opt["pc1"],
                    y=grp_opt["pc2"],
                    mode="markers",
                    marker=dict(
                        symbol="star",
                        size=13,
                        color=grp_opt["evaluations"],
                        colorscale="Viridis",
                        showscale=False,
                        line=dict(color="black", width=1),
                    ),
                    name="Optimal iteration",
                )
            )

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
            # title=dict(
            #     text=f"fid={fid}, iid={iid}, rep={rep}",
            #     x=0.5,
            #     xanchor="center",
            # ),
        )

        # --- No fixed range anymore ---
        fig.update_xaxes(
            title="PC1",
            showline=True,
            linewidth=2,
            linecolor="black",
            ticks="outside",
            showgrid=True,
            gridcolor="white",
            tickfont=dict(size=24, family="Latin Modern Roman"),
            title_font=dict(size=24, family="Latin Modern Roman"),
            zeroline=False,
        )

        fig.update_yaxes(
            title="PC2",
            showline=True,
            linewidth=2,
            linecolor="black",
            ticks="outside",
            showgrid=True,
            gridcolor="white",
            tickfont=dict(size=24, family="Latin Modern Roman"),
            title_font=dict(size=24, family="Latin Modern Roman"),
            zeroline=False,
            scaleanchor="x",
            scaleratio=1,
        )

        filename = f"search_pca_fid{fid}_iid{iid}_rep{rep}.pdf"
        fig.write_image(os.path.join(fid_folder, filename))

        print(f"Processed PCA trajectory for fid={fid}, iid={iid}, rep={rep}")

def plot_current_best_progress(csv_path, output_folder):

    pio.kaleido.scope.mathjax = None
    os.makedirs(output_folder, exist_ok=True)

    df = pd.read_csv(csv_path)

    # Ensure correct dtypes
    df["fid"] = df["fid"].astype(int)
    df["iid"] = df["iid"].astype(int)
    df["rep"] = df["rep"].astype(int)
    df["evaluations"] = df["evaluations"].astype(int)
    df["raw_y"] = df["raw_y"].astype(float)
    df["current_best"] = df["current_best"].astype(float)

    if df["is_optimal"].dtype != bool:
        df["is_optimal"] = df["is_optimal"].astype(str).str.lower().isin(["true", "1"])

    df["current_best_plot"] = df["current_best"].clip(lower=1e-12)

    for (fid, iid, rep), grp in df.groupby(["fid", "iid", "rep"], sort=False):
        if iid != 1: continue
        grp = grp.sort_values("evaluations").copy()

        if grp.empty:
            continue

        fid_folder = os.path.join(output_folder, f"fid_{fid}")
        os.makedirs(fid_folder, exist_ok=True)

        y_vals = grp["current_best_plot"]
        y_min = y_vals.min()
        y_max = y_vals.max()

        if y_min <= 0:
            y_min = 1e-12
        if y_max <= 0:
            y_max = 1e-11
        if y_min >= y_max:
            y_min = y_max / 10.0

        y_range = [np.log10(y_min), np.log10(y_max)]

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=grp["evaluations"],
                y=grp["current_best_plot"],
                mode="lines",
                line=dict(color="royalblue", width=2),
                name="Current best",
                showlegend=True,
            )
        )

        grp_opt = grp[grp["is_optimal"]]
        if not grp_opt.empty:
            fig.add_trace(
                go.Scatter(
                    x=grp_opt["evaluations"],
                    y=grp_opt["current_best_plot"],
                    mode="markers",
                    marker=dict(
                        symbol="star",
                        size=11,
                        color="royalblue",
                        line=dict(color="black", width=1),
                    ),
                    name="Optimal iteration",
                    showlegend=True,
                )
            )

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
            title=dict(
                text=f"fid={fid}, iid={iid}, rep={rep}",
                x=0.5,
                xanchor="center",
            ),
        )

        fig.update_xaxes(
            title="Evaluations",
            range=[grp["evaluations"].min(), grp["evaluations"].max()],
            showline=True,
            linewidth=2,
            linecolor="black",
            ticks="outside",
            tickcolor="black",
            showgrid=True,
            gridcolor="white",
            tickfont=dict(size=24, family="Latin Modern Roman", color="black"),
            title_font=dict(size=24, family="Latin Modern Roman", color="black"),
        )

        fig.update_yaxes(
            title="Current best regret",
            type="log",
            range=y_range,
            showline=True,
            linewidth=2,
            linecolor="black",
            ticks="outside",
            tickcolor="black",
            tickformat=".0e",
            showgrid=True,
            gridcolor="white",
            tickfont=dict(size=24, family="Latin Modern Roman", color="black"),
            title_font=dict(size=24, family="Latin Modern Roman", color="black"),
        )

        filename = f"current_best_fid{fid}_iid{iid}_rep{rep}.pdf"
        save_path = os.path.join(fid_folder, filename)
        fig.write_image(save_path)

        print(f"Processed current best for fid={fid}, iid={iid}, rep={rep}")

# plot_current_best_progress("../data/A1_B1000_5D_with_current_best_with_is_optimal_lowest_and_x.csv",
#                          "../results/current_best_with_optimal_points_lowest")

for word in ["all", "lowest", "highest"]:
    plot_search_trajectories_pca(f"../data/A1_B1000_5D_with_current_best_with_is_optimal_{word}_and_x.csv",
                                "../results/search_trajectories_pca_with_optimal_points_" + word)
    
    plot_current_best_progress(f"../data/A1_B1000_5D_with_current_best_with_is_optimal_{word}_and_x.csv",
                            f"../results/current_best_with_optimal_points_{word}")