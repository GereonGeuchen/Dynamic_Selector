import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_search_trajectories(csv_path, output_folder):
    """
    Create a PCA-based 2D trajectory plot for each (fid, iid, rep).

    Parameters
    ----------
    csv_path : str
        Path to the CSV file with columns:
        ['fid', 'iid', 'rep', 'evaluations', x0, x1, ..., 'optimal', ...]
    output_folder : str
        Folder in which to save the plots.
    """

    # Make sure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Load data
    df = pd.read_csv(csv_path)

    # Ensure basic types
    df['fid'] = df['fid'].astype(int)
    df['iid'] = df['iid'].astype(int)
    df['rep'] = df['rep'].astype(int)
    df['evaluations'] = df['evaluations'].astype(int)

    # Make sure 'optimal' is boolean, even if saved as 0/1 or strings
    if df['optimal'].dtype != bool:
        df['optimal'] = df['optimal'].astype(str).str.lower().isin(['true', '1'])

    # Automatically detect x-columns (x0, x1, x2, ...)
    x_cols = [c for c in df.columns if c.startswith('x')]
    if len(x_cols) < 2:
        raise ValueError(f"Need at least 2 x-columns for PCA, found: {x_cols}")

    # Helper: simple PCA to 2D using numpy
    def pca_2d(X):
        """
        X: (n_samples, n_features)
        returns: X projected to 2D (n_samples, 2)
        """
        # Center
        Xc = X - X.mean(axis=0, keepdims=True)
        # Covariance
        cov = np.cov(Xc, rowvar=False)
        # Eigen-decomposition
        eigvals, eigvecs = np.linalg.eigh(cov)
        # Sort eigenvectors by eigenvalue descending
        idx = np.argsort(eigvals)[::-1]
        W = eigvecs[:, idx[:2]]  # top-2 components
        # Project
        return Xc @ W

    # Group by (fid, iid, rep)
    for (fid, iid, rep), grp in df.groupby(['fid', 'iid', 'rep']):
        # If too few points, skip
        if len(grp) < 2:
            continue
        
        # --- Create fid-specific subfolder ---
        fid_folder = os.path.join(output_folder, f"fid_{fid}")
        os.makedirs(fid_folder, exist_ok=True)

        # Sort by evaluation index so trajectory is in time order
        grp = grp.sort_values('evaluations')

        X = grp[x_cols].to_numpy()
        X_pca = pca_2d(X)

        # Prepare plotting data
        evals = grp['evaluations'].to_numpy()
        is_optimal = grp['optimal'].to_numpy()

        # Create figure
        fig, ax = plt.subplots(figsize=(6, 5))

        # All points: colored by evaluation index (color gradient)
        sc = ax.scatter(
            X_pca[:, 0],
            X_pca[:, 1],
            c=evals,
            cmap='viridis',
            s=25,
            alpha=0.8
        )

        # Highlight optimal evaluations with star marker
        if is_optimal.any():
            X_opt = X_pca[is_optimal]
            evals_opt = evals[is_optimal]
            ax.scatter(
                X_opt[:, 0],
                X_opt[:, 1],
                c=evals_opt,
                cmap='viridis',
                marker='*',
                s=120,
                edgecolors='black',
                linewidths=0.7
            )

        # Colorbar for evaluation index
        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label('Evaluation index')

        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_title(f'fid={fid}, iid={iid}, rep={rep}')

        # Save figure
        filename = f"fid{fid}_iid{iid}_rep{rep}.png"
        out_path = os.path.join(output_folder, filename)
        plt.tight_layout()
        
        # Save into fid-specific folder
        filename = f"fid{fid}_iid{iid}_rep{rep}.png"
        save_path = os.path.join(fid_folder, filename)
        fig.savefig(save_path, dpi=150)
        plt.close(fig)

        print(f"Processed fid={fid}, iid={iid}, rep={rep}")

        # (Optional) print or log where things went
        # print(f"Saved {out_path}")

import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio


def plot_current_best_progress(csv_path, output_folder):
    """
    For each (fid, iid, rep), create one Plotly figure showing the progression of
    current_best over evaluations.

    - x-axis: evaluations
    - y-axis: current_best on log scale
    - plot every evaluation
    - highlight rows with is_optimal == True using star markers
    - save plots in subfolders per fid inside output_folder
    """

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

    # Use is_optimal, not optimal
    if "is_optimal" not in df.columns:
        raise ValueError("Expected column 'is_optimal' in input file.")

    if df["is_optimal"].dtype != bool:
        df["is_optimal"] = df["is_optimal"].astype(str).str.lower().isin(["true", "1"])

    # Avoid log-scale problems
    df["current_best_plot"] = df["current_best"].clip(lower=1e-12)

    for (fid, iid, rep), grp in df.groupby(["fid", "iid", "rep"], sort=False):
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

        # Main trajectory line
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

        # Optimal points as stars
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
                    name="Optimal interval",
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

plot_current_best_progress("../data/A1_B1000_5D_with_current_best_with_is_optimal_all_and_x.csv",
                         "../results/current_best_with_optimal_points_all")