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
import matplotlib.pyplot as plt

def plot_current_best_progress(csv_path, output_folder):
    """
    For each (fid, iid, rep), plot how current_best progresses over evaluations.
    - x-axis: evaluations
    - y-axis: current_best (log10 scale)
    - only plot every 8th evaluation (since current_best changes only then)
    - mark evaluations where 'optimal' is True with a star marker
    - save plots in subfolders per fid inside output_folder
    """

    os.makedirs(output_folder, exist_ok=True)

    df = pd.read_csv(csv_path)

    # Ensure correct dtypes
    df['fid'] = df['fid'].astype(int)
    df['iid'] = df['iid'].astype(int)
    df['rep'] = df['rep'].astype(int)
    df['evaluations'] = df['evaluations'].astype(int)

    # Make sure 'optimal' is boolean
    if df['optimal'].dtype != bool:
        df['optimal'] = df['optimal'].astype(str).str.lower().isin(['true', '1'])

    # Loop over each run
    for (fid, iid, rep), grp in df.groupby(['fid', 'iid', 'rep']):
        grp = grp.sort_values('evaluations')

        # --- only keep every 8th evaluation ---
        # we base this on the first evaluation in this group
        first_eval = grp['evaluations'].iloc[0]
        mask_every_8 = ((grp['evaluations'] - first_eval) % 8 == 0)
        grp_sub = grp[mask_every_8]

        if grp_sub.empty:
            continue  # nothing to plot

        evals = grp_sub['evaluations'].to_numpy()
        curr_best = grp_sub['current_best'].to_numpy()
        is_optimal = grp_sub['optimal'].to_numpy()

        # --- per-fid folder ---
        fid_folder = os.path.join(output_folder, f"fid_{fid}")
        os.makedirs(fid_folder, exist_ok=True)

        fig, ax = plt.subplots(figsize=(7, 5))

        # Line plot (no scatter) for the curve
        ax.plot(
            evals,
            curr_best,
            linestyle='-',
            marker=None,   # no markers for regular points
            linewidth=1.7
        )

        # Mark optimal points with stars (still using plot, not scatter)
        if is_optimal.any():
            evals_opt = evals[is_optimal]
            curr_best_opt = curr_best[is_optimal]
            ax.plot(
                evals_opt,
                curr_best_opt,
                linestyle='None',
                marker='*',
                markersize=10,
                markeredgecolor='black'
            )

        ax.set_xlabel('Evaluations')
        ax.set_ylabel('Current best (log10 scale)')

        # log10 y-scale
        ax.set_yscale('log', base=10)

        # ensure the scale goes down to 1e-12
        # ax.set_ylim(bottom=1e-14)

        ax.set_title(f'Current best progression\nfid={fid}, iid={iid}, rep={rep}')
        ax.grid(True, which='both', linestyle='--', alpha=0.4)

        filename = f"current_best_fid{fid}_iid{iid}_rep{rep}.png"
        save_path = os.path.join(fid_folder, filename)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close(fig)

        print(f"Processed current best for fid={fid}, iid={iid}, rep={rep}")

plot_current_best_progress("../data/A1_B1000_5D_with_optimal_last.csv",
                         "../results/current_best_last")