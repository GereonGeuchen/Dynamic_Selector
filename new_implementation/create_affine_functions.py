import matplotlib.pyplot as plt
import ioh
import numpy as np



def generate_weights(fid, dimension):
    """Use IOH's default weight generation, seeded by our mixture FID.

    IOH draws 24 uniform values with seed 2000 + fid, raises the two
    largest to at least 0.85, zeros values below 0.85, and normalizes.
    Reusing IOH preserves its exact BBOB random-number sequence. Weights
    do not depend on dimension.
    """
    return list(ioh.problem.ManyAffine(instance=fid, n_variables=dimension).weights)


def create_affine_function(fid, iid, weights, n_variables=2):
    """Keep a mixture fixed while varying BBOB instances and optimum."""
    location_instance = 1000 * fid + iid

    realization = ioh.problem.ManyAffine(
        instance=location_instance,
        n_variables=n_variables,
    )
    problem = ioh.problem.ManyAffine(
        xopt=list(realization.optimum.x),
        weights=weights,
        instances=[iid] * 24,
        n_variables=n_variables,
    )
    problem.set_id(fid)
    return problem


def main():
    selected_fids = [10,20]  # Choose fids here, e.g. [1, 5, 12, 20].
    iids = [1,2,3,4,5,67]
    dimension = 5
    weights_by_fid = {fid: generate_weights(fid, dimension) for fid in selected_fids}

    x1_range = np.linspace(-5, 5, 100)
    x2_range = np.linspace(-5, 5, 100)
    X1, X2 = np.meshgrid(x1_range, x2_range)

    nrows, ncols = len(weights_by_fid), len(iids)
    fig, axs = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4.5 * nrows), squeeze=False)
    for row, (fid, weights) in enumerate(weights_by_fid.items()):
        surfaces = []
        for iid in iids:
            f = create_affine_function(fid, iid, weights, n_variables=dimension)
            # Plot x1/x2 with remaining coordinates fixed at this optimum.
            points = np.tile(f.optimum.x, (X1.size, 1))
            points[:, 0] = X1.ravel()
            points[:, 1] = X2.ravel()
            surfaces.append(np.array([f(point) for point in points]).reshape(X1.shape))

        # Use the same color scale for both realizations of this mixture.
        vmin = min(Z.min() for Z in surfaces)
        vmax = max(Z.max() for Z in surfaces)
        if vmin == vmax:
            vmax = vmin + max(1.0, abs(vmin)) * 1e-6
        levels = np.linspace(vmin, vmax, 51)
        for col, (iid, Z) in enumerate(zip(iids, surfaces)):
            ax = axs[row, col]
            contour = ax.contourf(X1, X2, Z, levels=levels, cmap='viridis')
            fig.colorbar(contour, ax=ax, label='Function value')
            ax.set_title(f'FID {fid}, IID {iid} ({dimension}D)')
            ax.set_xlabel('x1')
            ax.set_ylabel('x2')

    if dimension > 2:
        fig.suptitle('x1/x2 slices; remaining coordinates fixed at each optimum')
    fig.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
