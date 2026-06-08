## Data collection workflow

For each dimension and candidate A1 budget, we run `collect_data(A1_budget, dim)` to generate all data required for training and evaluation. Each run uses an overall budget of 1000 function evaluations. The first `A1_budget` evaluations are allocated to the initial A1 algorithm; afterwards, one of the candidate A2 algorithms is warm-started and executed for the remaining budget.

The A2 portfolio consists of BFGS, MLSL, PSO, DE, non-elitist CMA-ES, and elitist CMA-ES.

For every A1 budget and dimension, the data collection produces:

- logger files containing the raw evaluation traces of all recorded runs;
- `achieved_regrets_B{A1_budget}_D{dim}.csv`, storing the final achieved regrets;
- `achieved_aucs_B{A1_budget}_D{dim}.csv`, storing the achieved AUC values;
- one ELA feature file per A2 algorithm:
  `ela_features/{algname}_B{A1_budget}_{dim}D.csv`.

Each ELA file contains one observation for each `(fid, iid, rep, ela_budget)` combination. For a fixed run, ELA features are repeatedly extracted from increasingly long trajectory prefixes, yielding one row per sampled evaluation budget.

More specifically, each row corresponds to a specific run `(fid, iid, rep)` and a specific ELA sampling budget (`ela_budget`). ELA features are computed from the first `ela_budget` evaluations of the trajectory, where `ela_budget ∈ {100, 150, 200, ..., 1000}`. Consequently, multiple rows are recorded for each run, one for every considered ELA sampling budget.