## Data collection workflow

The function `collect_data()` creates the offline data used by the dynamic selector. It runs the optimization algorithms on BBOB instances, records their evaluation traces, computes performance metrics, and extracts ELA features from the observed search trajectory.

The default setup is:

- **Benchmark:** 24 BBOB functions
- **Instances:** 1–7
- **Repetitions:** 20 runs per `(fid, iid)` combination
- **Overall evaluation budget:** 1000 function evaluations
- **Candidate switching budgets:** every 50 evaluations
- **Initial algorithm (A1):** Non-elitist CMA-ES
- **Candidate A2 algorithms:**
  - Non-elitist CMA-ES
  - Elitist CMA-ES
  - PSO
  - DE
  - BFGS
  - MLSL

Each run starts with the A1 algorithm. For every candidate switching budget, the state reached by A1 is reused to warm-start each candidate A2 algorithm. This produces the offline reference data needed to ask questions such as:

- Which algorithm would have been best if we switched at this budget?
- Would it have been better to switch now or wait for a later budget?
- What ELA features were visible at the time the selector would have had to decide?

### Standalone baselines

`collect_data()` can also collect direct, non-switched baselines by passing
`standalone=True`.  In this mode DE, MLSL, PSO, BFGS, and both CMA-ES variants
are initialized from scratch, run for the full budget, and have their raw IOH
traces, regrets, and AUCs saved.  It deliberately skips ELA extraction and
internal-state logging.  There is no A1 phase; metric rows use
`a1_budget=0` to make that explicit.  For the requested 20 runs on BBOB
instances 6 and 7, for example:

```python
collect_data(
    a1_budget=0, dim=40, standalone=True, iids=range(6, 8),
    repetitions=20, use_ma=False,
)
```

Set `use_ma=True` for the ManyAffine BBOB realization.  Standalone outputs are
stored in `data/dim_<D>_standalone` (or `data/dim_<D>_ma_standalone`) and use
`B0` in their file names, e.g. `achieved_aucs_DE_B0_40D.csv`.  They therefore
cannot overwrite switched-run data.

### Generated data

The data collection writes three kinds of data.

1. **Raw logger data**

   Complete IOH logger traces are stored for every

   ```
   (fid, iid, rep, A2_algorithm, switching_budget)
   ```

   combination.

2. **Performance metrics**

   CSV files contain the achieved regret and achieved AUC for every

   ```
   (fid, iid, rep, A2_algorithm, switching_budget)
   ```

   combination.

3. **ELA feature data**

   ELA feature files contain one row for every

   ```
   (fid, iid, rep, A2_algorithm, A1_budget, ela_budget)
   ```

   combination.

   Here,

   - `A1_budget` is the candidate budget at which A1 is stopped and A2 starts.
   - `ela_budget` is the number of evaluations used to compute the ELA feature vector.

   ELA features are only computed when

   - `ela_budget ≥ A1_budget`, or
   - `A2_algorithm` is **Non-elitist CMA-ES** (i.e., no switch).

   For `ela_budget < A1_budget`, all samples still come from the initial A1 trajectory. The feature vector would therefore be identical for every possible A2 algorithm, so duplicate rows are not stored.



## Running the selector

The selector is implemented in `selector.py`. It uses three model types:

- **Selection models** choose the algorithm to use if a switch is made at a given budget.
- **Lookahead models** predict future performance targets `t_0`, `t_1`, ... from the current ELA features.
- **Switching models** decide whether the selector should switch now or keep waiting.

At evaluation time, the selector iterates over the candidate budgets. At each budget it computes the current ELA feature vector, augments it with lookahead predictions, asks the switching model whether to switch, and then uses the selection model to choose the A2 algorithm if a switch is made.

The script can be run directly from the `new_implementation` directory:

```bash
cd Dynamic_Selector/new_implementation
python selector.py --mode build-switch-data
```

The default paths are:

```text
base data path:    ./data
base model path:   ./models
base results path: ./results
```

The command-line interface uses `./data`, `./models`, and `./results`. All
selector-managed artifacts are stored in a metric-specific subfolder under the
dimension/dataset directory. For the default 40-dimensional regret experiment,
the effective directories are:

```text
data:    ./data/dim_40/regret
models:  ./models/dim_40/regret
results: ./results/dim_40/regret
```

Raw inputs produced by `data_collection.py` (`achieved_*` and `ela_features`) are read from the base data folder (`./data`). Selector-generated tables, models, and results are still written under the metric-scoped subfolders.

The available execution modes are:

```text
build-switch-data   Build switching-model training data from stored selection and lookahead tables.
train               Train selection, lookahead, and switching models.
evaluate            Load trained models and evaluate on the test instances.
train-evaluate      Train models and then evaluate them in the same run.
```

### Training

If the intermediate training tables already exist, train the final models with:

```bash
python selector.py --mode train --training-data-is-stored
```

This trains, for every switching budget:

- a selection model,
- one lookahead model per `t_*` target column,
- a switching model, and
- the ELA scaler.

Trained models are stored in `./models/dim_40/regret` by default.

If the intermediate training tables should be recreated from the collected raw data, omit `--training-data-is-stored`:

```bash
python selector.py --mode train
```

As training data, the selector uses all runs on instances **1–5**.

### Evaluation

Evaluate stored models with:

```bash
python selector.py --mode evaluate
```

To evaluate models trained on the standard collection against ManyAffine (MA)
data, keep the model source and evaluation dataset separate:

```bash
python selector.py --mode evaluate --dimension 40 --metric regret \
  --dataset ma --model-dataset default --lookahead-count 0
```

This loads from `models/dim_40/regret` and writes to
`results/dim_40_ma_models_default/regret/lookahead_0`, making the cross-dataset
configuration explicit. Submit all lookahead variants with
`./scripts/selector.sh 40 regret default ma`.

The evaluation loads models from `./models/regret` by default. Each budget directory must contain:

```text
selection_model.joblib
lookahead_models.joblib
switching_model.joblib
ela_scaler.joblib
```

To train and evaluate in one run:

```bash
python selector.py --mode train-evaluate --training-data-is-stored
```

The `evaluate()` method runs the selector on the test set, i.e., all runs on instances **6** and **7**. The performance is recorded in `results/regret/selector_results.csv` and contains, for each test run,

- the performance of the VBS,
- the performance of the selector,
- the algorithm and switching budget chosen by the selector, and
- the performance of each static selection model across all candidate switching budgets (including the no-switch baseline).

### Python API

The same workflow can be run from Python:

```python
selector = DynamicSelector(
    data_path="./data",
    results_path="./results",
    model_path="./models",
    load_models=False,
)

selector.train_models(training_data_is_stored=True)
selector.evaluate()
```

With the defaults above, the instance uses metric-scoped directories (`./data/regret`, `./results/regret`, `./models/regret`).

To evaluate already trained models:

```python
selector = DynamicSelector(
    data_path="./data",
    results_path="./results",
    model_path="./models",
    load_models=True,
)

selector.evaluate()
```
