## Data collection workflow

The function `collect_data()` generates all data required for training and evaluating the selector. It executes the necessary optimization runs, computes the corresponding performance metrics, and extracts ELA features.

By default, the data collection uses the following configuration:

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

For each run, the initial CMA-ES trajectory is replayed, and every candidate A2 algorithm is warm-started from each candidate switching budget.

### Generated data

The data collection produces three types of outputs:

1. **Raw logger data**

   Complete optimization traces for every

   ```
   (fid, iid, rep, A2_algorithm, switching_budget)
   ```

   combination.

2. **Performance metrics**

   CSV files containing the achieved regret and achieved AUC for every

   ```
   (fid, iid, rep, A2_algorithm, switching_budget)
   ```

   combination.

3. **ELA feature data**

   ELA feature files containing one observation for every

   ```
   (fid, iid, rep, A2_algorithm, A1_budget, ela_budget)
   ```

   combination.

   Here,

   - `A1_budget` denotes the budget at which the switch to the A2 algorithm is performed.
   - `ela_budget` denotes the number of evaluations used to compute the ELA features.

   ELA features are only computed when

   - `ela_budget ≥ A1_budget`, or
   - `A2_algorithm` is **Non-elitist CMA-ES** (i.e., no switch).

   The reason is that for `ela_budget < A1_budget`, all evaluated samples originate from the initial non-elitist CMA-ES trajectory. Consequently, the extracted ELA features would be identical for every A2 algorithm, making it unnecessary to store duplicate feature vectors.



## Running the selector

The selector is implemented in `selector.py`. A selector is created by calling

```python
selector = Selector(
)
```

If all parameters are left at their default values, the selector uses the folder structure created by `data_collection.py`. When instantiating the selector, one can set `load_models=True` to load already trained models.

### Training

If no models have been trained yet, the training pipeline is executed by calling

```python
selector.train()
```

If the parameter `training_data_is_stored=False`, the required data for training the models is first created. This includes

- the training data for the selection models, and
- the training data for the switching models.

As training data, the selector uses all runs on instances **1–5**.

### Evaluation

The trained selector is evaluated by calling

```python
results = selector.evaluate()
```

The `evaluate()` method runs the selector on the test set, i.e., all runs on instances **6** and **7**. The performance is recorded in `results/selector_results.py` and contains, for each test run,

- the performance of the VBS,
- the performance of the selector,
- the algorithm and switching budget chosen by the selector, and
- the performance of each static selection model across all candidate switching budgets (including the no-switch baseline).