"""
This file contains the implementation of the data collection process. This includes:
- Running the optimisation algorithms on the BBOB instances and logging their evaluations.
- Extracting the regrets achieved by the algorithms for creation of the training data.
- Calculating ELA features from the logged evaluations and saving them to a CSV file for use in model training/tuning.
"""

# === First part: Running the optimisation algorithms and logging their evaluations ===
from dataclasses import dataclass, fields
import os
import sys
import pandas as pd
import warnings

import ioh
from ioh import ProblemClass
from modcma import ModularCMAES, Parameters
import numpy as np

from sklearn.metrics import auc 

from pathlib import Path

# Import the algorithms to be used in A2

sys.path.append(os.path.join(os.path.dirname(__file__), 'optimisation_algorithms')) 
from bfgs import BFGS # type: ignore
from pso import PSO # type: ignore
from mlsl import MLSL # type: ignore
from de import DE # type: ignore

# Import ELA feature calculation functions
sys.path.append(os.path.join(os.path.dirname(__file__), 'pflacco'))
 
from classical_ela_features import ( # type: ignore
    calculate_ela_distribution,
    calculate_ela_meta,
    calculate_ela_level,
    calculate_dispersion,
    calculate_information_content,
    calculate_nbc
)

# Wrapper class for IOH problem to store evaluations in array for ELA calculation
class IOHProblemWrapper:
    def __init__(self, *args, **kwargs):
        self.problem = ioh.get_problem(*args, **kwargs)
        self.function_evals = {}
        self.best_so_far_evals = {}
        self.best_eval_so_far = np.inf

    def __call__(self, x):
        y = self.problem(x)

        # Update best eval so far
        if y < self.best_eval_so_far:
            self.best_eval_so_far = y

        self.function_evals[self.problem.state.evaluations] = (x, y)
        # Update the best evaluation so far
        self.best_so_far_evals[self.problem.state.evaluations] = (x, self.best_eval_so_far)
        return y

    def reset(self):
        self.problem.reset()
        self.function_evals = {}
        self.best_so_far_evals = {}
        self.best_eval_so_far = np.inf

    def __getattr__(self, name):
        return getattr(self.problem, name)

@dataclass
class TrackedParameters:
    # Static meta info
    rep: int = -1
    iid: int = -1

#     # Time series features
#     sigma: float = 0
#     t: int = 0
#     d_norm: float = 0
#     d_mean: float = 0 
#     ps_norm: float = 0
#     ps_mean: float = 0
#     pc_norm: float = 0
#     pc_mean: float = 0
    
#     # Anja parameters:
#     # ps_ratio: float = 0
#     ps_squared: float = 0
#     loglikelihood: float = 0
    
#     # check if this should only be one parameter
#     mhl_norm: float = 0
#     mhl_mean: float = 0
    
#     def update(self, parameters: Parameters):
#         self.sigma = parameters.sigma
#         self.t = parameters.t
#         for attr in ('D', 'ps', 'pc'):
#             setattr(self, f'{attr}_norm'.lower(), np.linalg.norm(getattr(parameters, attr)))
#             setattr(self, f'{attr}_mean'.lower(), np.mean(getattr(parameters, attr)))

#         self.ps_squared = np.sum(parameters.ps ** 2)
#         # self.ps_ratio = np.sqrt(self.ps_squared) / parameters.chiN

#         sigma2 = self.sigma ** 2
        
#         if hasattr(parameters.population, "x"):
#             delta = parameters.population.x.T - parameters.m.T
#             self.loglikelihood = -.5 * (parameters.lambda_ * (
#                 parameters.d * np.log(2 * np.pi * sigma2) + np.log(np.prod(parameters.D) ** 2)) 
#                 + np.diag(delta.dot(parameters.inv_root_C / sigma2).dot(delta.T)).sum()                
#             )
#         else:
#             delta = np.zeros((5, parameters.d))
#             self.loglikelihood = 0        
        
#         mhl = np.sqrt(
#             np.power(np.dot(parameters.B.T, delta.T) / parameters.D, 2).sum(axis=0)
#         ) / self.sigma
#         self.mhl_norm = np.linalg.norm(mhl)
#         self.mhl_mean = mhl.mean()

            
class TrackedCMAES(ModularCMAES):
    def __init__(self, tracked_parameters = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tracked_parameters = tracked_parameters
        # if self.tracked_parameters is not None:
        #     self.tracked_parameters.update(self.parameters)
        
    def step(self):
        # self.mutate()
        # self.select()
        # if self.tracked_parameters is not None:
        #     self.tracked_parameters.update(self.parameters)
        # self.recombine()
        # self.parameters.adapt()
        # self.tracked_parameters.t = self.parameters.t
        # return not any(self.break_conditions)
        res = super().step()
        # if self.tracked_parameters is not None:
        #     self.tracked_parameters.update(self.parameters)
        return res 
            
class From_CMA_To_CMA():
    def __init__(self, a1_budget, dim, A2, total_budget=1000):
        self.a1_budget = a1_budget
        self.dim = dim
        self.A2 = A2
        self.total_budget = total_budget
        
    def __call__(self, problem, A2, hparams = {}):
        if A2 == "Non-elitist":
            budget = self.total_budget
        else:
            budget = self.a1_budget
            
        cma = TrackedCMAES(
                    None, 
                    problem, 
                    self.dim, 
                    budget= budget,
                    active=True,
                    bound_correction='saturate',
                    sigma0 = 2.0,
                    x0 = np.zeros((self.dim,1)),
                    elitist = False
                ).run()
        
        if A2 == "Non-elitist":
            return
        
        if A2 == "Elitist":
            cma.parameters.elitist = True
            cma.parameters.budget = self.total_budget
        cma.run()
        
        
class Switched_From_CMA():
    def __init__(self, a1_budget, dim, A2, total_budget=1000):
        self.a1_budget = a1_budget
        self.dim = dim
        self.A2 = A2
        self.total_budget = total_budget
        
    def __call__(self, problem, A2, hparams = {}):
        
        cma = TrackedCMAES(
                    None, 
                    problem, 
                    self.dim, 
                    budget= self.a1_budget,
                    active=True,
                    bound_correction='saturate',
                    sigma0 = 2.0,
                    x0 = np.zeros((self.dim,1)),
                    elitist = False
                ).run()
        
        params = {}
        params['x_opt'] = cma.parameters.xopt
        params['pop'] = cma.parameters.population.x.T
        params['pop_f'] = cma.parameters.population.f
        params['stepsize'] = cma.parameters.sigma
        params['C'] = cma.parameters.C
        params['m'] = cma.parameters.m
        params['budget'] = self.total_budget
        
        
        algorithm = A2(problem, verbose = False, seed = np.random.get_state())
        # Set algorithm parameters based on parameters object
        algorithm.set_hyperparams(hparams)
        algorithm.set_params(params)
        
        def stopping_criteria():
            return problem.state.evaluations >= self.total_budget
        
        algorithm.set_stopping_criteria(stopping_criteria)
        algorithm.run()

def safe_df_to_csv(folder_path, file_name, df, append=True):
    """
    Safely saves a DataFrame to a CSV file, ensuring that the target directory exists.

    Parameters
    ----------
    folder_path : str
        The path to the folder where the CSV file should be saved.
    file_name : str
        The name of the CSV file (including .csv extension).
    df : pandas.DataFrame
        The DataFrame to be saved as a CSV file.
    """
    os.makedirs(folder_path, exist_ok=True)

    output_path = os.path.join(folder_path, file_name)

    if append:
        write_header = not os.path.exists(output_path)

        df.to_csv(
            output_path,
            mode="a",
            header=write_header,
            index=False
        )
    else:
        df.to_csv(output_path, index=False)

def get_ela_level_quantiles(budget):
    """
    Returns the quantiles to be used for ELA level calculation based on the budget. 
    For smaller budgets, not all quantiles are available, so we need to adjust accordingly.

    Parameters
    ----------
    budget : int
        The number of evaluations (budget) for which the ELA level features will be calculated.

    Returns
    -------
    list or str or None
        The quantiles to be used for ELA level calculation. None if no quantiles are available, "default" 
        pflacco's default settings are applicable.
    """
    if budget <= 16:
        return None
    if budget <= 88:
        return [0.50]
    return "default"

def get_hlc_from_fid(fid):
    """
    Returns the high level category based on the fid.

    Parameters
    ----------
    fid : int
        The function ID (fid) of the BBOB instance.

    Returns
    -------
    int
        The high level category corresponding to the given fid.
    """
    if fid in [1, 2, 3, 4, 5]:
        return 1
    elif fid in [6, 7, 8, 9]:  
        return 2
    elif fid in [10, 11, 12, 13, 14]:
        return 3
    elif fid in [15, 16, 17, 18, 19]:
        return 4
    elif fid in [20, 21, 22, 23, 24]:
        return 5

def calculate_ela_features(evaluations, fid, iid, rep, a1_budget, dim, algname):
    """
    This function calculates the ELA features from the given evaluations and returns them in a dictionary.

    Parameters
    ----------
    evaluations : dict
        A dictionary containing the evaluations, where the keys are the evaluation numbers and the values are tuples
        of the form (x, y), where x is the input and y is the objective value.
    fid : int
        The function ID (fid) of the BBOB instance from which the evaluations were obtained.
    iid : int
        The instance ID (iid) of the BBOB instance from which the evaluations were obtained.
    rep : int
        The repetition number of the run from which the evaluations were obtained.
    a1_budget : int
        The candidate switching budget to A2 (the budget at which the switch from A1 to A2 happens).
    dim : int
        The dimension of the BBOB instances to be used.
    algname : str
        The name of the algorithm which was switched to (or "Non-elitist", if no switch was made)

    Final csv will contain the following columns:
    - fid: function ID of the BBOB instance
    - iid: instance ID of the BBOB instance
    - rep: repetition number of the run
    - high_level_category: high level category of the BBOB function
    - a1_budget: the candidate switching budget to A2 (the budget at which the switch from A1 to A2 happens)
    - ela_budget: the number of evaluations (budget) from which the ELA features were calculated
    - a2_algorithm: the name of the algorithm which was switched to (or "Non-elitist" in case no switch occured)
    - Remaining columns: the calculated ELA features
    """
    # Prepare the data for ELA calculation
    x_cols = [f'x{i}' for i in range(dim)]
    X = np.array([eval[0] for eval in evaluations.values()])
    y = np.array([eval[1] for eval in evaluations.values()], dtype=float)

    budget = len(X)

    features = {}

    features["fid"] = fid
    features["iid"] = iid
    features["rep"] = rep
    
    # Add high level category based on fid
    features["high_level_category"] = get_hlc_from_fid(fid)

    features["a1_budget"] = a1_budget
    features["ela_budget"] = budget
    features["a2_algorithm"] = algname

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        warnings.filterwarnings("ignore", category=UserWarning)

        features.update(calculate_ela_distribution(X, y))
        features.update(calculate_ela_meta(X, y))
        
        # Need to handle different budgets, as not all quantiles are available for smaller budgets
        quantiles = get_ela_level_quantiles(budget)

        if quantiles is not None:
            if quantiles == "default":
                features.update(calculate_ela_level(X, y))
            else:
                features.update(calculate_ela_level(X, y, ela_level_quantiles=quantiles))

        features.update(calculate_dispersion(X, y))

        # Set range of epsilon values for information content to deal with early convergence
        features.update(calculate_information_content(X, y,
                                                    ic_epsilon=np.insert(10 ** np.linspace(start=-7, stop=15, num=1000), 0, 0)))
        
        if budget <= 16:
            features.update(calculate_nbc(X, y, fast_k=2))
        else:
            features.update(calculate_nbc(X, y))

        # features are inf if budget is too small  
        if budget <= 56:
            features.pop('ela_meta.quad_w_interact.adj_r2', None)
            if budget <= 16:
                features.pop('ela_meta.lin_w_interact.adj_r2', None)

        # No runtime features as we only want to consider features that characterise the landscape and do not depend on the hardware
        for key in list(features.keys()):
            if key.endswith('.costs_runtime'):
                features.pop(key)

    return features


def collect_data(a1_budget, dim, algs_to_run=["DE", "MLSL", "PSO", "BFGS", "Non-elitist", "Elitist"]):
    """
    This function runs the optimisation algorithms on the BBOB instances and logs
    their evaluations. It additionally computes ELA features every 50 evaluations
    and saves them to a csv file for later use in model training.
    It also extracts the regrets achieved by the algorithms for creation of the training data.

    Parameters
    ----------
    a1_budget : int
        The candidate switching budget to A2 (the budget at which the switch
        from A1 to A2 happens)

    dim : int
        The dimension of the BBOB instances to be used

    algs_to_run : list of str, optional
        The list of algorithms to run. Possible values are "DE", "MLSL", "PSO", "BFGS", "Non-elitist", "Elitist". 
        If not provided, all algorithms are run. For lower budgets, for which many ELA features are computed, it might be benificial
        to distribute the computation between algorithms across different jobs.
    """
    achieved_regrets = {}
    achieved_aucs = {}

    trigger = ioh.logger.trigger.Always()

    for A2, algname in zip([DE, MLSL, PSO, BFGS, None, None], ["DE", "MLSL", "PSO", "BFGS", "Non-elitist", "Elitist"]):
        if algname not in algs_to_run:
            continue

        # We only need to record Non-elitist iff A1_budget is 1000 to avoid redundancy
        if algname == "Non-elitist" and a1_budget != 1000:
            continue
        if algname != "Non-elitist" and a1_budget == 1000:
            continue

        logger = ioh.logger.Analyzer(
            triggers=[trigger],
            folder_name=f'./data/raw_evaluations/{algname}_B{a1_budget}_{dim}D',
            algorithm_name=algname,
            store_positions=True,
        )
        tracked_parameters = TrackedParameters()
        logger.watch(tracked_parameters, [x.name for x in fields(tracked_parameters)])
        for fid in range(1, 25):
            ela_features = []
            for iid in range(1, 8):

                problem = IOHProblemWrapper(fid, iid, dim, ProblemClass.BBOB)
        
                # Attach the logger to the problem
                problem.attach_logger(logger)

                for rep in range(20):
                    tracked_parameters.rep = rep
                    tracked_parameters.iid = iid
                    print(f"Running function {fid} instance {iid} repetition {rep} with A2 {algname}, budget {a1_budget}")
                    np.random.seed(rep)
                
                    if algname in ["Elitist", "Non-elitist"]:
                        alg = From_CMA_To_CMA(a1_budget, dim, algname, total_budget=1000)
                        alg(problem, algname)
                    else:
                        alg = Switched_From_CMA(a1_budget, dim, A2, total_budget=1000)
                        alg(problem, A2)
            
                    # Calculate ELA features every 50 evaluations and save to csv
                    for i in range(50, 1001, 50):
                        # If the algorithm is not Non-elitist, we only calculate features if budget > A1_budget to avoid redundancy
                        if algname != "Non-elitist" and i <= a1_budget:
                            continue

                        current_evaluations = {j: v for j, v in problem.function_evals.items() if j <= i}
                        ela_features.append(calculate_ela_features(current_evaluations, fid, iid, rep, a1_budget, dim, algname))

                    # The achieved regret of this specific run is the lowest objective value 
                    # that is within 1000 evals and within bounds
                    evals_to_consider_for_regret = {i: v for i, v in problem.function_evals.items() if i <= 1000 and np.all(np.abs(v[0]) <= 5)}
                    if evals_to_consider_for_regret:
                        best_eval = min(evals_to_consider_for_regret.values(), key=lambda x: x[1])
                        achieved_regrets[(fid, iid, rep, a1_budget, algname)] = best_eval[1] - problem.optimum.y

                    # The auc of the convergence curve
                    evals_to_consider_for_auc = {i: v for i, v in problem.best_so_far_evals.items() if i <= 1000}
                    if evals_to_consider_for_auc:
                        items = sorted(evals_to_consider_for_auc.items())

                        x = [k for k, _ in items]
                        y = [v[1] - problem.optimum.y for _, v in items]

                        # # Print curve
                        # for i in range(len(x)):
                        #     print(f"Eval: {x[i]}, Best so far: {y[i]}")

                        achieved_aucs[(fid, iid, rep, a1_budget, algname)] = auc(x, y)

                    problem.reset()
                
                problem.detach_logger()

            if ela_features:
                df = pd.DataFrame(ela_features)
                safe_df_to_csv(f'./data/ela_features/{algname}_B{a1_budget}_{dim}D', f"ELA_features.csv", df, append=True)

    # Save the achieved regrets to a csv file for later use in model training/tuning
    regrets_df = pd.DataFrame([{"fid": fid, "iid": iid, "rep": rep, "a1_budget": a1_budget, "algname": algname, "achieved_regret": regret} 
                                for (fid, iid, rep, a1_budget, algname), regret in achieved_regrets.items()])

    # Save the achieved AUCs to a csv file for later use in model training/tuning
    aucs_df = pd.DataFrame([{"fid": fid, "iid": iid, "rep": rep, "a1_budget": a1_budget, "algname": algname, "achieved_auc": auc} 
                            for (fid, iid, rep, a1_budget, algname), auc in achieved_aucs.items()])

    if len(algs_to_run) < 6:
        # store alg names in df name
        algs_str = "_".join(algs_to_run)
        safe_df_to_csv(f'./data/achieved_regrets/', f'achieved_regrets_{algs_str}_B{a1_budget}_{dim}D.csv', regrets_df)
        safe_df_to_csv(f'./data/achieved_aucs/', f'achieved_aucs_{algs_str}_B{a1_budget}_{dim}D.csv', aucs_df)
    else:
        safe_df_to_csv(f'./data/achieved_regrets/', f'achieved_regrets_B{a1_budget}_{dim}D.csv', regrets_df)
        safe_df_to_csv(f'./data/achieved_aucs/', f'achieved_aucs_B{a1_budget}_{dim}D.csv', aucs_df)

if __name__ == "__main__":
    # Read budget from command line argument, default to 500 if not provided
    if len(sys.argv) > 1:
        a1_budget = int(sys.argv[1])
    else:        
        a1_budget = 500

    if len(sys.argv) > 2:
        algorithms_to_run = sys.argv[2].split(",")
    else:
        algorithms_to_run = ["DE", "MLSL", "PSO", "BFGS", "Non-elitist", "Elitist"]

    # a1_budget = 500
    # algorithms_to_run = ["BFGS"]

    collect_data(a1_budget=a1_budget, dim=5, algs_to_run=algorithms_to_run)