"""
This file contains the implementation of the data collection process. This includes:
- Running the optimisation algorithms on the BBOB instances and logging their evaluations.
- Extracting the regrets achieved by the algorithms for creation of the training data.
- Saving the logged evaluations to a csv file for later use in ELA feature calculation.
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
            cma.parameters.elitist = False
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

        
def collect_A1_data(budget, dim = 5):
    trigger = ioh.logger.trigger.Always()

    logger = ioh.logger.Analyzer(
        triggers=[trigger],
        folder_name=f'./data/run_data_5D/A1_data_5D/A1_B{budget}_{dim}D',
        algorithm_name='ModCMA_A1',
        store_positions=True,
    )
    tracked_parameters = TrackedParameters()
    logger.watch(tracked_parameters, [x.name for x in fields(tracked_parameters)])
    
    for fid in range(1,25):
        for iid in range(1, 6):
            problem = ioh.get_problem(fid, iid, dim, ProblemClass.BBOB)

            
            # Attach the logger to the problem
            problem.attach_logger(logger)
            
            for rep in range(20):
                tracked_parameters.rep = rep
                tracked_parameters.iid = iid
                print(f"Running function {fid} instance {iid} repetition {rep} with A1, budget {budget}")
                np.random.seed(rep)
                cma = TrackedCMAES(
                    tracked_parameters, 
                    problem, 
                    dim, 
                    budget=budget,
                    active=True,
                    bound_correction='saturate',
                    sigma0 = 2.0,
                    x0 = np.zeros((dim,1)),
                    elitist = False
                ).run()
                problem.reset()
            problem.detach_logger()
            
            
def collect_A2(a1_budget, dim, A2, algname):
    trigger = ioh.logger.trigger.OnImprovement()
    # For BFGS, we log every evaluation s.t. later on we can only consider those that are within bounds
    if algname == "BFGS":
        trigger = ioh.logger.trigger.Always()

    logger = ioh.logger.Analyzer(
        triggers=[trigger],
        folder_name=f'./data/run_data_5D/A2_data_5D_scratch_750_test/A2_{algname}_B{a1_budget}_{dim}D',
        algorithm_name=algname,
        store_positions=True,
    )
    tracked_parameters = TrackedParameters()
    logger.watch(tracked_parameters, [x.name for x in fields(tracked_parameters)])

    for fid in range(1, 25):
        for iid in range(6, 8):

            problem = ioh.get_problem(fid, iid, dim, ProblemClass.BBOB)
    
            # Attach the logger to the problem
            problem.attach_logger(logger)

            for rep in range(20):
                tracked_parameters.rep = rep
                tracked_parameters.iid = iid
                print(f"Running function {fid} instance {iid} repetition {rep} with A2 {algname}, budget {a1_budget}, run_from_scratch={run_A2_from_scratch}")
                np.random.seed(rep)
             
                    
                if algname in ["Elitist", "Non-elitist"]:
                    alg = From_CMA_To_CMA(a1_budget, dim, algname, total_budget=1000)
                    alg(problem, algname)
                else:
                    alg = Switched_From_CMA(a1_budget, dim, A2, total_budget=1000)
                    alg(problem, A2)
                print("Evaluations:", problem.state.evaluations)
        
                problem.reset()
            
            problem.detach_logger()

### ======= ELA feature calculation ======= ###
'''
This part of the code will read the raw evaluation from the logger files created in collect_A1
and, for each run at each candidate switching budget, calculate the corresponding ELA features.
The calculated features will be saved in a csv file for later use in model training/tuning.
'''

class ELAFeatureCalculator:
    def __init__(self, data_folder, intermediate_evaluation_data_folder, output_folder, lower_bound = -5, upper_bound = 5, dim=5):
        """
        :param data_folder: The folder where the raw evaluation logs are stored (output of collect_A1)
        :param intermediate_evaluation_data_folder: The folder where the extracted evaluation with true objective values will be stored in a csv format (intermediate output of this class, used for ELA feature calculation)
        :param output_folder: The folder where the calculated ELA features will be stored (output of this class)
        :param lower_bound: The lower bound of the search space 
        :param upper_bound: The upper bound of the search space
        :param dim: The dimension of the problems
        """
        self.data_folder = data_folder
        self.intermediate_evaluation_data_folder = intermediate_evaluation_data_folder
        self.output_folder = output_folder
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.dim = dim

    def extract_evaluation_data(self):
        """
        Read the raw evaluation logs from the data folder, extract the evaluations with true objective values, 
        and save them in a csv file in the intermediate_evaluation_data_folder.
        """
        dim = self.dim
        for budget_dir in os.listdir(self.data_folder):

            budget_path = os.path.join(self.data_folder, budget_dir)
            if not os.path.isdir(budget_path):
                print(f"Skipping non-directory: {budget_path}")
                continue

            all_rows = []

            for func_dir in os.listdir(budget_path):
                func_path = os.path.join(budget_path, func_dir)
                if not os.path.isdir(func_path):
                    continue

                # Extract fid from directory name like 'data_f1_Sphere'
                try:
                    fid = int(func_dir.split('_')[1][1:])
                except (IndexError, ValueError):
                    print(f"Skipping malformed directory: {func_dir}")
                    continue

                dat_file = os.path.join(func_path, f"IOHprofiler_f{fid}_DIM{dim}.dat")
                if not os.path.isfile(dat_file):
                    print(f"Missing .dat file: {dat_file}")
                    continue

                try:
                    df = pd.read_csv(dat_file, delim_whitespace=True, comment="#", dtype=str)
                except Exception as e:
                    print(f"Error reading {dat_file}: {e}")
                    continue

                # Filter out repeated header rows
                df = df[df['iid'] != 'iid']

                # Convert selected columns to numeric
                numeric_cols = ['evaluations', 'raw_y', 'rep', 'iid', 'x0', 'x1', 'x2', 'x3', 'x4']

                df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')

                # Group by iid and compute absolute objective values from regrets
                for iid_val, group in df.groupby('iid'):
                    print(f"Processing fid={fid}, iid={iid_val}, budget dir={budget_dir}")
                    try:
                        iid_int = int(float(iid_val))
                        problem = ioh.get_problem(fid, iid_int, dim, ProblemClass.BBOB)
                        optimum = problem.optimum.y
                    except Exception as e:
                        print(f"Could not load problem fid={fid}, iid={iid_val}: {e}")
                        continue

                    group = group[numeric_cols].copy()
                    group['fid'] = fid
                    # Absolute objective value: Regret + Optimum

                    group['abs_obj'] = group['raw_y'] + optimum
                    all_rows.append(group)

            if all_rows:
                combined = pd.concat(all_rows, ignore_index=True)
                column_order = ['fid', 'iid', 'rep', 'evaluations', 'raw_y', 'x0', 'x1', 'x2', 'x3', 'x4', 'abs_obj']
                combined = combined[column_order]
                combined = combined.sort_values(by=['fid', 'iid', 'rep']).reset_index(drop=True)

                # Save CSV
                if not os.path.exists(self.intermediate_evaluation_data_folder):
                    os.makedirs(self.intermediate_evaluation_data_folder)

                output_path = os.path.join(self.intermediate_evaluation_data_folder, f"{budget_dir}.csv")
                combined.to_csv(output_path, index=False)
                print(f"Saved: {output_path}")

    def calculate_features(self):
        # Read the evaluations with true objective values from the intermediate_evaluation_data_folder
        # For each run at each candidate switching budget, calculate the ELA features
        # Save the calculated features to a CSV file in the output folder
        for csv_file in os.listdir(self.intermediate_evaluation_data_folder):
            if not csv_file.endswith('.csv'):
                continue

            # Assuming filename format is like 'A1_B{budget}_5D.csv'
            budget = int(csv_file.split('_')[1][1:])
            print(budget)  

            df = pd.read_csv(os.path.join(self.intermediate_evaluation_data_folder, csv_file))

            x_cols = [col for col in df.columns if col.startswith('x')]

            # ELA calulation for each run identified by (fid, iid, rep)
            for (fid, iid, rep), group in df.groupby(['fid', 'iid', 'rep']):
                np.random.seed(int(rep))

                print(f"Calculating features for fid={fid}, iid={iid}, rep={rep} from file {csv_file}")

                group = group.reset_index(drop=True)

                # Prepare the data for ELA calculation
                X = group[x_cols].to_numpy()
                y = group['abs_obj'].to_numpy(dtype=float)

                features = {}

                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=RuntimeWarning)
                    warnings.filterwarnings("ignore", category=UserWarning)
                    features.update(calculate_ela_distribution(X, y))
                    features.update(calculate_ela_meta(X, y))

                    # Need to handle different budgets, as not all quantiles are available for smaller budgets
                    if budget > 16:
                        if budget <= 88:
                            if budget <= 32:
                                features.update(calculate_ela_level(X, y, ela_level_quantiles=[0.50]))
                            else:
                                features.update(calculate_ela_level(X, y, ela_level_quantiles=[0.25, 0.50]))
                        else:
                            features.update(calculate_ela_level(X, y))

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
                        
                features["fid"] = fid
                features["iid"] = iid
                features["rep"] = rep

                if fid in [1, 2, 3, 4, 5]:
                    features["high_level_category"] = 1
                elif fid in [6, 7, 8, 9]:
                    features["high_level_category"] = 2
                elif fid in [10, 11, 12, 13, 14]:
                    features["high_level_category"] = 3
                elif fid in [15, 16, 17, 18, 19]:
                    features["high_level_category"] = 4
                elif fid in [20, 21, 22, 23, 24]:
                    features["high_level_category"] = 5
                else:
                    features["high_level_category"] = -1

                row_df = pd.DataFrame([features])
                
                #Rearrange columns to have fid, iid, rep, high_level_category at the front
                cols = row_df.columns.tolist()
                cols = ['fid', 'iid', 'rep', 'high_level_category'] + [col for col in cols if col not in ['fid', 'iid', 'rep', 'high_level_category']]
                row_df = row_df[cols]

                # Append row to file
                output_path = os.path.join(self.output_folder, f"ELA_features_B{budget}_5D.csv")

                if not os.path.exists(self.output_folder):
                    os.makedirs(self.output_folder)

                write_header = not os.path.exists(output_path)
                row_df.to_csv(output_path, mode='a', header=write_header, index=False)

if __name__ == "__main__":
    # # Example usage for runs with a1_budget = 500
    # # 1. Collect A1 data
    # collect_A1_data(budget=500, dim=5)
    # # 2. Collect A2 data for different algorithms
    # # for algname in ["Elitist", "Non-elitist", "BFGS", "PSO", "MLSL", "DE"]:
    # #     collect_A2(a1_budget=500, dim=5, A2=algname, algname=algname)

    # # 3. Extract evaluation data and calculate ELA features
    ela_calculator = ELAFeatureCalculator(
        data_folder='./data/run_data_5D/A1_data_5D',
        intermediate_evaluation_data_folder='./data/intermediate_evaluation_data_5D/A1',
        output_folder='./data/ELA_features_5D/A1'
    )
    # ela_calculator.extract_evaluation_data()
    ela_calculator.calculate_features()