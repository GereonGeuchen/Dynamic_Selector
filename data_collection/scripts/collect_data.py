import sys
import os 
sys.path.append(os.path.join(os.path.dirname(__file__), 'algorithms'))
import time

import shutil
import argparse
from dataclasses import dataclass, fields
import pandas as pd

import ioh
from ioh import ProblemClass, problem
from modcma import ModularCMAES, Parameters
import numpy as np
# from lasso import LassoBenchWrapper

from bfgs import BFGS # type: ignore
from pso import PSO # type: ignore
from mlsl import MLSL # type: ignore
from de import DE # type: ignore
import warnings
from itertools import product
from functools import partial
from multiprocessing import cpu_count
from multiprocessing import Pool

def runParallelFunction(runFunction, arguments):
    """
        Return the output of runFunction for each set of arguments,
        making use of as much parallelization as possible on this systemc

        :param runFunction: The function that can be executed in parallel
        :param arguments:   List of tuples, where each tuple are the arguments
                            to pass to the function
        :return:
    """
    

    arguments = list(arguments)
    p = Pool(min(cpu_count(), len(arguments)))
#     local_func = partial(func_star, func=runFunction)
    results = p.map(runFunction, arguments)
    p.close()
    return results

# Class to time the algorithm execution
class TimedProblemWrapper:
    def __init__(self, problem, start_eval=0, stop_eval=1000):
        self.problem = problem
        self.start_eval = start_eval
        self.stop_eval = stop_eval
        self._start_time = None
        self._start_cpu = None
        self._timing_done = False

    def __call__(self, x):
        evals = self.problem.state.evaluations

        if evals <= self.start_eval:
            self._start_time = time.time()
            self._start_cpu = time.process_time()

        y = self.problem(x)

        evals = self.problem.state.evaluations
        if evals >= self.stop_eval and not self._timing_done:
            self._timing_done = True
            wall = time.time() - self._start_time
            cpu = time.process_time() - self._start_cpu
            print(f"[Timing] Evaluations {self.start_eval}–{self.stop_eval}, {self.state.evaluations}:")
            print(f"  Wall-clock time: {wall:.6f} seconds")
            print(f"  CPU time:        {cpu:.6f} seconds")

        return y

    def attach_logger(self, logger):
        self.problem.attach_logger(logger)

    def detach_logger(self):
        self.problem.detach_logger()

    def reset(self):
        self.problem.reset()
        self._start_time = None
        self._start_cpu = None
        self._timing_done = False

    def __getattr__(self, attr):
        return getattr(self.problem, attr)


# class LoggingProblemWrapper:
#     def __init__(self, problem):
#         self.problem = problem
#         self.logged_evals = []

#     def __call__(self, x):
#         y = self.problem(x)
#         self.logged_evals.append((np.copy(x), y))
#         return y

#     def reset_log(self):
#         self.logged_evals = []

#     def get_log(self):
#         return self.logged_evals

#     def get_array(self):
#         if not self.logged_evals:
#             return np.array([]), np.array([])
#         xs = np.array([x for x, y in self.logged_evals])
#         ys = np.array([y for x, y in self.logged_evals])
#         return xs, ys

#     def attach_logger(self, logger):
#         """Pass IOH logger through to the underlying problem"""
#         if hasattr(self.problem, "attach_logger"):
#             self.problem.attach_logger(logger)
#         else:
#             raise AttributeError("Underlying problem does not support logger attachment")

#     def detach_logger(self):
#         """Detach IOH logger from the underlying problem"""
#         if hasattr(self.problem, "detach_logger"):
#             self.problem.detach_logger()
    
#     def reset_log(self):
#         """Reset the logged evaluations"""
#         self.logged_evals = []

#     def __getattr__(self, attr):
#         return getattr(self.problem, attr)





@dataclass
class TrackedParameters:
    # Static meta info
    rep: int = -1
    iid: int = -1

    # Time series features
    sigma: float = 0
    t: int = 0
    d_norm: float = 0
    d_mean: float = 0 
    ps_norm: float = 0
    ps_mean: float = 0
    pc_norm: float = 0
    pc_mean: float = 0
    
    # Anja parameters:
    ps_ratio: float = 0
    ps_squared: float = 0
    loglikelihood: float = 0
    
    # check if this should only be one parameter
    mhl_norm: float = 0
    mhl_sum: float = 0
    
    def update(self, parameters: Parameters):
        self.sigma = parameters.sigma
        self.t = parameters.t
        for attr in ('D', 'ps', 'pc'):
            setattr(self, f'{attr}_norm'.lower(), np.linalg.norm(getattr(parameters, attr)))
            setattr(self, f'{attr}_mean'.lower(), np.mean(getattr(parameters, attr)))

        self.ps_squared = np.sum(parameters.ps ** 2)
        self.ps_ratio = np.sqrt(self.ps_squared) / parameters.chiN

        sigma2 = self.sigma ** 2
        
        if hasattr(parameters.population, "x"):
            delta = parameters.population.x.T - parameters.m.T
            self.loglikelihood = -.5 * (parameters.lambda_ * (
                parameters.d * np.log(2 * np.pi * sigma2) + np.log(np.prod(parameters.D) ** 2)) 
                + np.diag(delta.dot(parameters.inv_root_C / sigma2).dot(delta.T)).sum()                
            )
        else:
            delta = np.zeros((5, parameters.d))
            self.loglikelihood = 0        
        
        mhl = np.sqrt(
            np.power(np.dot(parameters.B.T, delta.T) / parameters.D, 2).sum(axis=0)
        ) / self.sigma
        self.mhl_norm = np.linalg.norm(mhl)
        self.mhl_sum = mhl.sum()

            
class TrackedCMAES(ModularCMAES):
    def __init__(self, tracked_parameters = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tracked_parameters = tracked_parameters
        if self.tracked_parameters is not None:
            self.tracked_parameters.update(self.parameters)
        
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
        if self.tracked_parameters is not None:
            self.tracked_parameters.update(self.parameters)
        return res 
            
class From_CMA_To_CMA():
    def __init__(self, budget_factor, dim, A2, total_budget_factor=200):
        self.budget_factor = budget_factor
        self.dim = dim
        self.A2 = A2
        self.total_budget = total_budget_factor*self.dim
        
    def __call__(self, problem, A2, hparams = {}):
        if A2 == "Same":
            budget = self.total_budget
        else:
            budget = self.budget_factor
            
        cma = TrackedCMAES(
                    None, 
                    problem, 
                    self.dim, 
                    budget= budget,
                    active=True,
                    bound_correction='saturate',
                    sigma0 = 2.0,
                    x0 = np.zeros((self.dim,1)),
                    elitist = True
                ).run()
        
        if A2 == "Same":
            return
        
        if A2 == "Non-elitist":
            cma.parameters.elitist = False
            cma.parameters.budget = self.total_budget
        cma.run()
        
        
class Switched_From_CMA():
    def __init__(self, budget_factor, dim, A2, total_budget_factor=200):
        self.budget_factor = budget_factor
        self.dim = dim
        self.A2 = A2
        self.total_budget = total_budget_factor*self.dim
        
    def __call__(self, problem, A2, hparams = {}):
        
        cma = TrackedCMAES(
                    None, 
                    problem, 
                    self.dim, 
                    budget= self.budget_factor,
                    active=True,
                    bound_correction='saturate',
                    sigma0 = 2.0,
                    x0 = np.zeros((self.dim,1)),
                    elitist = True
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

        
def collect_A1_data(budget_factor, dim = 5, time_run=False):
    trigger = ioh.logger.trigger.Always()

    logger = ioh.logger.Analyzer(
        triggers=[trigger],
        folder_name=f'../data/run_data_10D/A1_data_10D/A1_B{budget_factor}_{dim}D',
        algorithm_name='ModCMA_A1',
        store_positions=True
    )
    tracked_parameters = TrackedParameters()
    logger.watch(tracked_parameters, [x.name for x in fields(tracked_parameters)])
    
    for fid in range(1,25):
        for iid in range(1, 6):
            problem = ioh.get_problem(fid, iid, dim, ProblemClass.BBOB)

            # If we want to time the run, wrap the problem in a TimedProblemWrapper
            if time_run:
                problem = TimedProblemWrapper(problem, start_eval=0, stop_eval=1000)
                print(f"Timing enabled for function {fid}, instance {iid}, budget {budget_factor}")
            
            # Attach the logger to the problem
            problem.attach_logger(logger)
            
            for rep in range(20):
                tracked_parameters.rep = rep
                tracked_parameters.iid = iid
                print(f"Running fundction {fid} instance {iid} repetition {rep} with A1, budget {budget_factor}")
                np.random.seed(rep)
                cma = TrackedCMAES(
                    tracked_parameters, 
                    problem, 
                    dim, 
                    budget=budget_factor,
                    active=True,
                    bound_correction='saturate',
                    sigma0 = 2.0,
                    x0 = np.zeros((dim,1)),
                    elitist = True
                ).run()
                problem.reset()
            problem.detach_logger()
            
            
def collect_A2(budget_factor, dim, A2, algname, run_A2_from_scratch=False, time_run=False, find_best=False):
    if budget_factor == 0:
        run_A2_from_scratch = True
    trigger = ioh.logger.trigger.OnImprovement()
    if algname == "BFGS":
        trigger = ioh.logger.trigger.Always()

    logger = ioh.logger.Analyzer(
        triggers=[trigger],
        folder_name=f'../data/run_data_10D/A2_data_10D/A2_{algname}_B{budget_factor}_{dim}D',
        algorithm_name=algname,
        store_positions=True,
    )
    tracked_parameters = TrackedParameters()
    logger.watch(tracked_parameters, [x.name for x in fields(tracked_parameters)])

    for fid in range(1, 25):
        for iid in range(1, 6):

            problem = ioh.get_problem(fid, iid, dim, ProblemClass.BBOB)
            
            # If we want to time the run, wrap the problem in a TimedProblemWrapper
            if time_run:
                problem = TimedProblemWrapper(problem, start_eval=0, stop_eval=1000)

            # Attach the logger to the problem
            problem.attach_logger(logger)

            for rep in range(20):
                tracked_parameters.rep = rep
                tracked_parameters.iid = iid
                print(f"Running function {fid} instance {iid} repetition {rep} with A2 {algname}, budget {budget_factor}, run_from_scratch={run_A2_from_scratch}")
                np.random.seed(rep)
                
                if run_A2_from_scratch:
                    if algname not in ["Same", "Non-elitist"]:
                        # Run A2 directly from scratch without CMA-ES warm-starting
                        algorithm = A2(problem, verbose=False, seed=np.random.get_state())
                        # Run for 1000 evals
                        algorithm.set_params({'budget': 1000})
                        algorithm.set_hyperparams({})
                        def stopping_criteria():
                            return problem.state.evaluations >= 1000
                        
                        algorithm.set_stopping_criteria(stopping_criteria)
                        algorithm.run()
                    else:
                        cma = TrackedCMAES(
                            None, 
                            problem, 
                            dim, 
                            budget= 1000,
                            active=True,
                            bound_correction='saturate',
                            sigma0 = 2.0,
                            x0 = np.zeros((5,1)),
                            elitist = True if algname == "Same" else False
                        ).run()
                    
                elif algname in ["Same", "Non-elitist"]:
                    alg = From_CMA_To_CMA(budget_factor, dim, algname, total_budget_factor=200)
                    alg(problem, algname)
                else:
                    alg = Switched_From_CMA(budget_factor, dim, A2, total_budget_factor=200)
                    alg(problem, A2)
                print("Evaluations:", problem.state.evaluations)
                
                # if find_best:
                #     xs, ys = problem.get_array()

                #     dim = xs.shape[1]
                #     lower_bound = -5
                #     upper_bound = 5
                #     in_bounds_mask = np.all((xs >= lower_bound) & (xs <= upper_bound), axis=1)

                #     xs_in = xs[in_bounds_mask]
                #     ys_in = ys[in_bounds_mask]

                #     if len(ys_in) > 0:
                #         best_precision = np.min(ys_in - problem.optimum.y)
                #         print(f"[Best precision] {best_precision:.6e}")
                #     else:
                #         best_precision = float("nan")
                #         print("[Best precision] No in-bound evaluations found.")
                #     print(best_precision)
                #     # Prepare the output file path
                #     output_file = f"../data/precision_files/{algname}_precisions_{budget_factor}.csv"

                #     # Check if file exists to determine if we should write the header
                #     write_header = not os.path.exists(output_file)

                #     # Create one-row DataFrame and append to file
                #     df = pd.DataFrame([{
                #         "fid": fid,
                #         "iid": iid,
                #         "rep": rep,
                #         "budget": budget_factor,
                #         "algorithm": algname,
                #         "precision": best_precision,
                #     }])
                #     df.to_csv(output_file, mode='a', header=write_header, index=False)

                problem.reset()
                # if find_best:
                #     problem.reset_log()
            problem.detach_logger()
            
def collect_all(x = None):
    budget_factor, dim = x
    # First, collect A1 data
    collect_A1_data(budget_factor, dim, time_run=False)
    
    # Then collect A2 data
    for A2, algname in zip([MLSL, DE, PSO, BFGS, None, None], ["MLSL", "DE", "PSO", "BFGS", "Same", "Non-elitist"]):
        collect_A2(budget_factor, dim, A2, algname, time_run=False)
    # Only run BFGS
    # collect_A2(budget_factor, dim, BFGS, "BFGS", run_A2_from_scratch=False, time_run=False, find_best=False)

    # for A2, algname in zip([MLSL], ["MLSL"]):
    #     collect_A2(budget_factor, dim, A2, algname)
    
    # Only run DE
    # collect_A2(budget_factor, dim, DE, "DE", run_A2_from_scratch=False, time_run=False, find_best=False)
                
                
def get_combinations():
    budget_factors = [8*i for i in range (1,13)] + [50*i for i in range(1, 21)] # 10, 20, ..., 1000
    # budget_factors = [300]
    dim = 10
    return [(bf, dim) for bf in budget_factors]

# if __name__=='__main__':
#     warnings.filterwarnings("ignore", category=RuntimeWarning) 
#     warnings.filterwarnings("ignore", category=FutureWarning)
    
    
#     x = get_combinations()
#     temp = list(x)
#     # collect_all(temp[0])  # Run the first combination for testing
#     for combination in temp:
#         collect_all(combination)
#     partial_run = partial(collect_all)
#     runParallelFunction(partial_run, temp)

#     # problem = ioh.get_problem(1, 1, 5, ProblemClass.BBOB)
#     # print(type(problem))

if __name__ == '__main__':
    warnings.filterwarnings("ignore", category=RuntimeWarning) 
    warnings.filterwarnings("ignore", category=FutureWarning)

    parser = argparse.ArgumentParser()
    parser.add_argument('--budget', type=int, required=True, help='Budget factor (e.g., 100, 200, ...)')
    args = parser.parse_args()
    dim = 10  # Fixed dimensionality!
    budget_factor = args.budget
    collect_all((budget_factor, dim))