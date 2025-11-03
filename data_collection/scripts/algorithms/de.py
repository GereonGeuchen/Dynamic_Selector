from scipy.optimize._differentialevolution import DifferentialEvolutionSolver
# from scipy_differentialevolution import DifferentialEvolutionSolver
# import numpy as np
from algorithm import Algorithm
import numpy as np
import copy
from scipy.optimize import Bounds
# from scipy._lib._util import check_random_state
# from scipy.optimize import differential_evolution

class DE(Algorithm):
    """DE algorithm
    """
    __doc__ += Algorithm.__doc__
    
    def __init__(self, func, **kwargs):
        super(DE, self).__init__(func, **kwargs)
        if self.uses_old_ioh:
            bounds = Bounds(self.func.lowerbound, self.func.upperbound)
            self.de_wrapper = DifferentialEvolutionSolver(
                internal_eval, 
                bounds=bounds, 
                tol=0,                # ← disable relative convergence
                atol=0               # ← disable absolute convergence
            )
        else:
            bounds = Bounds(self.func.bounds.lb, self.func.bounds.ub)
            def internal_eval(x):
                return self.func(x)
            self.de_wrapper = DifferentialEvolutionSolver(
                internal_eval, 
                bounds=bounds, 
                tol=0,                # ← disable relative convergence
                atol=0               # ← disable absolute convergence
            )
            #         self.random_number_generator = check_random_state(seed=None)

    def set_params(self, parameters):
        self.budget = parameters['budget']
        
        # Warmstarting
        # Warm-start population around x_opt
  
        if 'x_opt' in parameters:
            x_opt = parameters['x_opt']
            eta = 0.1
            scale_arg = 10   # important for scaling variables to interval between 0 and 1
#             rng = self.random_number_generator
            
            for i in range(0, self.de_wrapper.num_population_members):
                for j in range(0, self.dim):
                    self.de_wrapper.population[i][j] = (x_opt[j] + np.random.uniform(low=-eta, high=eta)) / scale_arg + 0.5
                    # Clip value to bounds
                    self.de_wrapper.population[i][j] = np.clip(self.de_wrapper.population[i][j], 0, 1)
         

            # reset population energies
            self.de_wrapper.population_energies = np.full(self.de_wrapper.num_population_members,
                                           np.inf)
            
            #print(f'init pop: {self.de_wrapper.population}')


    @classmethod
    def get_hyperparams(cls):
        return {#'num_population_members' : {'type' : 'i', 'range' : [5,80]}, Disabled because of bugs
                'cross_over_probability' : {'type' : 'r', 'range' : [0,1]},
               }   
     
    def set_hyperparams(self, hyperparam_dict):
#         hyperparam_dict['num_population_members'] = int(hyperparam_dict['num_population_members'])
        for k,v in hyperparam_dict.items():
            setattr(self.de_wrapper, k, v)        
        
    def get_params(self, parameters):
        parameters = super(DE, self).get_params(parameters)

#         parameters['x_opt'] = self.func.best_so_far_variables
#         parameters['de_x_opt'] = self.func.best_so_far_variables
#         parameters['de_x_hist'] = self.de_wrapper.a2_x_hist
#         parameters['de_f_hist'] = self.de_wrapper.f_hist
        parameters['de_gen_counter'] = self.de_wrapper.generation_counter

        return parameters
    
    def run(self):
        if self.verbose:
            print(f'DE started')
        self.de_wrapper.maxfun = self.budget
        self.de_wrapper.stop = self.stop
        self.de_wrapper.solve()
        
        if self.verbose:
            print(f'DE complete')
#             print(f'evals: {self.func.evaluations} prec: {self.func.best_so_far_precision}')

#         return self.func.best_so_far_variables, self.func.best_so_far_fvalue


