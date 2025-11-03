# import random as rd
import math
import numpy as np
import copy
from algorithm import Algorithm

# setting up numpy random
# np.random.seed()
# generator = np.random.default_rng()

class Particle:
    """ Create one particle for the swarm.

    Parameters:
    --------------
    dim : int
        Number of problem variables, dimensionality.

    Attributes:
    --------------
    position : array-like
        The current position of the particle.

    velocity : array-like
        The current velocity of the particle.

    fitness : float
        The current fitness value of the particle.

    pos_best : array-like
        The all-time best position found by the particle.

    best_fitness : float
        The fitness value at the all-time best position.

    """

    def __init__(self, dim, vel_range = 5):
        self.dim = dim
        self.position = np.zeros(self.dim)
        self.velocity = np.zeros(self.dim)
        self.fitness = np.inf
        self.pos_best = np.zeros(self.dim)
        self.best_fitness = np.inf
        self.velocity = np.random.uniform(-1 * vel_range, vel_range, self.dim) #TODO: set based on func
        self.position = np.random.uniform(-5, 5, self.dim)
        # initialize particle with uniform random position and velocities
#         for i in range(0, self.dim):
#             self.velocity[i] = generator.uniform(low= -5, high=5)
#             self.position[i] = generator.uniform(low= -5, high=5)

    def update_velocity(self, pos_best_g, w, c1 = 1.4944, c2 = 1.4944, vel_range = 5):
        """ Update the particle's velocity.

        Parameters:
        --------------
        pos_best_g : array-like
            The best position found by the swarm.
            
        w : float
            Inertia weight to control velocity update.

        """
        
        vel_lowerbound = -1 * vel_range   # lower bound for velocity
        vel_upperbound = vel_range    # upper bound for velocity

        # create uniform random vectors between 0 and c1 / c2
        u1 = np.random.uniform(0, c1, self.dim)
        u2 = np.random.uniform(0, c2, self.dim)

        """ update velocity with previous velocity and component-wise multiplication of u-vectors
        with differences between current solution, previous best and global best"""
        for i in range(0, self.dim):
            self.velocity[i] = w * self.velocity[i] + u1[i] * (self.pos_best[i]
                                                    - self.position[i]) + u2[i] * (pos_best_g[i] - self.position[i])

        self.velocity = np.clip(self.velocity, vel_lowerbound, vel_upperbound)

    def update_position(self):
        """ Update a particle's position.

        """
        for i in range(0, self.dim):
            result = self.position[i] + self.velocity[i]

            # check boundary conditions, reset velocity to zero if particle is on boundary
            if result < -5:
                self.position[i] = -5
                self.velocity[i] = 0
            elif result > 5:
                self.position[i] = 5
                self.velocity[i] = 0
            else:
                self.position[i] = result

class PSO(Algorithm):
    """ Class for the PSO algorithm.

    Parameters:
    ---------------
    None

    Attributes:
    ---------------
    g_best : array-type
        Best solution found so far.
        
    g_best_f : float
        Fitness value at best solution so far.
    
    Methods:
    --------------
    
    set_params(parameters):
        Sets algorithm parameters for warm-start.
        
    get_params(parameters):
        Transfers internal parameters to parameters dictionary.
        
    run():
        Runs the EA algorithm.

    Parent:
    ------------

    """
    __doc__ += Algorithm.__doc__

    def __init__(self, func, **kwargs):
        super(PSO, self).__init__(func, **kwargs) 
        self.g_best = None
        self.g_best_f = np.inf
        self.popsize = 40
        self.swarm = None
        self.vel_overtime = []
        self.eval_count = []
        self.c1 = 1.4944     # social constant
        self.c2 = 1.4944     # cognitive constant
        self.vel_range = 1
        
    @classmethod
    def get_hyperparams(cls):
        return {'popsize' : {'type' : 'i', 'range' : [4,80]},
                'c1' : {'type' : 'r', 'range' : [0,2.5]},
                'c2' : {'type' : 'r', 'range' : [0,2.5]},
                'vel_range' : {'type' : 'r', 'range' : [0.1,5]},
               }   
     
    def set_hyperparams(self, hyperparam_dict):
        super(PSO, self).set_hyperparams(hyperparam_dict)
        self.popsize = int(self.popsize)
        
    def set_params(self, parameters):
        self.budget = parameters['budget']

        """Warm start routine"""
        
        ## Initialize swarm 
        # Stratey 1: Initialize swarm in the neighbourhood of the best point found by A1
        
        if 'x_opt' in parameters:
            x_opt = parameters['x_opt']
            self.swarm = []
            eta = 0.1
            #vmax_slow = 1
            #vmin_slow = -vmax_slow
            vel_vector = []
            for i in range(0, self.popsize):
                self.swarm.append(Particle(self.dim, self.vel_range))   # append objects of class Particle
                
                # update particle position and velocity
                for j in range(0, self.dim):
                    self.swarm[i].position[j] = x_opt[j] + np.random.uniform(low = -eta, high = eta)
                    #self.swarm[i].velocity[j] = generator.uniform(low = vmin_slow, high = vmax_slow)
                
                # Clip particle position to bounds
                # Clip to bounds after warmstarting
                self.swarm[i].position = np.clip(
                    self.swarm[i].position,
                    self.func.bounds.lb,
                    self.func.bounds.ub
                )
                
                # update fitness and best positions / fitness
                if self.uses_old_ioh:
                    self.swarm[i].fitness = self.func(self.swarm[i].position)   # evaluate fitness for each particle
                else:
                    self.swarm[i].fitness = self.func(self.swarm[i].position)   # evaluate fitness for each particle
                self.swarm[i].pos_best = self.swarm[i].position.copy()   # set personal_best to current position
                self.swarm[i].best_fitness = self.swarm[i].fitness   # set best_fitness to current particle's fitness
                
                # save information for plots
                vel_vector.append(np.linalg.norm(self.swarm[i].velocity))
                self.x_hist.append(self.swarm[i].position.copy())
                self.f_hist.append(self.swarm[i].fitness)
                
                # update global best if particle is best solution so far
                if self.swarm[i].fitness < self.g_best_f:
                    self.g_best = self.swarm[i].position.copy()
                    self.g_best_f = self.swarm[i].fitness
                
            self.vel_overtime.append(np.asarray(vel_vector))
#             self.eval_count.append(self.func.evaluations)
#             parameters.internal_dict['init_swarm'] = self.x_hist.copy()
        
        # Initialize swarm with best points found by A1
        """if 'mlsl_x_hist' in parameters.internal_dict:

            x_hist = parameters.internal_dict['mlsl_x_hist']
            f_hist = parameters.internal_dict['mlsl_f_hist']

            best_points = np.zeros((self.popsize, self.func.number_of_variables))
            m = np.hstack((np.asarray(x_hist), np.expand_dims(np.asarray(f_hist), axis=1)))
            sorted_m = m[np.argsort(m[:, self.func.number_of_variables])]
            best_points = sorted_m[0:len(best_points), 0:self.func.number_of_variables]
            
            self.swarm = []
            for i in range(0, self.popsize):
                self.swarm.append(Particle(self.dim))   # append objects of class Particle
                # update particle position and velocity

                self.swarm[i].position = best_points[i]
                self.x_hist.append(self.swarm[i].position.copy())     
                self.swarm[i].fitness = self.func(self.swarm[i].position)   # evaluate fitness for each particle
                self.f_hist.append(self.swarm[i].fitness)
                self.swarm[i].pos_best = self.swarm[i].position.copy()   # set personal_best to current position
                self.swarm[i].best_fitness = self.swarm[i].fitness   # set best_fitness to current particle's fitness
                
                # update global best if particle is best solution so far
                if self.swarm[i].fitness < self.g_best_f:
                    self.g_best = self.swarm[i].position.copy()
                    self.g_best_f = self.swarm[i].fitness
                
            parameters.internal_dict['init_swarm'] = self.x_hist.copy()"""

    def get_params(self, parameters):
        parameters = super(PSO, self).get_params(parameters)
#         parameters['x_opt'] = self.func.best_so_far_variables
#         parameters['pso_x_opt'] = self.func.best_so_far_variables
#         parameters['pso_x_hist'] = self.x_hist.copy()
#         parameters['pso_f_hist'] = self.f_hist.copy()
        parameters['vel_overtime'] = self.vel_overtime
#         parameters['evalcount'] = self.eval_count
    
        return parameters


    def run(self):
        """ Runs the PSO algorithm.

        Parameters:
        --------------
        None

        Returns:
        --------------
        best_so_far_variables : array
                The best found solution.

        best_so_far_fvaluet: float
               The fitness value for the best found solution.

        """
        if self.verbose:
            print(f'PSO started')

        # establish the swarm
        if self.swarm is None:   # only establish swarm if not initialized by set_params
            self.swarm = []
            for i in range(0, self.popsize):
                self.swarm.append(Particle(self.dim, self.vel_range))   # append objects of class Particle
                self.x_hist.append(self.swarm[i].position.copy())
                if self.uses_old_ioh:
                    self.swarm[i].fitness = self.func(self.swarm[i].position) # evaluate fitness for each particle
                else:
                    self.swarm[i].fitness = self.func(self.swarm[i].position) # evaluate fitness for each particle

                self.f_hist.append(self.swarm[i].fitness)   
                self.swarm[i].pos_best = self.swarm[i].position.copy()   # set personal_best to current position
                self.swarm[i].best_fitness = self.swarm[i].fitness   # set best_fitness to current particle's fitness

                # update global best if particle is best solution so far
                if self.swarm[i].fitness < self.g_best_f:
                    self.g_best = self.swarm[i].position.copy()
                    self.g_best_f = self.swarm[i].fitness

        # evaluation loop
        while not self.stop():
#             if self.verbose:
#                 print(f"starting loop: {self.func.evaluations} used")
            # Update inertia weight
            if self.uses_old_ioh:
                w = 0.9 - 0.8 * self.func.evaluations / self.budget
            else:
                w = 0.9 - 0.8 * self.func.state.evaluations / self.budget

            # Iterate through particle swarm
            vel_vector = []
            for k in range(0, self.popsize):
                self.swarm[k].update_velocity(self.g_best, w, self.c1, self.c2, self.vel_range)    # update velocity based on global_best and inertia weight
                vel_vector.append(np.linalg.norm(self.swarm[k].velocity))
                self.swarm[k].update_position()
                if self.uses_old_ioh:
                    self.swarm[k].fitness = self.func(self.swarm[k].position)    # evaluate the particle's fitness    
                else:
                    self.swarm[k].fitness = self.func(self.swarm[k].position)   # evaluate the particle's fitness    
                
                self.x_hist.append(self.swarm[k].position.copy())
                self.f_hist.append(self.swarm[k].fitness)

                # check if new particle position is best-so-far for this particle
                if self.swarm[k].fitness < self.swarm[k].best_fitness:
                    self.swarm[k].pos_best = self.swarm[k].position.copy()
                    self.swarm[k].best_fitness = self.swarm[k].fitness

                # check if new particle position is best-so-far globally
                if self.swarm[k].fitness < self.g_best_f:
                    self.g_best = self.swarm[k].position.copy()
                    self.g_best_f = self.swarm[k].fitness

            self.vel_overtime.append(np.asarray(vel_vector))
#             self.eval_count.append(self.func.evaluations)
            
        if self.verbose:
            print(f'PSO complete')
#             print(f'prec: {self.func.best_so_far_precision} evals: {self.func.evaluations}')

        # return best global solution including its fitness value
#         return self.func.best_so_far_variables, self.func.best_so_far_fvalue