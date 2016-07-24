# Copyright 2016 Andrich van Wyk
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

""" Collection of functions used to implement the PSO algorithm.
The implementation defined here is the synchronous modified gbest PSO with the
inertia term as per:

* Shi, Yuhui, and Russell Eberhart. "A modified particle swarm optimizer."
  Evolutionary Computation Proceedings, 1998.
  IEEE World Congress on Computational Intelligence.,
  The 1998 IEEE International Conference on. IEEE, 1998.

Function 'pso' defines the entry point for running the algorithm.
"""
import numpy as np

from cipy.algorithms.pso import functions
from cipy.algorithms.pso import types


def optimize(problem, stopping_condition, parameters=None,
             position_update=functions.std_position,
             velocity_update=functions.std_velocity_with_v_max,
             parameter_update=functions.std_parameter_update):
    """ Perform particle swarm optimization of the given fitness function.
    Args:
        problem: optimization problem encapsulating the fitness function.
        stopping_condition: function specifying the stopping condition.
        parameters: dictionary: parameter dictionary for the PSO.

    Returns:
        cipy.algorithms.pso.Particle: The global best particle.
    """
    params = __init_parameters__(parameters)

    rng = np.random.RandomState(params['seed'])

    initial_swarm = [init_particle(rng, problem.domain)
                     for i in range(params['swarm_size'])]
    state = types.State(rng, params, initial_swarm, iterations=0)

    topology_function = state.params['topology']
    update_fitness = functions.update_fitness
    update_particle = functions.update_particle

    while not stopping_condition(state):
        state = state._replace(swarm=[update_fitness(problem, particle)
                                      for particle in state.swarm])

        n_bests = topology_function(state)

        state = state._replace(swarm=[update_particle(position_update,
                                                      velocity_update,
                                                      state, n_bests, ip)
                                      for ip in enumerate(state.swarm)],
                               iterations=state.iterations + 1)

        state = parameter_update(state, problem)

    return functions.solution(state.swarm)


def init_particle(rng, domain):
    """ Initializes a particle within a domain.
    Args:
        rng: numpy.random.RandomState: The random number generator.
        domain: cipy.problems.core.Domain: The domain of the problem.

    Returns:
        cipy.algorithms.pso.Particle: A new, fully initialized particle.
    """
    position = rng.uniform(domain.lower, domain.upper, domain.dimension)
    return types.Particle(position=position,
                          velocity=np.zeros(domain.dimension),
                          fitness=None,
                          best_fitness=None,
                          best_position=position)


def default_parameters():
    return {'swarm_size': 25, 'n_s': 5, 'inertia': 0.729844,
            'c_1': 1.496180, 'c_2': 1.496180, 'v_max': None,
            'topology': functions.gbest_topology, 'seed': None}


def __init_parameters__(params):
    return {**default_parameters(), **({} if params is None else params)}