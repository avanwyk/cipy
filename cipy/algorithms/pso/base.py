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

from cipy.algorithms.core import dictionary_based_measurements
from cipy.algorithms.pso import functions
from cipy.algorithms.pso import types


def optimize(objective_function, domain,
             stopping_condition, parameters=None,
             position_update=functions.std_position,
             velocity_update=functions.std_velocity_with_v_max,
             parameter_update=functions.std_parameter_update,
             measurements=(),
             measurer=dictionary_based_measurements):
    """ Perform particle swarm optimization of the given fitness function.
    Args:
        objective_function: the cost function to optimize.
        stopping_condition: function specifying the stopping condition.
        parameters: dictionary: parameter dictionary for the PSO.

    Returns:
        cipy.algorithms.pso.Particle: The global best particle.
    """
    params = __init_parameters__(parameters)

    rng = np.random.RandomState(params['seed'])

    initial_swarm = [functions.init_particle(rng, domain,
                                             objective_function)
                     for i in range(params['swarm_size'])]
    state = types.PSOState(rng, params, iterations=0, swarm=initial_swarm)

    topology_function = state.params['topology']
    update_fitness = functions.update_fitness
    update_particle = functions.update_particle

    results, measure = measurer(measurements)
    while not stopping_condition(state):
        n_bests = topology_function(state)

        state = state._replace(swarm=[update_particle(position_update,
                                                      velocity_update,
                                                      state, n_bests, ip)
                                      for ip in enumerate(state.swarm)])

        state = state._replace(swarm=[update_fitness(objective_function,
                                                     particle)
                                      for particle in state.swarm],
                               iterations=state.iterations + 1)

        state = parameter_update(state, objective_function)

        results = measure(results, state)

    return functions.solution(state.swarm), results


def default_parameters():
    return {'swarm_size': 25, 'n_s': 5, 'inertia': 0.729844,
            'c_1': 1.496180, 'c_2': 1.496180, 'v_max': None,
            'topology': functions.gbest_topology, 'seed': None}


def __init_parameters__(params):
    return {**default_parameters(), **({} if params is None else params)}
