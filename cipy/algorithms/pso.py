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
from collections import namedtuple

import numpy as np

Particle = namedtuple('Particle',
                      ['position', 'velocity', 'fitness',
                       'best_fitness', 'best_position'])


State = namedtuple('State', ['rng', 'params', 'swarm', 'iterations'])


def global_best(optimal, swarm):
    """ Determines the global best particle in the swarm.

    Args:
        optimal: function(a,b): boolean function comparing a, b for optimality
        swarm: iterable: an iterable that yields all particles in the swarm.

    Returns:
        cipy.algorithms.pso.Particle: The best particle in the swarm when
        comparing the best_fitness values of the particles.
    """
    best = None
    for particle in swarm:
        if best is None or optimal(particle.best_fitness, best.best_fitness):
            best = particle
    return best


def gbest(optimal, state, idx):
    """ gbest Neighbourhood topology function.

    Args:
        optimal: function(a,b): boolean function comparing a, b for optimality
        state: cipy.algorithms.pso.State: The state of the PSO algorithm.
        idx: int: index of particle in the swarm.

    Returns:
        cipy.algorithms.pso.Particle: The best particle in the swarm when
        comparing the best_fitness values of the particles.
    """
    best = state.swarm[idx]
    for particle in state.swarm:
        if optimal(particle.best_fitness, best.best_fitness):
            best = particle
    return best.best_position


def lbest(optimal, state, idx):
    """ lbest Neighbourhood topology function.

    Neighbourhood size is determined by state.params['n_s'].

    Args:
        optimal: function(a,b): boolean function comparing a, b for optimality
        state: cipy.algorithms.pso.State: The state of the PSO algorithm.
        idx: int: index of particle in the swarm.

    Returns:
        cipy.algorithms.pso.Particle: The best particle local
        (in the neighbourhood) to the indexed particle.
    """
    swarm = state.swarm
    n_s = state.params['n_s']
    start = idx - (n_s // 2)
    best = None
    size = len(swarm)
    for k in range(n_s):
        particle = swarm[(start + k) % size]
        if best is None or optimal(particle.best_fitness, best.best_fitness):
            best = particle
    return best.best_position


def std_position(position, velocity):
    """ Standard particle position update.

    Args:
        position: numpy.ndarray: The current position.
        velocity: numpy.ndarray: The particle velocity.

    Returns:
        numpy.ndarray: the calculated position.
    """
    return position + velocity


def std_velocity(particle, social, state):
    """ Standard particle velocity update.

    Args:
        particle: cipy.algorithms.pso.Particle: Particle to update the velocity
            for.
        social: cipy.algorithms.pso.Particle: The social best for the particle.
        state: cipy.algorithms.pso.State: The state of the PSO algorithm.

    Returns:
        numpy.ndarray: the calculated velocity.
    """
    inertia = state.params['inertia']
    c_1, c_2 = state.params['c_1'], state.params['c_2']
    size = particle.position.size

    c_1r_1 = state.rng.uniform(0.0, c_1, size)
    c_2r_2 = state.rng.uniform(0.0, c_2, size)

    return (inertia * particle.velocity +
            c_1r_1 * (particle.best_position - particle.position) +
            c_2r_2 * (social - particle.position))


def clamp(velocity, v_max):
    return velocity if v_max is None else np.clip(velocity, -v_max, v_max)


def std_velocity_with_v_max(particle, social, state):
    return clamp(std_velocity(particle, social, state), state.params['v_max'])


def update_particle(optimal, state, idx_particle):
    """ Update function for a particle.

    Calculates and updates the velocity and position of a particle for a
    single iteration of the PSO algorithm. Social best particle is determined by
    the state.params['topology'] function.

    Args:
        optimal: function(a,b): boolean function comparing a, b for optimality
        state: cipy.algorithms.pso.State: The state of the PSO algorithm.
        idx_particle: tuple: Tuple of the index of the particle and the
            particle itself.

    Returns:
        cipy.algorithms.pso.Particle: A new particle with the updated position
        and velocity.

    """
    (idx, particle) = idx_particle

    nbest = state.params['topology'](optimal, state, idx)

    velocity = std_velocity_with_v_max(particle, nbest, state)
    position = std_position(particle.position, velocity)
    return particle._replace(position=position, velocity=velocity)


def update_fitness(problem, particle):
    """ Calculates and updates the fitness and best_fitness of a particle.

    Fitness is calculated using the 'problem.fitness' function.

    Args:
        problem: The optimization problem encapsulating the fitness function and
            optimization type.
        particle: cipy.algorithms.pso.Particle: Particle to update the fitness
            for.

    Returns:
        cipy.algorithms.pso.Particle: A new particle with the updated fitness.

    """
    fitness = problem.fitness(particle.position)
    best_fitness = particle.best_fitness
    if best_fitness is None or problem.optimal(fitness, best_fitness):
        best_position = particle.position
        return particle._replace(fitness=fitness,
                                 best_fitness=fitness,
                                 best_position=best_position)
    else:
        return particle._replace(fitness=fitness)


def init_particle(rng, domain):
    """ Initializes a particle within a domain.
    Args:
        rng: numpy.random.RandomState: The random number generator.
        domain: cipy.problems.core.Domain: The domain of the problem.

    Returns:
        cipy.algorithms.pso.Particle: A new, fully initialized particle.
    """
    position = rng.uniform(domain.lower, domain.upper, domain.dimension)
    return Particle(position=position,
                    velocity=np.zeros(domain.dimension),
                    fitness=None,
                    best_fitness=None,
                    best_position=position)


def init_swarm(rng, size, domain):
    return [init_particle(rng, domain) for particle in range(size)]


def default_parameters():
    return {'swarm_size': 25, 'n_s': 5, 'inertia': 0.729844,
            'c_1': 1.496180, 'c_2': 1.496180, 'v_max': None,
            'topology': gbest, 'seed': None}


def __init_parameters__(params):
    return {**default_parameters(), **({} if params is None else params)}


def pso(problem, stopping_condition, parameters=None):
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
    state = State(rng, params,
                  init_swarm(rng, params['swarm_size'], problem.domain),
                  iterations=0)
    optimal = problem.optimal

    while not stopping_condition(state):
        state = state._replace(swarm=[update_fitness(problem, particle)
                                      for particle in state.swarm])
        state = state._replace(swarm=[update_particle(optimal, state, ip)
                                      for ip in enumerate(state.swarm)],
                               iterations=state.iterations+1)

    return global_best(problem.optimal, state.swarm)
