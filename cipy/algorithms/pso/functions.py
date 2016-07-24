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
"""

import numpy as np

from cipy.algorithms.pso.types import Particle
from cipy.problems.core import comparator


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

    c_1r_1 = c_1 * state.rng.uniform(0.0, 1.0, size)
    c_2r_2 = c_2 * state.rng.uniform(0.0, 1.0, size)

    return (inertia * particle.velocity +
            c_1r_1 * (particle.best_position - particle.position) +
            c_2r_2 * (social - particle.position))


def std_velocity_with_v_max(particle, social, state):
    return clamp(std_velocity(particle, social, state), state.params['v_max'])


def clamp(velocity, v_max):
    return velocity if v_max is None else np.clip(velocity, -v_max, v_max)


def gc_velocity_update(particle, social, state):
    """ Guaranteed convergence velocity update.

    Args:
        particle: cipy.algorithms.pso.Particle: Particle to update the velocity
            for.
        social: cipy.algorithms.pso.Particle: The social best for the particle.
        state: cipy.algorithms.pso.State: The state of the PSO algorithm.

    Returns:
        numpy.ndarray: the calculated velocity.
    """
    gbest_position = state.swarm[gbest_idx(state.swarm)].position
    if not np.array_equal(gbest_position, particle.position):
        return std_velocity_with_v_max(particle, social, state)

    rho = state.params['rho']
    inertia = state.params['inertia']
    size = particle.position.size

    r2 = state.rng.uniform(0.0, 1.0, size)
    return (-1 * particle.position + gbest_position + inertia *
            particle.velocity + rho * (1 - 2 * r2))


def std_parameter_update(state, problem):
    return state


def default_parameters():
    return {'swarm_size': 25, 'n_s': 5, 'inertia': 0.729844,
            'c_1': 1.496180, 'c_2': 1.496180, 'v_max': None,
            'topology': gbest_topology, 'seed': None}


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


def update_fitness(problem, particle):
    """ Calculates and updates the fitness and best_fitness of a particle.

    Fitness is calculated using the 'problem.fitness' function.

    Args:
        problem: The optimization problem encapsulating the fitness function
            and optimization type.
        particle: cipy.algorithms.pso.Particle: Particle to update the fitness
            for.

    Returns:
        cipy.algorithms.pso.Particle: A new particle with the updated fitness.

    """
    fitness = problem.fitness(particle.position)
    best_fitness = particle.best_fitness
    cmp = comparator(fitness)
    if best_fitness is None or cmp(fitness, best_fitness):
        best_position = particle.position
        return particle._replace(fitness=fitness,
                                 best_fitness=fitness,
                                 best_position=best_position)
    else:
        return particle._replace(fitness=fitness)


def update_particle(position_update, velocity_update, state, nbest_topology,
                    idx_particle):
    """ Update function for a particle.

    Calculates and updates the velocity and position of a particle for a
    single iteration of the PSO algorithm. Social best particle is determined
    by the state.params['topology'] function.

    Args:
        state: cipy.algorithms.pso.State: The state of the PSO algorithm.
        nbest_topology: dict: Containing neighbourhood best index for each
            particle index.
        idx_particle: tuple: Tuple of the index of the particle and the
            particle itself.

    Returns:
        cipy.algorithms.pso.Particle: A new particle with the updated position
        and velocity.

    """
    (idx, particle) = idx_particle

    nbest = state.swarm[nbest_topology[idx]].best_position

    velocity = velocity_update(particle, nbest, state)
    position = position_update(particle.position, velocity)
    return particle._replace(position=position, velocity=velocity)


def gbest_topology(state):
    gbest = gbest_idx(state.swarm)
    return __topology__(state.swarm, lambda i: gbest)


def gbest_idx(swarm):
    """ gbest Neighbourhood topology function.

    Args:
        swarm: list: The list of particles.

    Returns:
        int: The index of the gbest particle.
    """
    best = 0
    cmp = comparator(swarm[best].best_fitness)
    for (idx, particle) in enumerate(swarm):
        if cmp(particle.best_fitness, swarm[best].best_fitness):
            best = idx
    return best


def lbest_topology(state):
    return __topology__(state.swarm, lambda i: lbest_idx(state, i))


def lbest_idx(state, idx):
    """ lbest Neighbourhood topology function.

    Neighbourhood size is determined by state.params['n_s'].

    Args:
        state: cipy.algorithms.pso.State: The state of the PSO algorithm.
        idx: int: index of the particle in the swarm.

    Returns:
        int: The index of the lbest particle.
    """
    swarm = state.swarm
    n_s = state.params['n_s']
    start = idx - (n_s // 2) + 1
    best = start - 1
    size = len(swarm)
    cmp = comparator(swarm[best].best_fitness)
    for k in range(n_s):
        idx = (start + k) % size
        particle = swarm[idx]
        if cmp(particle.best_fitness, swarm[best].best_fitness):
            best = idx
    return best


def update_rho(state, problem):
    params = state.params

    rho = params['rho']
    e_s = params['e_s']
    e_f = params['e_f']

    successes = params.get('successes', 0)
    failures = params.get('failures', 0)

    global_best = solution(state.swarm)
    fitness = problem.fitness(global_best.position)
    cmp = comparator(global_best.best_fitness)
    if cmp(fitness, global_best.best_fitness):
        successes += 1
        failures = 0
    else:
        failures += 1
        successes = 0

    if successes > e_s:
        rho *= 2
    elif failures > e_f:
        rho *= 0.5
    else:
        rho = rho

    params['rho'] = rho
    params['successes'] = successes
    params['failures'] = failures

    return state._replace(params=params)


def solution(swarm):
    """ Determines the global best particle in the swarm.

    Args:
        swarm: iterable: an iterable that yields all particles in the swarm.

    Returns:
        cipy.algorithms.pso.Particle: The best particle in the swarm when
        comparing the best_fitness values of the particles.
    """
    best = swarm[0]
    cmp = comparator(best.best_fitness)
    for particle in swarm:
        if cmp(particle.best_fitness, best.best_fitness):
            best = particle
    return best


def __topology__(swarm, social_best):
    return dict([(idx, social_best(idx)) for idx in range(len(swarm))])


def __init_parameters__(params):
    return {**default_parameters(), **({} if params is None else params)}
