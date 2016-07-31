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

from cipy.algorithms.core import comparator
from cipy.algorithms.pso.types import Particle


def std_position(position, velocity):
    """
    Standard particle position update according to the equation:

    :math:`x_{ij}(t+1) = x_{ij}(t) + \
    v_{ij}(t),\\;\\;\\forall\\; j \\in\\; \\{1,...,n\\}`

    Args:
        position (numpy.array): The current position.
        velocity (numpy.array): The particle velocity.

    Returns:
        numpy.array: The calculated position.
    """
    return position + velocity


def std_velocity(particle, social, state):
    """
    Standard particle velocity update according to the equation:

    :math:`v_{ij}(t+1) &= \\omega v_{ij}(t) + \
    c_1 r_{1j}(t)[y_{ij}(t) - x_{ij}(t)]\\:+ \
    c_2 r_{2j}(t)[\\hat{y}_{ij}(t) - x_{ij}(t)],\\;\\;\
    \\forall\\; j \\in\\; \\{1,...,n\\}`

    If a v_max parameter is supplied (state.params['v_max'] is not None) the
    returned velocity is clamped to v_max.

    Args:
        particle (cipy.algorithms.pso.types.Particle): Particle to update the
            velocity for.
        social (numpy.array): The social best for the
            particle.
        state (cipy.algorithms.pso.types.State): The PSO algorithm state.

    Returns:
        numpy.array: The calculated velocity, clamped to state.params['v_max'].
    """
    inertia = state.params['inertia']
    c_1, c_2 = state.params['c_1'], state.params['c_2']
    v_max = state.params['v_max']
    size = particle.position.size

    c1r1 = __acceleration__(state.rng, c_1, size)
    c2r2 = __acceleration__(state.rng, c_2, size)

    velocity = __std_velocity_equation__(inertia, c1r1, c2r2, particle, social)
    return __clamp__(velocity, v_max)


def __std_velocity_equation__(inertia, c1r1, c2r2, particle, social):
    return (inertia * particle.velocity +
            c1r1 * (particle.best_position - particle.position) +
            c2r2 * (social - particle.position))


def __acceleration__(rng, coefficient, size):
    return rng.uniform(0.0, coefficient, size)


def __clamp__(velocity, v_max):
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
    gbest = state.swarm[gbest_idx(state.swarm)].position
    if not np.array_equal(gbest, particle.position):
        return std_velocity(particle, social, state)

    rho = state.params['rho']
    inertia = state.params['inertia']
    v_max = state.params['v_max']
    size = particle.position.size

    r2 = state.rng.uniform(0.0, 1.0, size)
    velocity = __gc_velocity_equation__(particle, gbest, inertia, rho, r2)
    return __clamp__(velocity, v_max)


def __gc_velocity_equation__(inertia, rho, r2, particle, gbest):
    return (-1 * particle.position + gbest + inertia *
            particle.velocity + rho * (1 - 2 * r2))


def std_parameter_update(state, objective_function):
    return state


def init_particle(rng, domain, fitness_function):
    """ Initializes a particle within a domain.
    Args:
        rng: numpy.random.RandomState: The random number generator.
        domain: cipy.problems.core.Domain: The domain of the problem.

    Returns:
        cipy.algorithms.pso.Particle: A new, fully initialized particle.
    """
    position = rng.uniform(domain.lower, domain.upper, domain.dimension)
    fitness = fitness_function(position)
    return Particle(position=position,
                    velocity=np.zeros(domain.dimension),
                    fitness=fitness,
                    best_fitness=fitness,
                    best_position=position)


def update_fitness(objective_function, particle):
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
    fitness = objective_function(particle.position)
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


def update_rho(state, objective_function):
    params = state.params

    rho = params['rho']
    e_s = params['e_s']
    e_f = params['e_f']

    successes = params.get('successes', 0)
    failures = params.get('failures', 0)

    global_best = solution(state.swarm)
    fitness = objective_function(global_best.position)
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


def fitness_measurement(state):
    swarm = state.swarm
    return 'fitness', swarm[gbest_idx(swarm)].best_fitness


def __topology__(swarm, social_best):
    return dict([(idx, social_best(idx)) for idx in range(len(swarm))])
