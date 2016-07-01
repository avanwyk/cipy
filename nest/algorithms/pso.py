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

Function :func:`pso` defines the entry point for running the algorithm.
"""
from collections import namedtuple

import numpy as np

Particle = namedtuple('Particle',
                      ['fitness', 'best_fitness', 'position',
                       'velocity', 'best_position'])


class State(namedtuple('State', ['rng', 'params', 'swarm'])):

    def __call__(self, key):
        return self.params[key]


def global_best(swarm):
    best = None
    for particle in swarm:
        if best is None or particle.best_fitness < best.best_fitness:
            best = particle
    return best


def gbest(state, idx):
    best = state.swarm[idx]
    for particle in state.swarm:
        if particle.best_fitness < best.best_fitness:
            best = particle
    return best.best_position


def lbest(state, idx):
    """ Lbest topology function.

    Args:
        :param state: State: PSO algorithm state.
        :param idx: index of particle in neighbourhood.

    :return: Particle: the locally best particle position.
    """
    swarm = state.swarm
    n_s = state('n_s')
    start = idx - (n_s // 2)
    best = None
    size = len(swarm)
    for k in range(n_s):
        particle = swarm[(start + k) % size]
        if best is None or particle.best_fitness < best.best_fitness:
            best = particle
    return best.best_position


def standard_position(position, velocity):
    return position + velocity


def standard_velocity(particle, social, state):
    """ Standard particle velocity update.

    Args:
        :param particle: Particle: particle to update velocity for.
        :param social: social best position of the particle.
        :param state: PSO algorithm state.

    :return: numpy.ndarray: updated velocity of particle.
    """
    inertia, c_1, c_2 = state('inertia'), state('c_1'), state('c_2')
    size = particle.position.size
    c_1r_1 = state.rng.uniform(0.0, c_1, size)
    c_2r_2 = state.rng.uniform(0.0, c_2, size)
    return inertia * particle.velocity + \
           c_1r_1 * (particle.best_position - particle.position) + \
           c_2r_2 * (social - particle.position)


def clamp(velocity, v_max):
    return np.clip(velocity, -v_max, v_max)


def standard_velocity_with_v_max(particle, social, state):
    return clamp(standard_velocity(particle, social, state), state('v_max'))


def update_particle(state, idx_particle):
    (idx, particle) = idx_particle
    nbest = state('topology')(state, idx)
    velocity = standard_velocity_with_v_max(particle, nbest, state)
    position = standard_position(particle.position, velocity)
    return particle._replace(position=position, velocity=velocity)


def update_fitness(fitness_f, particle):
    fitness = fitness_f(particle.position)
    if particle.best_fitness is None or fitness < particle.best_fitness:
        best_position = particle.position
        return particle._replace(fitness=fitness,
                                 best_fitness=fitness,
                                 best_position=best_position)
    else:
        return particle._replace(fitness=fitness)


def init_particle(rng, domain):
    position = rng.uniform(domain.lower, domain.upper, domain.dimension)
    return Particle(position=position,
                    velocity=np.zeros(domain.dimension),
                    fitness=None,
                    best_fitness=None,
                    best_position=position)


def init_swarm(rng, size, domain):
    return [init_particle(rng, domain) for particle in range(size)]


def pso(fitness, iterations, parameters=None):
    """ Perform particle swarm optimization of the given fitness function.

    Args:
        :param fitness: fitness function.
        :param iterations: int: number of iterations to execute PSO for.
        :param parameters: dictionary: parameter dictionary for the PSO.

    :return: Particle: the global best particle.
    """
    defaults = {'size': 25, 'n_s': 5, 'inertia': 0.729844,
                'c_1': 1.496180, 'c_2': 1.496180, 'v_max': 0.5,
                'topology': lbest, 'seed': None}
    params = {**defaults, **({} if parameters is None else parameters)}

    rng = np.random.RandomState(params['seed'])
    state = State(rng, params, init_swarm(rng, params['size'], fitness.domain))

    for iteration in range(iterations):
        state = state._replace(swarm=[update_fitness(fitness, particle)
                                      for particle in state.swarm])
        state = state._replace(swarm=[update_particle(state, ip)
                                      for ip in enumerate(state.swarm)])

    return global_best(state.swarm)
