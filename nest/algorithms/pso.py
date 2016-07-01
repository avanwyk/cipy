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
    for p in swarm:
        if best is None or p.best_fitness < best.best_fitness:
            best = p
    return best


def gbest(state, idx):
    best = None
    for p in state.swarm:
        if best is None or p.best_fitness < best.best_fitness:
            best = p
    return best.best_position


def lbest(state, idx):
    swarm = state.swarm
    n_s = state('n_s')
    start = idx - (n_s // 2)
    best = None
    size = len(swarm)
    for k in range(n_s):
        p = swarm[(start + k) % size]
        if best is None or p.best_fitness < best.best_fitness:
            best = p
    return best.best_position


def standard_position(position, velocity):
    return position + velocity


def standard_velocity(particle, social, state):
    inertia, c_1, c_2 = state('inertia'), state('c_1'), state('c_2')
    size = particle.position.size
    c_1r_1 = state.rng.uniform(0.0, c_1, size)
    c_2r_2 = state.rng.uniform(0.0, c_2, size)
    return inertia * particle.velocity + \
        c_1r_1 * (particle.best_position - particle.position) + \
        c_2r_2 * (social - particle.position)


def clamp(velocity, v_max):
    return np.clip(velocity, -v_max, v_max)


def standard_velocity_with_vmax(particle, social, state):
    return clamp(standard_velocity(particle, social, state), state('v_max'))


def update_particle(state, idx_particle):
    (idx, particle) = idx_particle
    nbest = state('topology')(state, idx)
    velocity = standard_velocity_with_vmax(particle, nbest, state)
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
    return [init_particle(rng, domain) for i in range(size)]


def pso(fitness, iterations, parameters=None):
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
