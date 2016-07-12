import pytest
import numpy as np

import cipy.algorithms.pso as pso


@pytest.fixture
def rng():
    seed = 1028609616
    return np.random.RandomState(seed)


@pytest.mark.parametrize("swarm_size", [
    1,
    25,
    100000,
])
def test_global_best(rng, swarm_size):
    swarm = [mk_particle(best_fitness=rng.rand()) for i in range(swarm_size)]

    desired = sorted(swarm, key=lambda p: p.best_fitness)[0]
    actual = pso.global_best(swarm)

    assert desired == actual


@pytest.mark.parametrize("swarm_size", [
    1,
    25,
    100,
])
@pytest.mark.parametrize("dimension", [
    1,
    30
])
def test_gbest(rng, swarm_size, dimension):
    swarm = [mk_particle(best_fitness=rng.rand(),
                         best_position=rng.uniform(-5.12, 5.12, dimension))
             for i in range(swarm_size)]
    state = pso.State(swarm=swarm, rng=rng, params={})

    desired = sorted(state.swarm, key=lambda p: p.best_fitness)[0].best_position
    actual = pso.gbest(state, 0)

    np.testing.assert_allclose(actual, desired)


@pytest.mark.parametrize("dimension", [
    1,
    30,
    100000,
])
def test_standard_position(rng, dimension):
    position = rng.uniform(-5.12, 5.12, dimension)
    velocity = rng.uniform(-1.0, 1.0, dimension)

    # Naive standard position calculation
    desired = np.zeros(dimension)
    for i in range(dimension):
        desired[i] = position[i] + velocity[i]

    actual = pso.std_position(position, velocity)

    np.testing.assert_allclose(actual, desired)


def mk_particle(position=None, velocity=None, fitness=None,
                best_fitness=None, best_position=None):
    return pso.Particle(position, velocity, fitness, best_fitness,
                        best_position)