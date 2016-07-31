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

""" Unit tests for the cipy.algorithms.pso.functions module.
"""
import numpy as np
import pytest

from cipy.algorithms.core import Domain, minimize
from cipy.algorithms.core import Minimum
from cipy.algorithms.pso import functions
from cipy.algorithms.pso import types


@pytest.fixture
def rng():
    seed = 1028609616
    return np.random.RandomState(seed)


@pytest.mark.parametrize("swarm_size", [
    1, 25, 100000
])
def test_solution(rng, swarm_size):
    swarm = [mk_particle(best_fitness=Minimum(rng.rand()))
             for i in range(swarm_size)]

    desired = sorted(swarm, key=lambda p: p.best_fitness)[0]
    actual = functions.solution(swarm)

    assert desired == actual


@pytest.mark.parametrize("swarm_size", [
    1, 25, 100
])
@pytest.mark.parametrize("dimension", [
    1, 30
])
def test_gbest(rng, swarm_size, dimension):
    swarm = [mk_particle(best_fitness=Minimum(rng.rand()),
                         best_position=rng.uniform(-5.12, 5.12, dimension))
             for i in range(swarm_size)]

    desired = sorted(enumerate(swarm), key=lambda ip: ip[1].best_fitness)[0][0]
    actual = functions.gbest_idx(swarm)

    assert actual == desired


@pytest.mark.parametrize("dimension", [
    1, 30, 100000
])
def test_standard_position(rng, dimension):
    position = rng.uniform(-5.12, 5.12, dimension)
    velocity = rng.uniform(-1.0, 1.0, dimension)

    # Naive standard position calculation
    desired = np.zeros(dimension)
    for i in range(dimension):
        desired[i] = position[i] + velocity[i]

    actual = functions.std_position(position, velocity)

    np.testing.assert_array_equal(actual, desired)


@pytest.mark.parametrize("dimension", [
    1, 30, 10000
])
@pytest.mark.parametrize("inertia", [
    0.0, 0.1, 1.0, 2.0
])
@pytest.mark.parametrize("c", [
    0.0, 0.1, 1.0, 2.0
])
def test_standard_velocity_equation(rng, dimension, inertia, c):
    position = rng.uniform(-5.12, 5.12, dimension)
    velocity = rng.uniform(-5.12, 5.12, dimension)
    best_position = rng.uniform(-5.12, 5.12, dimension)
    particle = mk_particle(position, velocity, None, None, best_position)

    c1r1 = functions.__acceleration__(rng, c, dimension)
    c2r2 = functions.__acceleration__(rng, c, dimension)
    social = rng.uniform(-5.12, 5.12, dimension)

    # Naive standard velocity calculation
    desired = np.zeros(dimension)
    for i in range(dimension):
        desired[i] = (inertia * velocity[i] +
                      c1r1[i] * (best_position[i] - position[i]) +
                      c2r2[i] * (social[i] - position[i]))

    actual = functions.__std_velocity_equation__(inertia, c1r1, c2r2,
                                                 particle, social)

    np.testing.assert_array_equal(actual, desired)


@pytest.mark.parametrize("coefficient", [
    0.1, 1.0, 2.0
])
@pytest.mark.parametrize("dimension", [
    1, 30, 100000
])
def test_acceleration(rng, coefficient, dimension):
    acceleration = functions.__acceleration__(rng, coefficient, dimension)

    assert acceleration.size == dimension
    np.testing.assert_array_less(acceleration, np.full(dimension, coefficient))


@pytest.mark.parametrize("dimension", [
    1, 30, 100000
])
@pytest.mark.parametrize("v_max", [
    0.1, 3.0, None
])
def test_clamp(rng, dimension, v_max):
    velocity = rng.uniform(-5.12, 5.12, dimension)

    clamped = functions.__clamp__(velocity, v_max)

    assert clamped.size == dimension
    if v_max is not None:
        assert clamped[np.where(clamped > v_max)].size == 0
        assert clamped[np.where(clamped < -v_max)].size == 0
    else:
        np.testing.assert_array_equal(velocity, clamped)


def mk_particle(position=None, velocity=None, fitness=None,
                best_fitness=None, best_position=None):
    return types.Particle(position, velocity, fitness, best_fitness,
                          best_position)


@pytest.mark.parametrize("dimension", [
    1, 30, 10000
])
@pytest.mark.parametrize("inertia", [
    0.0, 0.1, 1.0, 2.0
])
@pytest.mark.parametrize("rho", [
    0.0, 0.1, 1.0
])
def test_gc_velocity_equation(rng, dimension, inertia, rho):
    position = rng.uniform(-5.12, 5.12, dimension)
    velocity = rng.uniform(-5.12, 5.12, dimension)
    gbest = rng.uniform(-5.12, 5.12, dimension)
    particle = mk_particle(position, velocity, None, None, None)

    r2 = rng.uniform(0.0, 1.0, dimension)

    # Naive GC velocity calculation
    desired = np.zeros(dimension)
    for i in range(dimension):
        desired[i] = (-1 * position[i] + gbest[i] + inertia * velocity[i] +
                      rho * (1 - 2 * r2[i]))

    actual = functions.__gc_velocity_equation__(inertia, rho, r2, particle,
                                                gbest)

    np.testing.assert_array_equal(actual, desired)


@pytest.mark.parametrize("dimension", [
    1, 30, 10000
])
def test_particle_initialization(rng, dimension):
    fitness_function = lambda x: np.sum(x)
    domain = Domain(-5.12, 5.12, dimension)

    particle = functions.initialize_particle(rng, domain, fitness_function)

    position = particle.position
    np.testing.assert_array_less(position, np.full(dimension, 5.12))
    np.testing.assert_array_less(np.full(dimension, -5.12), position)

    assert particle.fitness is not None
    assert particle.best_fitness is not None
    assert particle.velocity.size == dimension
    np.testing.assert_array_equal(particle.best_position, position)


@pytest.mark.parametrize("dimension", [
    1, 30
])
def test_update_best_fitness(rng, dimension):
    objective_function = minimize(lambda x: np.sum(x))
    position = rng.uniform(-5.12, 5.12, dimension)
    fitness = objective_function(position)

    particle = mk_particle(position, None, 2 * fitness, 2 * fitness, None)
    updated = functions.update_fitness(objective_function, particle)

    np.testing.assert_array_equal(updated.position, position)
    np.testing.assert_array_equal(updated.best_position, position)
    assert updated.fitness == fitness
    assert updated.best_fitness == fitness


@pytest.mark.parametrize("dimension", [
    1, 30
])
def test_update_fitness(rng, dimension):
    objective_function = minimize(lambda x: np.abs(np.sum(x)))
    position = rng.uniform(-5.12, 5.12, dimension)
    fitness = objective_function(position)

    particle = mk_particle(position, None, 2 * fitness, Minimum(0.0), None)
    updated = functions.update_fitness(objective_function, particle)

    assert updated.fitness == fitness
    assert updated.best_fitness == Minimum(0.0)