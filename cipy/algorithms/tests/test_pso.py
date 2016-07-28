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

""" Unit tests for the cipy.algorithms.pso module.
"""
import numpy as np
import pytest

from cipy.algorithms.core import Domain
from cipy.algorithms.core import Minimum
from cipy.algorithms.core import max_iterations
from cipy.algorithms.core import minimize
from cipy.algorithms.pso import base
from cipy.algorithms.pso import functions
from cipy.algorithms.pso import types
from cipy.benchmarks import functions as benchmarks


@pytest.fixture
def rng():
    seed = 1028609616
    return np.random.RandomState(seed)


@pytest.mark.parametrize("swarm_size", [
    1,
    25,
    100000,
])
def test_solution(rng, swarm_size):
    swarm = [mk_particle(best_fitness=Minimum(rng.rand()))
             for i in range(swarm_size)]

    desired = sorted(swarm, key=lambda p: p.best_fitness)[0]
    actual = functions.solution(swarm)

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
    swarm = [mk_particle(best_fitness=Minimum(rng.rand()),
                         best_position=rng.uniform(-5.12, 5.12, dimension))
             for i in range(swarm_size)]

    desired = sorted(enumerate(swarm), key=lambda ip: ip[1].best_fitness)[0][0]
    actual = functions.gbest_idx(swarm)

    assert actual == desired


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

    actual = functions.std_position(position, velocity)

    np.testing.assert_allclose(actual, desired)


@pytest.mark.parametrize("swarm_size", [
    0,
    1,
    15,
    100
])
def test_parameter_initialization(swarm_size):
    params = base.__init_parameters__({"swarm_size": swarm_size})

    assert params["swarm_size"] == swarm_size

    params = base.__init_parameters__({})

    assert params["swarm_size"] is not None


def mk_particle(position=None, velocity=None, fitness=None,
                best_fitness=None, best_position=None):
    return types.Particle(position, velocity, fitness, best_fitness,
                          best_position)
