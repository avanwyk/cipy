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
import pytest
import numpy as np

import cipy.algorithms.pso as pso

from cipy.algorithms.core import max_iterations
from cipy.benchmarks import functions
from cipy.problems.core import Domain, Minimum, minimize
from cipy.problems.function import FunctionOptimization


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
    swarm = [mk_particle(best_fitness=Minimum(rng.rand()))
             for i in range(swarm_size)]

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
    swarm = [mk_particle(best_fitness=Minimum(rng.rand()),
                         best_position=rng.uniform(-5.12, 5.12, dimension))
             for i in range(swarm_size)]
    state = pso.State(swarm=swarm, rng=rng, params={}, iterations=0)

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


@pytest.mark.parametrize("dimension", [
    1,
    30
])
def test_execution(dimension):
    """ Smoke test for PSO algorithm testing complete execution of algorithm.
    """
    domain = Domain(-5.12, 5.12, dimension)
    fitness = minimize(functions.sphere)
    optimization_problem = FunctionOptimization(fitness=fitness,
                                                domain=domain)
    result = pso.pso(problem=optimization_problem,
                     stopping_condition=max_iterations(2),
                     parameters={'seed': 3758117674})

    assert result.fitness != np.nan
    assert result.position.size == dimension


@pytest.mark.parametrize("swarm_size", [
    0,
    1,
    15,
    100
])
def test_parameter_initialization(swarm_size):
    params = pso.__init_parameters__({"swarm_size": swarm_size})
    assert params["swarm_size"] == swarm_size

    params = pso.__init_parameters__({})
    assert params["swarm_size"] is not None


def mk_particle(position=None, velocity=None, fitness=None,
                best_fitness=None, best_position=None):
    return pso.Particle(position, velocity, fitness, best_fitness,
                        best_position)
