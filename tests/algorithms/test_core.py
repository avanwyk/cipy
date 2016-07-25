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

""" Unit tests for the cipy.algorithms.core module.
"""
import numpy as np
import pytest

import cipy.algorithms.core as core
import cipy.algorithms.pso.types as types


@pytest.mark.parametrize("iterations", [
    0,
    1,
    10,
    15
])
@pytest.mark.parametrize("max_iterations", [
    0,
    1,
    10
])
def test_maximum_iterations(iterations, max_iterations):
    stopping_condition = core.max_iterations(max_iterations)

    state = types.PSOState(rng=None, params=None, swarm=None, iterations=iterations)
    assert stopping_condition(state) == (iterations >= max_iterations)


@pytest.mark.parametrize("max_iterations", [
    0,
    1,
    10
])
def test_dictionary_based_collector(max_iterations):
    measurements = [lambda state: ('fitness', 1.0),
                    lambda state: ('position', np.array([1,2,3]))]

    (results, collect) = core.dictionary_based_measurements(measurements)
    for iteration in range(max_iterations):
        state = core.State(rng=None, params={}, iterations=iteration)
        results = collect(results, state)
    assert len(results.keys()) == max_iterations
    for key in results.keys():
        assert len(results[key].keys()) == len(measurements)