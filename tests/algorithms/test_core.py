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
import pytest

import cipy.algorithms.pso as pso
import cipy.algorithms.core as core


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

    state = pso.State(rng=None, params=None, swarm=None, iterations=iterations)
    assert stopping_condition(state) == (iterations >= max_iterations)
