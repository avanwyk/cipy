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

""" Unit tests for the cipy.problems.core module.
"""
import pytest
import numpy as np

from cipy.benchmarks.functions import sphere

import cipy.problems.core as core


def __unit__(x):
    return x


@pytest.mark.parametrize("fitness_function", [
    sphere,
    __unit__
])
@pytest.mark.parametrize("solution", [
    np.array([]),
    np.array([1,2,3,4,5])
])
def test_minimize(fitness_function, solution):
    minimize = core.minimize(fitness_function)
    assert isinstance(minimize(solution), core.Minimum)


@pytest.mark.parametrize("fitness_function", [
    sphere,
    __unit__
])
@pytest.mark.parametrize("solution", [
    np.array([]),
    np.array([1,2,3,4,5])
])
def test_maximize(fitness_function, solution):
    maximize = core.maximize(fitness_function)
    assert isinstance(maximize(solution), core.Maximum)
