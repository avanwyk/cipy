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

""" Unit tests for the cipy.algorithms.pso.base module.
"""
import pytest

from cipy.algorithms.pso import base


@pytest.mark.parametrize("swarm_size", [
    0, 1, 15, 100
])
def test_parameter_initialization(swarm_size):
    params = base.__init_parameters__({"swarm_size": swarm_size})

    assert params["swarm_size"] == swarm_size

    params = base.__init_parameters__({})

    assert params["swarm_size"] is not None