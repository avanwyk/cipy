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

from cipy.algorithms.core import max_iterations
from cipy.algorithms.pso import pso
from cipy.benchmarks import functions
from cipy.problems.core import Domain, minimize
from cipy.problems.function import FunctionOptimization

optimization_problem = FunctionOptimization(optimization=minimize,
                                            fitness=functions.sphere,
                                            domain=Domain(-5.12, 5.12, 30))
result = pso(problem=optimization_problem,
             stopping_condition=max_iterations(1000),
             parameters={'seed': 3758117674})

print("Result fitness: %s" % result.fitness)
print("Result: %s" % result.position)