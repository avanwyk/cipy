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

import cipy.algorithms.core as core
import cipy.algorithms.pso as pso


class PSOOptimizer(object):
    def __init__(self, params=pso.default_parameters(),
                 position_update=pso.std_position,
                 velocity_update=pso.std_velocity_with_v_max,
                 parameter_update=pso.std_parameter_update,
                 measurements=(),
                 measurer=core.dictionary_based_metrics):
        self.params = params
        self.position_update = position_update
        self.velocity_update = velocity_update
        self.parameter_update = parameter_update
        self.measurements = measurements
        self.measurer = measurer
        self.solution = None
        self.metrics = {}

    def minimize(self, cost_function, domain_lower, domain_upper, dimension,
                 stopping_condition):
        return self.optimize(core.minimize(cost_function),
                             domain_lower, domain_upper, dimension,
                             stopping_condition)

    def maximize(self, cost_function, domain_lower, domain_upper, dimension,
                 stopping_condition):
        return self.optimize(core.maximize(cost_function), domain_lower,
                             domain_upper, dimension, stopping_condition)

    def optimize(self, objective_function, domain_lower, domain_upper, dimension,
                 stopping_condition):
        (solution, metrics) = pso.optimize(objective_function,
                                           Domain(domain_lower, domain_upper,
                                                  dimension),
                                           stopping_condition,
                                           self.params, self.position_update,
                                           self.velocity_update,
                                           self.parameter_update,
                                           self.measurements, self.measurer)
        self.solution = solution
        self.metrics = metrics
        return solution.best_position

    def accuracy(self):
        return self.solution.best_fitness.val

    def solution(self):
        return None if self.solution is None else self.solution.best_position


Domain = core.Domain
