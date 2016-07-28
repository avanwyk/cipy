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

""" Module defining core utility types and functions used by algorithms.
"""
from collections import namedtuple

State = namedtuple('State', ['rng', 'params', 'iterations'])
Domain = namedtuple('Domain', ['lower', 'upper', 'dimension'])
Minimum = namedtuple('Minimum', ['val'])
Maximum = namedtuple('Maximum', ['val'])


def max_iterations(maximum):
    def max_iterations_(state):
        return state.iterations >= maximum

    return max_iterations_


def dictionary_based_metrics(metrics):
    def collect(results, state):
        for measurement in metrics:
            (label, value) = measurement(state)
            iteration_dict = results.get(state.iterations, {})
            iteration_dict[label] = str(value)
            results[state.iterations] = iteration_dict
        return results

    return {}, collect


def minimize(fitness_function):
    def objective_function(x):
        return Minimum(fitness_function(x))

    return objective_function


def maximize(fitness_function):
    def objective_function(x):
        return Maximum(fitness_function(x))

    return objective_function


def comparator(p):
    if isinstance(p, Minimum):
        return lambda l, r: l < r
    else:
        return lambda l, r: l > r
