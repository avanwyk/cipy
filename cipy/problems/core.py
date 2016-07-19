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

""" Module defining core utility types and functions used for optimization
problems.
"""
from collections import namedtuple
from functools import singledispatch


Domain = namedtuple('Domain', ['lower', 'upper', 'dimension'])
Minimum = namedtuple('Minimum', ['val'])
Maximum = namedtuple('Maximum', ['val'])


def minimize(fitness_function):
    def objective_function(x):
        return Minimum(fitness_function(x))
    return objective_function


def maximize(fitness_function):
    def objective_function(x):
        return Maximum(fitness_function(x))
    return objective_function


@singledispatch
def compare(l, r):
    return l.val <= r.val


@compare.register(Maximum)
def __compare_maximum__(l, r):
    return l.val >= r.val
