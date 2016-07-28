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

""" Collection of optimization benchmark functions.
"""
import numpy as np


def ackley(x):
    A = -0.2 * np.sqrt(0.5 * (x[0] ** 2 + x[1] ** 2))
    B = 0.5 * (np.cos(2. * np.pi * x[0]) + np.cos(2. * np.pi * x[1]))
    return -20. * np.exp(A) - np.exp(B) + 20. + np.e


def h1(x):
    A = (np.sin(x[0] - x[1] / 8)) ** 2 + (np.sin(x[1] + x[0] / 8)) ** 2
    B = ((x[0] - 8.6998) ** 2 + (x[1] - 6.7665) ** 2) ** 0.5 + 1
    return A / B


def rastrigin(x):
    a = np.int32(10)
    return a * len(x) + np.sum(x * x - a * np.cos(2 * np.pi * x))


def rosenbrock(x):
    return sum(100.0 * (x[1:] - x[:-1] ** 2.0) ** 2.0 + (1 - x[:-1]) ** 2.0)


def sphere(x):
    return np.sum(x * x)
