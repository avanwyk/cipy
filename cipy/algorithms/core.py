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

Objective = namedtuple('Objective', ['val'])
Minimum = namedtuple('Minimum', Objective._fields)
Maximum = namedtuple('Maximum', Objective._fields)


def max_iterations(maximum):
    """
    Higher order function creating predicate for maximum iterations based
    stopping condition.

    Args:
        maximum (int): The maximum number of iterations.

    Returns:
        callable: Function accepting the current state, testing weather
            the maximum iterations have been reached.

    Examples:
        >>> state = State(iterations=0)
        >>> stopping_condition = max_iterations(1000)
        >>> stopping_condition(state) # False

    """

    def max_iterations_(state):
        return state.iterations >= maximum

    return max_iterations_


def dictionary_based_metrics(metrics):
    """
    Higher order function creating a result type and collection function from
    the given metrics.

    Args:
        metrics (iterable): Sequence of callable metrics, each
            accepting the algorithm state as parameter and returning the
            measured value with its label:
            measurement(state) -> (label, value).

    Returns:
        tuple(dict, callable): dictionary result type and a collection
            function accepting the current results and the state as arguments
            and returning updated result.

    Examples:
        >>> state = State()
        >>> metrics = [lambda state_: state_.iterations]
        >>> (results, collect) = dictionary_based_metrics(metrics)
        >>> results = collect(results, state)

    """

    def collect(results, state):
        """
        Measurement collection function for dictionary based metrics.

        Args:
            results (dict): Storing results of metrics.
            state (cipy.algorithms.core.State): Current state of the algorithm.

        Returns:
            dict: Updated results containing new metrics.
        """
        for measurement in metrics:
            (label, value) = measurement(state)
            iteration_dict = results.get(state.iterations, {})
            iteration_dict[label] = str(value)
            results[state.iterations] = iteration_dict
        return results

    return {}, collect


def minimize(cost_function):
    """
    Higher order function lifting the cost function into a objective function
    returning a cipy.algorithms.core.Minimum.

    Args:
        cost_function (callable): The cost function to minimize.

    Returns:
        callable: Function accepting a solution and returning a
            cipy.algorithms.core.Minimum wrapping the cost.

    Examples:
        >>> sphere = lambda x: x * x
        >>> objective_function = minimize(sphere)
        >>> objective_function(3) # Minimum(9)
    """

    def objective_function(x):
        return Minimum(cost_function(x))

    return objective_function


def maximize(cost_function):
    """
    Higher order function lifting the cost function into a objective function
    returning a cipy.algorithms.core.Maximum.

    Args:
        cost_function (callable): The cost function to maximize.

    Returns:
        callable: Function accepting a solution and returning a
            cipy.algorithms.core.Maximum wrapping the cost.

    Examples:
        >>> sphere = lambda x: x * x
        >>> objective_function = maximize(sphere)
        >>> objective_function(3) # Maximum(9)
    """

    def objective_function(x):
        return Maximum(cost_function(x))

    return objective_function


def comparator(objective):
    """
    Higher order function creating a compare function for objectives.

    Args:
        objective (cipy.algorithms.core.Objective): The objective to create a
            compare for.

    Returns:
        callable: Function accepting two objectives to compare.

    Examples:
        >>> a = Minimum(0.1)
        >>> b = Minimum(0.2)
        >>> compare = comparator(a)
        >>> comparison = compare(a, b) # False
    """

    if isinstance(objective, Minimum):
        return lambda l, r: l < r
    else:
        return lambda l, r: l > r
