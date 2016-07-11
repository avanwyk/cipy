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