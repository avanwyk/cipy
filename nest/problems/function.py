from collections import namedtuple


def minimize(a, b):
    return a < b


class FunctionOptimization(namedtuple('FunctionBase',
                                      ['type', 'function', 'domain'])):
    def __call__(self, solution):
        return self.function(solution)

    def compare(self, a, b):
        return self.type(a, b)
