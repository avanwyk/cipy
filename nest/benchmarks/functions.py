import numpy as np


def sphere(x):
    return np.sum(x * x)


def rastrigin(x):
    A = np.int32(10)
    return A * len(x) + np.sum(x * x - A * np.cos(2 * np.pi * x))
