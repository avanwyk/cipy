import pytest

from examples.gbest_pso import main as gbest
from examples.lbest_pso import main as lbest
from examples.gc_pso import main as gc
from examples.pso_optimizer import main as pso_optimizer


@pytest.mark.parametrize("dimension", [
    1,
    30
])
@pytest.mark.parametrize("iterations", [
    3
])
def test_gbest_pso(dimension, iterations):
    gbest(dimension, iterations)


@pytest.mark.parametrize("dimension", [
    1,
    30
])
@pytest.mark.parametrize("iterations", [
    3
])
def test_lbest_pso(dimension, iterations):
    lbest(dimension, iterations)


@pytest.mark.parametrize("dimension", [
    1,
    30
])
@pytest.mark.parametrize("iterations", [
    3
])
def test_gc_pso(dimension, iterations):
    gc(dimension, iterations)


@pytest.mark.parametrize("dimension", [
    1,
    30
])
@pytest.mark.parametrize("iterations", [
    3
])
def test_gc_pso(dimension, iterations):
    pso_optimizer(dimension, iterations)
