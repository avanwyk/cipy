def pytest_addoption(parser):
    parser.addoption("--runexamples", action="store_true",
                     help="run examples")
    parser.addoption("--runbenchmarks", action="store_true",
                     help="run benchmarks")
    parser.addoption("--all", action="store_true",
                     help="run benchmarks")