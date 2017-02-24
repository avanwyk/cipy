# CIPy

[![Build Status](https://travis-ci.org/avanwyk/cipy.svg?branch=master)](https://travis-ci.org/avanwyk/cipy)
[![license](https://img.shields.io/badge/license-Apache%202-blue.svg)](https://github.com/avanwyk/cipy/blob/master/LICENSE)

**C**omputational **I**ntelligence algorithms in **Py**thon using NumPy.

**Notice**: This library is in a pre-alpha stage and all code is subject to
major changes or removal. Additionally, although correctness of the
code is a priority, exhaustive testing of the code is an ongoing
endeavor.

### Installation

* Source (Anaconda environment): **recommended for development.**

    ```shell
    git clone https://github.com/avanwyk/cipy
    cd cipy
    conda env create -f environment.yml
    source activate cipy-env
    pip install -e .
    ```

* Source (development)

    ```shell
    git clone https://github.com/avanwyk/cipy
    cd cipy
    pip install -e .
    ```
    
* Source (setup.py)

    ```shell
    git clone https://github.com/avanwyk/cipy
    cd cipy
    python setup.py install
    ```
    
* PyPI: **recommended for use.**

    ```shell
    pip install cipy
    ``` 

### Examples
Examples of algorithms are given in the examples/ directory:

```shell
python examples/gbest_pso.py
```

### Tests
Unit tests may be run with pytest:

```shell
python -m pytest
```

or setup.py:

```shell
python setup.py test
```

Examples and project benchmarks may be run with pytest as follows:

```shell
python -m pytest --runexamples
```

and

```shell
python -m pytest --runbenchmarks
```

### Issues
This project uses the [Github issue tracker](https://github.com/avanwyk/cipy/issues). Please report any issues there.

### License
The project uses
[Apache License 2.0](http://www.apache.org/licenses/LICENSE-2.0)