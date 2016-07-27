# CIPy

[![Build Status](https://travis-ci.org/avanwyk/cipy.svg?branch=master)](https://travis-ci.org/avanwyk/cipy)
[![license](https://img.shields.io/badge/license-Apache%202-blue.svg)](https://github.com/avanwyk/cipy/blob/master/LICENSE)

**C**omputational **I**ntelligence algorithms in **Py**thon using NumPy.

**Notice**: This library is in a pre-alpha stage and all code is subject to
major changes or removal. Additionally, although correctness of the
code is a priority, exhaustive testing of the code is an ongoing
endeavor.

### Installation
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
* PyPI

    ```shell
    pip install cipy
    ``` 

### Examples
Examples of algorithms are given in the examples/ directory:

```shell
python examples/gbest_pso.py
```

### Tests
Tests may be run with pytest:

```shell
python -m pytest
```

or via setup.py

```shell
python setup.py test
```

### License
The project uses
[Apache License 2.0](http://www.apache.org/licenses/LICENSE-2.0)