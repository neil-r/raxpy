# raxpy, Python library to rapidly design and execute experiments
| | |
|---|---|
| Testing | [![CI - Test](https://github.com/neil-r/raxpy/actions/workflows/unit_tests.yml/badge.svg)](https://github.com/neil-r/raxpy/actions/workflows/unit_tests.yml) ![Code Coverage](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fraw.githubusercontent.com%2Fneil-r%2Fraxpy%2Fmain%2Fcoverage.json%3Ftoken%3DGHSAT0AAAAAACUX5ZW2YBA4DDCOU27KJPKSZVKMFCA&query=%24.totals.percent_covered_display&suffix=%25&label=Code%20Coverage&color=Green) |
| Meta | [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/neil-r/raxpy/blob/main/LICENSE)


## Description
raxpy is a Python library that designs and executes experiments on Python annotated functions. Given a Python function provided by the user, raxpy introspects functions' signatures to derive experiment input-spaces. With a function's derived input-space, raxpy utilizes different experiment design algorithms to create a small set of function arguments, i.e., the design points, that attempt to cover the whole input-space. With the experiment design, raxpy maps the design points to the function's arguments to execute the function with each point.  

To address limitations in factorial and random point selection algorithms, raxpy provide space-filling design algorithms to generate insightful results from a small number of function executions. For more information, see .

## Usage

```python


```

See examples folder for more usage examples.

## Features

raxpy can execute experiments on functions with the following types of parameters:
- float types
- int types
- str (categorical) types
- Optional, None types  
- Hierarchical types based on dataclasses
- Union types

### Experiment Design Algorithm Support

raxpy provides extended versions of the following algorithms to support optional, hierarchical, and union typed inputs. The space-filling designs work best for exploration use cases when function executions are highly constrained by time and compute resources. Random designs work best when the function needs executed to support the creation of a very large dataset.

 - Space-filling MaxPro
 - Space-filling Uniform (using scipy)
 - Random

## Installation

raxpy requires numpy and scipy.  To install with pip, execute

```
pip install raxpy
```

To execute distributed experiments with MPI, ensure you have the appropriate MPI cluster and also install mpi4py. 

## Support

For community support, please use GitHub issues. 

## Roadmap


### Version 1.0

- Refine and test configspace adapter with hyper-parameter optimization algorithms

### Version x.x

The following elements are being considered for development but not scheduled. 

- Auto-generated data schema and databases
- Advanced trial meta-data features (point ids, run-time, status, etc.)
- Adaptive experimentation algorithms
  - Response surface methodology
  - Sequential design algorithms
 - Support of more input-space constraint types
  - Mixture constraints
  - Multi-dimensional linear constraints
- Surrogate optimization features
- Trial artifact management

## Contributing
This project is open for new contributions. Contributions should follow the coding style as evident in codebase and be unit-tested. New dependencies should mostly be avoided; one exception is the creation of a new adapter, such as creating an adapter to use raxpy with an optimization library.

## Citing

If you used raxpy to support your academic research, please cite:

```

```

## Project status

raxpy is being actively developed as of 2025-01-01.
