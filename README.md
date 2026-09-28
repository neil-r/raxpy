# raxpy, Python library to rapidly design and execute experiments

|---|---|
| Testing | [![CI - Test](https://github.com/neil-r/raxpy/actions/workflows/unit_tests.yml/badge.svg)](https://github.com/neil-r/raxpy/actions/workflows/unit_tests.yml) |
| Meta | [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/neil-r/raxpy/blob/main/LICENSE)

## Description
raxpy is a Python library that designs and executes experiments on Python annotated functions. Given a Python function provided by the user, raxpy introspects the function signature to derive an experiment input-space. With a function's derived input-space, raxpy utilizes different experiment design algorithms to create a small set of function arguments, i.e., the design points, that attempt to cover the whole input-space. With the experiment design, raxpy maps the design points to the function's arguments to execute the function with each point.  

To address limitations in factorial and random point selection algorithms, raxpy provide space-filling design algorithms to generate insightful results from a small number of function executions. For more information about the experiment design algorithms, see [https://journals.sagepub.com/doi/10.1177/15485129261481845](https://journals.sagepub.com/doi/10.1177/15485129261481845).

## Usage

 1. Install raxpy if not already installed.
 2. Import raxpy, typing, and dataclass
 3. Create a annotated function that is to be the subject of experimentation 

```python
from typing import Optional, Union, Annotated
from dataclasses import dataclass

import raxpy


# To prepare for the experiment, each branch of a branching factor, 
# you create a dataclass to specify the nested factors

@dataclass
class HierarchicalFactorThreeA:
    x4: Annotated[int, raxpy.Float(lb=0, ub=5)] # nested factor
    # shared nested factor between HierarchicalFactorThreeA and HierarchicalFactorThreeB
    x5: Annotated[float, raxpy.Float(id="x5", lb=1.0, ub=2.0)] 

@dataclass
class HierarchicalFactorThreeB:
    x4: Annotated[float, raxpy.Float(lb=0.0, ub=1.0)]  # nested factor
    # shared nested factor between HierarchicalFactorThreeA and HierarchicalFactorThreeB
    x5: Annotated[float, raxpy.Float(id="x5", lb=1.0, ub=2.0)] 


def f(
    # The following specify the experiment inputs. Annotations
    # provide details such as the lower and upper bound of the inputs
    x1: Annotated[float, raxpy.Float(lb=0.0, ub=1.0)],
    x2: Annotated[Optional[float], raxpy.Float(lb=-1.0, ub=1.0, portion_null=0.4)],
    # x3 is a branching factor, either "branch" HierarchicalFactorThreeA or "branch" HierarchicalFactorThreeB
    x3: Union[HierarchicalFactorThreeA, HierarchicalFactorThreeB], 
):
    # The following code should execute the computations with these values,
    # such as running a simulation or training a machine learning model.
    # to keep it simple for this demonstration, we simply compute a polynominal.

    # In the function specification above, x2 is annotated as Optional. This 
    # indicates that this parameter is optional (users can call this function
    # with setting x2 to None)
    # The function specification also provides a lower and upper bound for
    # each float input parameter.

    # placeholder f logic
    return x1 + (x2 if x2 is not None else 0) * x3.x4 * x3.x5
```
 4. Run experiment 
 
```python
experiment_design, inputs, outputs = raxpy.perform_experiment(f, n_points=20)
```

 5. As needed, assess design of experiment


```python

raxpy.measure.assess_design(experiment_design)

```

See examples folder for more usage examples.

## Features

raxpy can execute experiments on functions with the following types of parameters:
- float types
- int types
- str (categorical) types
- Optional, None types  
- Nested and Nested Branches (Hierarchical) types based on dataclasses' attributes
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

To execute distributed experiments with MPI, also ensure you have the appropriate MPI cluster and install mpi4py. 

## Support

For community support, please use GitHub issues. 

## Roadmap

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

## Project status

raxpy is being actively developed as of 2026-09-25.
