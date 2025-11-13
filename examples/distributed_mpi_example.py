"""
Script designed to be called from an MPI executable, such as mpirun, to 
demonstrate the execution of a raxpy experiment with MPI. 

> mpirun -n 4 python distributed_mpi_example.py
"""

from typing import Annotated, Optional

import raxpy
from raxpy.runners.mpi import perform_distributed_experiment


def f(
    x1: Annotated[float, raxpy.Float(lb=3.0, ub=4.0)],
    x2: Annotated[Optional[float], raxpy.Float(lb=0.0, ub=3.0)] = 1.5,
) -> float:
    # the following code should execute the computations with these values, such as running a simulation or training a machine learning model
    # to keep it simple for this demonstration, we simply compute a polynominal.

    # In the function specification above, x2 is annotated as Optional. This indicates
    # that this parameter is optional (users can call this function with setting x2 to None)
    # The function specification also provides a lower and upper bound for each float input parameter.

    y = 0.4 * x1
    if x2 is not None:
        y += (x2 * 3.0) + (0.7 * x2 * x1)
    return y


def process_results(design, results):
    """ """
    print(design)
    print(results)


perform_distributed_experiment(f, n_points=100, post_process=process_results)
