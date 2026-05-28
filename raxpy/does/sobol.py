"""
This file contains code for generating Sobol sequences, which 
are a type of low-discrepancy sequence used in quasi-random 
sampling. Sobol sequences are often used in numerical integration
and optimization problems to achieve better convergence rates
than purely random sampling.
"""
from typing import List, Optional
import numpy as np
from scipy.stats.qmc import Sobol

from ..spaces import root as s
from .doe import DesignOfExperiment
from .full_sub_spaces import SubSpaceTargetAllocations
from . import lhs



def create_sobol_points_f(rng: np.random.Generator):
    """
    Creates and returns a function that can create an random matrix using rng


    Arguments
    ---------
    rng: Optional[np.random.Generator]
        random number generator used to pick values
    Returns
    -------
    a function that generates the random matrix

    """

    def create_sobol_points(n_dim_count: int, n_points: int):
        """
        Creates a matrix of random values ranging from 0 to 1

        Arguments
        ---------
        n_dim_count : int
            the number of columns in the matrix
        n_points : int
            the number of rows in the matrix

        Returns
        -------
        np.array
            the created matrix of random points
        """
        sobol_gen = Sobol(
            d=n_dim_count,
            scramble=True,
            seed=rng.integers(low=0, high=10000000)
        )

        data_points = sobol_gen.random(n_points)
        return data_points

    return create_sobol_points


def generate_design(
    space: s.InputSpace,
    n_points: int,
    rng: Optional[np.random.Generator] = None,
) -> DesignOfExperiment:
    """
    Designs an experiment using a Sobol sequence.

    Arguments
    ---------
    space : s.InputSpace
        the input space
    n_points : int
        the number of points to generate
    rng: Optional[np.random.Generator]
        random number generator used to pick values
    Returns
    -------
    DesignOfExperiment
        the designed experiment

    """
    if rng is None:
        rng = np.random.default_rng()
    return lhs.generate_design_with_projection(
        space, n_points, base_creator=create_sobol_points_f(rng)
    )