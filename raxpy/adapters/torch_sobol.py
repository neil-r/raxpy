"""
Provides an adapter to use torch to create a sobol design of experiment
"""

from typing import Optional, List

import numpy as np
import torch.quasirandom as qr

from raxpy.does.doe import DesignOfExperiment
from raxpy.does.full_sub_spaces import SubSpaceTargetAllocations
from raxpy.does import lhs
from raxpy.spaces.root import InputSpace


def create_points_f(rng: np.random.Generator, scramble: bool = True):
    """
    Creates and returns a function that can create points from a sobol sequence
    using rng


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
        engine = qr.SobolEngine(
            n_dim_count,
            scramble=scramble,
            seed=int(rng.integers(1, 10000000000)),
        )

        return engine.draw(n_points).numpy()

    return create_sobol_points


def design_experiment(
    space: InputSpace,
    n_points: int,
    scramble: bool = True,
    rng: Optional[np.random.Generator] = None,
) -> DesignOfExperiment:
    """
    Generates a design of experiment using the Sobol technqiue.
    Ignores heirarhical dimensions initially.

    Arguments
    ---------
    space: InputSpace
    n_points: int
    scramble:bool=True
    rng=None

    Returns
    -------
        a design of expriment

    """

    if rng is None:
        rng = np.random.default_rng()

    return lhs.generate_design_with_projection(
        space=space,
        n_points=n_points,
        base_creator=create_points_f(rng, scramble=scramble),
    )


def generate_seperate_designs_by_full_subspace(
    space: InputSpace,
    n_points: int,
    ensure_at_least_one=False,
    sub_space_target_allocations: Optional[
        List[SubSpaceTargetAllocations]
    ] = None,
    rng: Optional[np.random.Generator] = None,
) -> DesignOfExperiment:
    """
    Generates seperates sobol designs for each concept subspace.

    Arguments
    ---------
    space : s.InputSpace
        the input space
    n_points : int
        the number of points to generate
    ensure_at_least_one=False
        flag to indicate whether to ensure at least one point
        gets allocated to each full-sub-space
    sub_space_target_allocations: Optional[
        List[SubSpaceTargetAllocations]
    ]
        the subspace target allocations, if None, then
        the target allocations are derived from the
        dimensions and the full-sub-spaces
    rng: Optional[np.random.Generator]
        random number generator used to pick values

    Returns
    -------
    DesignOfExperiment
        the designed experiment
    """
    if rng is None:
        rng = np.random.default_rng()

    return lhs.generate_seperate_designs_by_full_subspace(
        space,
        n_points,
        base_creator=create_points_f(rng),
        ensure_at_least_one=ensure_at_least_one,
        sub_space_target_allocations=sub_space_target_allocations,
    )
