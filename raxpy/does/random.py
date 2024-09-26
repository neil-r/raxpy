""" 
    This modules provides support to create random
    experiment designs.
"""

from typing import List, Optional
import numpy as np

from ..spaces import root as s
from .doe import DesignOfExperiment
from .full_sub_spaces import SubSpaceTargetAllocations
from . import lhs


def create_random_points(n_dim_count: int, n_points: int):
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
    data_points = np.random.rand(n_points, n_dim_count)

    return data_points


def generate_design(space: s.InputSpace, n_points: int) -> DesignOfExperiment:
    """
    Designs an experiment using random number generation.

    Arguments
    ---------
    space : s.InputSpace
        the input space
    n_points : int
        the number of points to generate

    Returns
    -------
    DesignOfExperiment
        the designed experiment

    """
    return lhs.generate_design_with_projection(
        space, n_points, base_creator=create_random_points
    )


def generate_seperate_designs_by_full_subspace(
    space: s.InputSpace,
    n_points: int,
    ensure_at_least_one=False,
    sub_space_target_allocations: Optional[
        List[SubSpaceTargetAllocations]
    ] = None,
) -> DesignOfExperiment:
    """
    Generates a random design given the full-sub-space target
    allocation settings.

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

    Returns
    -------
    DesignOfExperiment
        the designed experiment
    """
    return lhs.generate_seperate_designs_by_full_subspace(
        space,
        n_points,
        base_creator=create_random_points,
        ensure_at_least_one=ensure_at_least_one,
        sub_space_target_allocations=sub_space_target_allocations,
    )
