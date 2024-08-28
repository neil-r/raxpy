""" TODO Explain Module"""

from typing import List, Optional
import numpy as np

from ..spaces import root as s
from .doe import DesignOfExperiment
from . import lhs


def create_points(n_dim_count: int, n_points: int):
    """
    TODO Explain the Function

    Arguments
    ---------
    n_dim_count : int
        **Explanation**
    n_points : int
        **Explanation**

    Returns
    -------
    data_points : np.array
        **Explanation**

    """
    data_points = np.random.rand(n_points, n_dim_count)

    return data_points


def generate_design(space: s.InputSpace, n_points: int) -> DesignOfExperiment:
    """
    TODO Explain the Function

    Arguments
    ---------
    space : s.InputSpace
        **Explanation**
    n_points : int
        **Explanation**

    Returns
    -------
    DesignOfExperiment
        **Explanation**

    """
    return lhs.generate_design_with_projection(
        space, n_points, base_creator=create_points
    )


def generate_seperate_designs_by_full_subspace(
    space: s.InputSpace,
    n_points: int,
    ensure_at_least_one=False,
    sub_space_target_allocations: Optional[
        List[lhs.SubSpaceTargetAllocations]
    ] = None,
) -> DesignOfExperiment:
    """
    TODO Explain the Function

    Arguments
    ---------
    space : s.InputSpace
        **Explanation**
    n_points : int
        **Explanation**

    Returns
    -------
    DesignOfExperiment
        **Explanation**

    """
    return lhs.generate_seperate_designs_by_full_subspace(
        space,
        n_points,
        base_creator=create_points,
        ensure_at_least_one=ensure_at_least_one,
        sub_space_target_allocations=sub_space_target_allocations,
    )
