"""
    This module provides support to compute 
    point-allocation portions of a design to sub-spaces.
"""

from typing import List, Optional
from dataclasses import dataclass

from ..spaces.complexity import compute_subspace_portions
from ..spaces import InputSpace


@dataclass
class SubSpaceTargetAllocations:
    """
    Representation of a design's sub-space allocation portion
    """

    active_dim_ids: List[str]
    target_portion: float
    allocated_point_count: Optional[int] = None

    def compute_offset_from_target(self, target_points):
        actual_portion = self.allocated_point_count / target_points
        return self.target_portion - actual_portion


def allocate_points_to_full_sub_spaces(
    space: InputSpace,
    n_points: int,
    ensure_at_least_one: bool = False,
    sub_space_target_allocations: Optional[
        List[SubSpaceTargetAllocations]
    ] = None,
) -> List[SubSpaceTargetAllocations]:
    """
    Allocations n_points to sub-spaces given the computed sub-spaces
    portions, if not provided with sub_space_target_allocations.
    If not provided, target portions are computed from the null-portion
    attributes of the dimensions in each sub-space. For example,
    if x1, x2 are in a subspace and the null_portions are 0.5 for each,
    then this sub-space is allocated 0.25 of the n_points.

    Arguments
    ---------
    space:InputSpace
        the whole design space

    n_points: int
        The number of points to allocate to sub-spaces

    ensure_at_least_one:bool=False
        Ensures each sub-space gets at least one point allocation if
        enough points are available

    sub_space_target_allocations:Optional[ List[SubSpaceTargetAllocations]]
        If provided, mutates the elements of list with point allocations,
        and returns this list

    Returns
    -------
    List[SubSpaceTargetAllocations]

    """

    if sub_space_target_allocations is None:
        sub_space_target_allocations = []
        full_subspace_sets = space.derive_full_subspaces()

        # compute portion of the n_points that each sub-design for each sub-space
        # should address
        portitions = compute_subspace_portions(space, full_subspace_sets)

        for target_portion, dim_ids in zip(portitions, full_subspace_sets):
            sub_space_target_allocations.append(
                SubSpaceTargetAllocations(
                    active_dim_ids=dim_ids,
                    target_portion=target_portion,
                )
            )

    # ensure points are allocated
    points_allocated = 0
    for sub_space_target in sub_space_target_allocations:
        if sub_space_target.allocated_point_count is not None:
            points_allocated += sub_space_target_allocations

    if points_allocated > 0 and points_allocated != n_points:
        raise ValueError(
            "If you manually allocate points to a sub-space,"
            " then you must allocate the same number of "
            "points as n_points"
        )
    elif points_allocated == 0:
        # check if any of the subspaces would create duplicates
        # if any duplicates, we need to distribute these given the portitions

        for sub_space_allocation in sub_space_target_allocations:

            sub_space_allocation.allocated_point_count = round(
                sub_space_allocation.target_portion * n_points
            )

            if (
                sub_space_allocation.allocated_point_count == 0
                and ensure_at_least_one
            ):
                sub_space_allocation.allocated_point_count = 1

            points_allocated += sub_space_allocation.allocated_point_count

        if points_allocated > n_points:
            skip_ones = ensure_at_least_one
            if len(sub_space_target_allocations) > n_points:
                skip_ones = False
            # adjust for overly-allocated points
            while points_allocated > n_points:
                max_target_allocation = -1.0
                min_ssa = None
                for sub_space_allocation in sub_space_target_allocations:
                    if (
                        skip_ones
                        and sub_space_allocation.allocated_point_count == 1
                    ) or sub_space_allocation.allocated_point_count < 1:
                        continue
                    offset = sub_space_allocation.compute_offset_from_target(
                        n_points,
                    )

                    if offset > max_target_allocation:
                        max_target_allocation = offset
                        min_ssa = sub_space_allocation

                min_ssa.allocated_point_count -= 1
                points_allocated -= 1
        else:
            # adjust for under-allocated points
            while points_allocated < n_points:
                min_target_allocation = 1.0
                min_ssa = None
                for sub_space_allocation in sub_space_target_allocations:
                    offset = sub_space_allocation.compute_offset_from_target(
                        n_points,
                    )

                    if offset < min_target_allocation:
                        min_target_allocation = offset
                        min_ssa = sub_space_allocation

                min_ssa.allocated_point_count += 1
                points_allocated += 1

    return sub_space_target_allocations
