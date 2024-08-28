"""
    This module provide logic to create
    LatinHypercube designs for InputSpaces.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.stats.qmc import LatinHypercube
from scipy.optimize import linear_sum_assignment

from ..spaces.dimensions import Dimension, Variant, Composite
from ..spaces.root import (
    InputSpace,
    create_level_iterable,
    create_all_iterable,
)
from ..spaces.complexity import compute_subspace_portions
from .doe import DesignOfExperiment, EncodingEnum
from ..spaces.complexity import estimate_complexity


def create_base_lhs_creator(
    scamble=False, strength=1, optimation: str = "random-cd"
):
    """
    TODO Explain the Function

    Arguments
    ---------
    scramble=True
        **Explanation**
    strength=1
        **Explanation**
    optimation : str
        random-cd **Explanation**

    Returns
    -------
    create : Function
        **Explanation**

    """

    def create(n_dim_count: int, n_points: int):
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
        sampler = LatinHypercube(
            d=n_dim_count,
            strength=strength,
            scramble=scamble,
            optimization=optimation,
        )

        data_points = sampler.random(n=n_points)

        return data_points

    return create


_default_base_lhs_creator = create_base_lhs_creator()

_default_base_lhs_creator_with_scramble = create_base_lhs_creator(scamble=True)


def _compute_cost_matrix(array1: np.array, array2: np.array):
    """
    Compute the cost matrix (distance matrix) between two sets of points.
    """
    cost_matrix = np.linalg.norm(
        array1[:, np.newaxis] - array2[np.newaxis, :], axis=2
    )
    return cost_matrix


def _match_points_hungarian(array1: np.array, array2: np.array):
    """
    Match points in two arrays using the Hungarian algorithm to minimize total distance.
    """
    cost_matrix = _compute_cost_matrix(array1, array2)
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    return row_ind, col_ind


def _level_iterator(space):
    # add level 1 dimensions from root space
    design_request_stack: List[Tuple[Optional[Dimension], List[Dimension]]] = [
        (None, list(create_level_iterable(space.children)))
    ]

    while len(design_request_stack) > 0:
        design_request = design_request_stack.pop(0)

        yield design_request

        active_dims = design_request[1]

        for dim in active_dims:
            if dim.has_child_dimensions():
                if isinstance(dim, Variant):
                    design_request_stack.append((dim, dim.options))
                else:
                    design_request_stack.append(
                        (
                            dim,
                            list(create_level_iterable(dim.children)),
                        )
                    )


class WorkingDesignOfExpertiment:
    """TODO implement"""

    def __init__(self, n_points, dim_count):
        self.total_dim_count = dim_count

        self.final_data_points = np.full(
            (n_points, self.total_dim_count), np.nan
        )

        self.active_index = 0
        self.column_map: Dict[str, Dimension] = {}

        self.input_set_map = {}

    def inject(self, new_dims, data_points, parent_mask):
        for i, dim in enumerate(new_dims):
            self.column_map[self.active_index] = dim
            self.input_set_map[dim.local_id] = self.active_index
            self.final_data_points[parent_mask, self.active_index] = (
                data_points[:, i]
            )

            self.active_index += 1


def _init_merge_with_shadown_design(working_design, base_creator):

    s = working_design.final_data_points.shape
    shadow_design = base_creator(s[1], s[0])

    def init_strategy(data_points):
        return shadow_design[:, 0 : data_points.shape[1]]

    def merge_strategy(data_points, parent_mask):

        start_column = working_design.active_index
        end_column = start_column + data_points.shape[1]

        shadow_design_overlap = shadow_design[
            parent_mask, start_column:end_column
        ]

        index_map = _match_points_hungarian(data_points, shadow_design_overlap)

        return data_points[index_map[1], :]

    return init_strategy, merge_strategy


def _init_merge_simple(working_design, base_creator):

    def init_strategy(data_points):
        return data_points

    def merge_strategy(data_points, parent_mask):
        return data_points

    return init_strategy, merge_strategy


MERGE_SHADOW_DESIGN = "shadow"
MERGE_SIMPLE = "simple"


_merge_method_map = {
    MERGE_SHADOW_DESIGN: _init_merge_with_shadown_design,
    MERGE_SIMPLE: _init_merge_simple,
}


def generate_design_merge_simple(
    space: InputSpace,
    n_points: int,
    base_creator=_default_base_lhs_creator,
) -> DesignOfExperiment:
    return generate_design(
        space, n_points, base_creator, merge_method=MERGE_SIMPLE
    )


def generate_design(
    space: InputSpace,
    n_points: int,
    base_creator=_default_base_lhs_creator,
    merge_method: str = MERGE_SHADOW_DESIGN,
) -> DesignOfExperiment:
    """
    Generates a space-filling design of experiment initially for the space's
    root level. Once it determines which points need children dimensions
    defined, experiments design for the children dimennsion collections are
    computed and merged with the working-parent design. Its repeact this to
    the deepest child dimensions.

    The working-parent designs dictate which points need children values.
    To merge the child designs to the parent designs, the distances between
    the working design and the child design are computed.  The child points
    are mapped in order from the largest distance to the working-parent
    designs smallest distances.

    Arguments
    ---------
    space : InputSpace
        **Explanation**
    n_points : int
        **Explanation**
    base_creator=_default_base_lhs_creator
        **Explanation**
    merge_method: Optional[str]

    Returns
    -------
    DesignOfExperiment :
        a collection of n_points input points

    """
    working_design = WorkingDesignOfExpertiment(
        n_points, space.count_dimensions()
    )

    init_strategy, merge_strategy = _merge_method_map[merge_method](
        working_design, base_creator
    )

    parent_dim = None

    level_iterator = _level_iterator(space)
    for design_request in level_iterator:

        base_level = design_request[0] is None

        if not base_level:
            # Count the number of non-null data-points for parent dimension
            parent_dim = design_request[0]
            parent_inputs = working_design.final_data_points[
                :, working_design.input_set_map[parent_dim.local_id]
            ]
            parent_mask = parent_inputs > parent_dim.portion_null
            points_to_create = np.sum(parent_mask)
        else:
            parent_mask = None
            points_to_create = n_points

        # addressed fixed dimensions
        active_dims = design_request[1]

        if parent_dim is None or not isinstance(parent_dim, Variant):
            parts = [(active_dims, parent_mask, points_to_create)]
        else:
            parts = []
            parent_values = parent_dim.collapse_uniform(parent_inputs)
            for option_index, option_dim in enumerate(active_dims):
                parent_mask = [option_index == pv for pv in parent_values]
                parts.append(
                    (
                        list(create_level_iterable([option_dim])),
                        parent_mask,
                        sum(parent_mask),
                    )
                )

        for active_dims, parent_mask, points_to_create in parts:
            data_points = base_creator(len(active_dims), points_to_create)
            if base_level:
                data_points = init_strategy(data_points)
                parent_mask = [True for _ in range(n_points)]
            else:
                data_points = merge_strategy(data_points, parent_mask)

            working_design.inject(active_dims, data_points, parent_mask)

    # decoded_values = space.decode_zero_one_matrix(
    #    final_data_points, input_set_map
    # )

    return DesignOfExperiment(
        input_space=space,
        input_sets=working_design.final_data_points,
        input_set_map=working_design.input_set_map,
        encoding=EncodingEnum.ZERO_ONE_RAW_ENCODING,
    )


def generate_design_by_level_opt_merge(
    space: InputSpace,
    n_points: int,
    base_creator=_default_base_lhs_creator,
) -> DesignOfExperiment:
    """
    Generates a space-filling design of experiment initially for the space's
    root level. Once it determines which points need children dimensions
    defined, experiments design for the children dimennsion collections are
    computed and merged with the working-parent design. Its repeact this to
    the deepest child dimensions.

    The working-parent designs dictate which points need children values.
    To merge the child designs to the parent designs, the distances between
    the working design and the child design are computed.  The child points
    are mapped in order from the largest distance to the working-parent
    designs smallest distances.

    Arguments
    ---------
    space : InputSpace
        **Explanation**
    n_points : int
        **Explanation**
    base_creator=_default_base_lhs_creator
        **Explanation**

    Returns
    -------
    DesignOfExperiment :
        a collection of n_points input points

    """

    total_dim_count = space.count_dimensions

    final_data_points = np.full((n_points, total_dim_count), np.nan)

    active_index = 0
    column_map: Dict[str, Dimension] = {}
    input_set_map = {}

    parent_dim = None

    level_iterator = _level_iterator(space)
    historical_row_column_masks = []
    for design_request in level_iterator:

        base_level = design_request[0] is None

        if not base_level:
            # Count the number of non-null data-points for parent dimension
            parent_dim = design_request[0]
            parent_inputs = final_data_points[
                :, input_set_map[parent_dim.local_id]
            ]
            parent_mask = parent_inputs > parent_dim.portion_null
            points_to_create = np.sum(parent_mask)
        else:
            parent_mask = [True for _ in range(n_points)]
            points_to_create = n_points

        # addressed fixed dimensions
        active_dims = design_request[1]

        if parent_dim is None or not isinstance(parent_dim, Variant):
            parts = [(active_dims, parent_mask, points_to_create)]
        else:
            parts = []
            parent_values = parent_dim.collapse_uniform(parent_inputs)
            for option_index, option_dim in enumerate(active_dims):
                parent_mask = [option_index == pv for pv in parent_values]
                parts.append(
                    (
                        list(create_level_iterable([option_dim])),
                        parent_mask,
                        sum(parent_mask),
                    )
                )

        for active_dims, parent_mask, points_to_create in parts:
            data_points = base_creator(len(active_dims), points_to_create)

            column_mask = []

            for i, dim in enumerate(active_dims):
                column_map[active_index] = dim
                input_set_map[dim.local_id] = active_index
                final_data_points[parent_mask, active_index] = data_points[
                    :, i
                ]
                column_mask.append(active_index)

                active_index += 1

            # now optimize discrepancy given historical level designs
            from .scipy_optimizations import random_cd

            for h_row_mask, h_column_mask in historical_row_column_masks:
                combination_row_mask = list(
                    m1 and m2 for m1, m2 in zip(h_row_mask, parent_mask)
                )

                # ensure design works
                design_c = final_data_points[
                    combination_row_mask, h_column_mask + column_mask
                ]

                design_c = random_cd(
                    design_c,
                    n_iters=100000,
                    n_nochange=100,
                    column_bounds=[
                        len(h_column_mask),
                        len(h_column_mask + column_mask) - 1,
                    ],
                )

                final_data_points[combination_row_mask, column_mask] = (
                    design_c[
                        :,
                        len(h_column_mask)
                        - 1 : len(h_column_mask + column_mask)
                        - 1,
                    ]
                )

            historical_row_column_masks.append((parent_mask, column_mask))

    # decoded_values = space.decode_zero_one_matrix(
    #    final_data_points, input_set_map
    # )

    return DesignOfExperiment(
        input_space=space,
        input_sets=final_data_points,
        input_set_map=input_set_map,
        encoding=EncodingEnum.ZERO_ONE_RAW_ENCODING,
    )


@dataclass
class SubSpaceTargetAllocations:
    active_dim_ids: List[str]
    target_portion: float
    allocated_point_count: Optional[int] = None

    def compute_offset_from_target(self, target_points):
        actual_portion = self.allocated_point_count / target_points
        return self.target_portion - actual_portion


def generate_seperate_designs_by_full_subspace(
    space: InputSpace,
    n_points: int,
    base_creator=_default_base_lhs_creator_with_scramble,
    ensure_at_least_one=False,
    sub_space_target_allocations: Optional[
        List[SubSpaceTargetAllocations]
    ] = None,
) -> DesignOfExperiment:
    """
    TODO Explain the Function

    Arguments
    ---------
    space : InputSpace
        **Explanation**
    n_points : int
        **Explanation**
    base_creator=_default_base_lhs_creator
        **Explanation**
    ensure_at_least_one=True
        **Explanation**
    sub_space_target_allocations =None

    Returns
    -------
    DesignOfExperiment :
        **Explanation**

    """
    total_dim_count = space.count_dimensions()

    final_data_points = np.full((n_points, total_dim_count), np.nan)
    active_index = 0
    column_map: Dict[str, Dimension] = {}
    input_set_map = {}

    # if target allocations are not supplied, then generate the defaults
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

    dim_map = space.create_dim_map()

    # check if points are allocated
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
    if points_allocated != n_points:

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

    lb_index = 0

    # create designs for sub-spaces given the allocated points counts
    for i, sub_space_allocation in zip(
        range(len(full_subspace_sets)), sub_space_target_allocations
    ):
        points_to_create = sub_space_allocation.allocated_point_count

        if points_to_create < 1:
            print("Skipping dimensions")
            continue

        # compute the rows that will specified
        a_lb_index = lb_index
        ub_index = lb_index + points_to_create
        lb_index = ub_index
        row_mask = [
            True if (i >= a_lb_index and i < ub_index) else False
            for i in range(n_points)
        ]

        # since the subspace dictakes the values of composite and variant
        # dimensions, exclude them and manually set the values of these
        # dimensions
        fixed_dims = []
        active_dims = []

        for dim_id in sub_space_allocation.active_dim_ids:
            dim = dim_map[dim_id]
            if isinstance(dim, (Variant, Composite)):
                fixed_dims.append(dim)
            else:
                active_dims.append(dim)

        if len(active_dims) > 0:
            data_points = base_creator(len(active_dims), points_to_create)
            part_input_set_map = {}
            for i, dim in enumerate(active_dims):
                part_input_set_map[dim.id] = i

            decoded_data_points = space.decode_zero_one_matrix(
                data_points, part_input_set_map, utilize_null_portitions=False
            )

            for i, dim in enumerate(active_dims):

                if dim.id not in input_set_map:
                    dim_index = active_index
                    active_index += 1

                    column_map[dim_index] = dim
                    input_set_map[dim.id] = dim_index
                else:
                    dim_index = input_set_map[dim.id]

                final_data_points[row_mask, dim_index] = decoded_data_points[
                    :, i
                ]

        if len(fixed_dims) > 0:

            for dim in fixed_dims:
                if dim.id not in input_set_map:
                    dim_index = active_index
                    active_index += 1

                    column_map[dim_index] = dim
                    input_set_map[dim.id] = dim_index
                else:
                    dim_index = input_set_map[dim.id]

                # determine value to inject into final data points
                if isinstance(dim, Composite):
                    v = 1
                else:
                    # if Variant type must determine the child dimension active
                    for i, child_dim in enumerate(dim.children):
                        if child_dim.id in sub_space_allocation.active_dim_ids:
                            v = i
                            break

                final_data_points[row_mask, dim_index] = v

    return DesignOfExperiment(
        input_space=space,
        input_sets=final_data_points,
        input_set_map=input_set_map,
        encoding=EncodingEnum.NONE,
    )


class ValuePool:

    def __init__(self, value_count):
        self._values = list(
            (i / value_count) + (1 / (value_count * 2))
            for i in range(value_count)
        )

    def pull(self, point_count):
        """
        Pull a random element from each quantile in a sorted list.

        Parameters:
        sorted_list (list): A sorted list of elements.
        num_quantiles (int): The number of quantiles to divide the list into.

        Returns:
        list: A list containing a randomly selected element from each quantile.
        """
        if point_count <= 0:
            raise ValueError("Number of points must be greater than 0")

        if len(self._values) < point_count:
            raise ValueError(
                "Number of quantiles is greater than the number of elements in the list"
            )
        elif len(self._values) == point_count:
            values = self._values
            self._values = []
            return values

        quantiles = np.linspace(
            0, len(self._values), point_count + 1, dtype=int
        )
        indices = []
        rng = np.random.default_rng()
        selected_values = []
        for i in range(point_count):
            start_index = quantiles[i]
            end_index = quantiles[i + 1] - 1

            indice = rng.choice(np.arange(start_index, end_index + 1))
            indices.append(indice)

            selected_values.append(self._values[indice])

        for r in reversed(indices):
            del self._values[r]

        return selected_values


def generate_seperate_designs_by_full_subspace_and_pool(
    space: InputSpace,
    n_points: int,
    ensure_at_least_one=False,
    sub_space_target_allocations: Optional[
        List[SubSpaceTargetAllocations]
    ] = None,
) -> DesignOfExperiment:
    """
    TODO Explain the Function

    Arguments
    ---------
    space : InputSpace
        **Explanation**
    n_points : int
        **Explanation**
    ensure_at_least_one=True
        **Explanation**
    sub_space_target_allocations =None

    Returns
    -------
    DesignOfExperiment :
        **Explanation**

    """
    total_dim_count = space.count_dimensions()

    final_data_points = np.full((n_points, total_dim_count), np.nan)
    active_index = 0
    column_map: Dict[str, Dimension] = {}
    input_set_map = {}

    # if target allocations are not supplied, then generate the defaults
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

    dim_map = space.create_dim_map()

    # check if points are allocated
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
    if points_allocated != n_points:

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

    # compute the number of values needed for each dimension
    dim_counts = {}

    for ssta in sub_space_target_allocations:
        for dim_ids in ssta.active_dim_ids:
            for dim_id in dim_ids:
                if dim_id not in dim_counts:
                    dim_counts[dim_id] = 0
                dim_counts[dim_id] += ssta.allocated_point_count

    value_pool = {
        dim_id: ValuePool(dim_count)
        for dim_id, dim_count in dim_counts.items()
    }

    sorted_allocations = sorted(
        sub_space_target_allocations,
        key=lambda ssta: ssta.allocated_point_count,
        reverse=True,
    )

    lb_index = 0

    # create designs for sub-spaces given the allocated points counts
    for i, sub_space_allocation in zip(
        range(len(full_subspace_sets)), sorted_allocations
    ):
        points_to_create = sub_space_allocation.allocated_point_count

        if points_to_create < 1:
            print("Skipping dimensions")
            continue

        # compute the rows that will specified
        a_lb_index = lb_index
        ub_index = lb_index + points_to_create
        lb_index = ub_index
        row_mask = [
            True if (i >= a_lb_index and i < ub_index) else False
            for i in range(n_points)
        ]

        # since the subspace dictakes the values of composite and variant
        # dimensions, exclude them and manually set the values of these
        # dimensions
        fixed_dims = []
        active_dims = []

        for dim_id in sub_space_allocation.active_dim_ids:
            dim = dim_map[dim_id]
            if isinstance(dim, (Variant, Composite)):
                fixed_dims.append(dim)
            else:
                active_dims.append(dim)

        if len(active_dims) > 0:

            data_points = np.array(
                [
                    value_pool[dim_id].pull(points_to_create)
                    for dim_id in active_dims
                ]
            )

            rng = np.random.default_rng()
            for i in range(len(active_dims)):
                rng.shuffle(data_points[i, :])
            data_points = data_points.T

            from .scipy_optimizations import random_cd

            data_points = random_cd(random_cd, 10000, 100)

            part_input_set_map = {}
            for i, dim in enumerate(active_dims):
                part_input_set_map[dim.id] = i

            decoded_data_points = space.decode_zero_one_matrix(
                data_points, part_input_set_map, utilize_null_portitions=False
            )

            for i, dim in enumerate(active_dims):

                if dim.id not in input_set_map:
                    dim_index = active_index
                    active_index += 1

                    column_map[dim_index] = dim
                    input_set_map[dim.id] = dim_index
                else:
                    dim_index = input_set_map[dim.id]

                final_data_points[row_mask, dim_index] = decoded_data_points[
                    :, i
                ]

        if len(fixed_dims) > 0:

            for dim in fixed_dims:
                if dim.id not in input_set_map:
                    dim_index = active_index
                    active_index += 1

                    column_map[dim_index] = dim
                    input_set_map[dim.id] = dim_index
                else:
                    dim_index = input_set_map[dim.id]

                # determine value to inject into final data points
                if isinstance(dim, Composite):
                    v = 1
                else:
                    # if Variant type must determine the child dimension active
                    for i, child_dim in enumerate(dim.children):
                        if child_dim.id in sub_space_allocation.active_dim_ids:
                            v = i
                            break

                final_data_points[row_mask, dim_index] = v

    return DesignOfExperiment(
        input_space=space,
        input_sets=final_data_points,
        input_set_map=input_set_map,
        encoding=EncodingEnum.NONE,
    )


def generate_design_with_projection(
    space: InputSpace, n_points: int, base_creator=_default_base_lhs_creator
) -> DesignOfExperiment:
    """
    TODO Explain the Function

    Arguments
    ---------
    space : InputSpace
        **Explanation**
    n_points : int
        **Explanation**
    base_creator=_default_base_lhs_creator
        **Explanation**

    Returns
    -------
    DesignOfExperiment :
        **Explanation**

    """
    active_dims = list(create_all_iterable(space.children))
    input_set_map = {}
    for i, dim in enumerate(active_dims):
        input_set_map[dim.id] = i

    data_points = base_creator(len(active_dims), n_points)

    return DesignOfExperiment(
        input_space=space,
        input_sets=data_points,
        input_set_map=input_set_map,
        encoding=EncodingEnum.ZERO_ONE_RAW_ENCODING,
    )
