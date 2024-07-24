"""
    This module provide logic to create LatinHypercube designs for InputSpaces.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.stats.qmc import LatinHypercube

from ..spaces.dimensions import Dimension, Variant, Composite
from ..spaces.root import InputSpace, create_level_iterable, create_all_iterable
from ..spaces.complexity import compute_subspace_portitions
from .doe import DesignOfExperiment
from ..spaces.complexity import estimate_complexity


def create_base_lhs_creator(scamble=True, strength=1, optimation: str = "random-cd"):
    def create(dims: List[Dimension], n_points: int):
        n_dim_count = len(dims)
        sampler = LatinHypercube(
            d=n_dim_count, strength=strength, scramble=scamble, optimization=optimation
        )

        data_points = sampler.random(n=n_points)
        encoded_flag = True

        return (encoded_flag, data_points)

    return create


_default_base_lhs_creator = create_base_lhs_creator()


def generate_design(
    space: InputSpace, n_points: int, base_creator=_default_base_lhs_creator
) -> DesignOfExperiment:

    total_dim_count = space.count_dimensions()

    final_data_points = np.full((n_points, total_dim_count), np.nan)
    active_index = 0
    column_map: Dict[str, Dimension] = {}

    # add level 1 dimensions from root space
    design_request_stack: List[Tuple[Optional[Dimension], List[Dimension]]] = [
        (None, list(create_level_iterable(space.children)))
    ]

    input_set_map = {}
    parent_dim = None
    while len(design_request_stack) > 0:
        design_request = design_request_stack.pop(0)
        points_to_create = n_points
        base_level = design_request[0] is None
        parent_mask = None
        if not base_level:
            # Count the number of non-null data-points for parent dimension
            parent_dim = design_request[0]
            parent_inputs = final_data_points[:, input_set_map[parent_dim.local_id]]
            parent_mask = parent_inputs > parent_dim.portion_null
            points_to_create = np.sum(parent_mask)

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
            encoded_flag, data_points = base_creator(active_dims, points_to_create)
            if base_level:
                # initalize input set
                for i, dim in enumerate(active_dims):
                    column_map[active_index] = dim
                    input_set_map[dim.local_id] = active_index
                    final_data_points[:, active_index] = data_points[:, i]

                    active_index += 1
                    if dim.has_child_dimensions():
                        if isinstance(dim, Variant):
                            design_request_stack.append((dim, dim.options))
                        else:
                            design_request_stack.append(
                                (dim, list(create_level_iterable(dim.children)))
                            )
            else:
                # inject design into input set
                for i, dim in enumerate(active_dims):
                    column_map[active_index] = dim
                    input_set_map[dim.local_id] = active_index
                    final_data_points[:, active_index][parent_mask] = data_points[:, i]

                    if dim.has_child_dimensions():
                        design_request_stack.append((dim, dim.children))
                    active_index += 1

    decoded_values = space.decode_zero_one_matrix(final_data_points, input_set_map)

    return DesignOfExperiment(
        input_space=space,
        input_sets=decoded_values,
        input_set_map=input_set_map,
        encoded_flag=False,
    )


def generate_seperate_designs_by_full_subspace(
    space: InputSpace,
    n_points: int,
    base_creator=_default_base_lhs_creator,
    ensure_at_least_one=True,
) -> DesignOfExperiment:

    total_dim_count = space.count_dimensions()

    final_data_points = np.full((n_points, total_dim_count), np.nan)
    active_index = 0
    column_map: Dict[str, Dimension] = {}
    input_set_map = {}

    full_subspace_sets = space.derive_full_subspaces()

    # compute portion of the n_points that each sub-design for each sub-space
    # should address
    portitions = compute_subspace_portitions(space, full_subspace_sets)

    dim_map = space.create_dim_map()

    # check if any of the subspaces would create duplicates
    # if any duplicates, we need to distribute these given the portitions
    n_extra_counts = 0
    point_count_override = {}
    last_space_to_place_extras = 0
    n_running_point_count = 0
    for i, portition, subspace in zip(
        range(len(full_subspace_sets)), portitions, full_subspace_sets
    ):
        points_to_create = round(portition * n_points)

        level_complexity_factor = 1.0
        for dim_id in subspace:
            dim = dim_map[dim_id]
            level_complexity_factor *= estimate_complexity(dim)
        level_complexity_factor = int(level_complexity_factor)
        if points_to_create > level_complexity_factor:
            n_extra_counts += points_to_create - level_complexity_factor
            point_count_override[i] = level_complexity_factor
            n_running_point_count += points_to_create - level_complexity_factor
        elif points_to_create < 1 and ensure_at_least_one:
            point_count_override[i] = 1
            n_extra_counts -= 1
            n_running_point_count += 1
        else:
            last_space_to_place_extras = i
            n_running_point_count += points_to_create

    points_left_to_allocate = n_points
    portition_weight = (n_extra_counts + n_points) / n_points
    lb_index = 0

    for i, portition, subspace in zip(
        range(len(full_subspace_sets)), portitions, full_subspace_sets
    ):
        if i in point_count_override:
            points_to_create = point_count_override[i]
        else:
            points_to_create = round(portition * n_points * portition_weight)

        if i == last_space_to_place_extras:
            pass

        if points_left_to_allocate < points_to_create:
            points_to_create = points_left_to_allocate

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
        # dimensions, exclude them and manually set the values of these dimensions
        fixed_dims = []
        active_dims = []

        for dim_id in subspace:
            dim = dim_map[dim_id]
            if isinstance(dim, (Variant, Composite)):
                fixed_dims.append(dim)
            else:
                active_dims.append(dim)

        if len(active_dims) > 0:
            encoded_flag, data_points = base_creator(active_dims, points_to_create)
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

                final_data_points[:, dim_index][row_mask] = decoded_data_points[:, i]

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
                        if child_dim.id in subspace:
                            v = i
                            break

                final_data_points[:, dim_index][row_mask] = v

        points_left_to_allocate = points_left_to_allocate - points_to_create

    return DesignOfExperiment(
        input_space=space,
        input_sets=final_data_points,
        input_set_map=input_set_map,
        encoded_flag=False,
    )


def generate_design_with_projection(
    space: InputSpace, n_points: int, base_creator=_default_base_lhs_creator
) -> DesignOfExperiment:
    active_dims = list(create_all_iterable(space.children))
    input_set_map = {}
    for i, dim in enumerate(active_dims):
        input_set_map[dim.id] = i

    encoded_flag, data_points = base_creator(active_dims, n_points)

    decoded_values = space.decode_zero_one_matrix(data_points, input_set_map, True)

    return DesignOfExperiment(
        input_space=space,
        input_sets=decoded_values,
        input_set_map=input_set_map,
        encoded_flag=False,
    )
