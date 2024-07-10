"""
    This module provide logic to create LatinHypercube designs for InputSpaces.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.stats.qmc import LatinHypercube

from ..spaces.dimensions import Dimension
from ..spaces.root import InputSpace, create_level_iterable
from .doe import DesignOfExperiment


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

    while len(design_request_stack) > 0:
        design_request = design_request_stack.pop(0)
        points_to_create = n_points
        base_level = design_request[0] is None
        if not base_level:
            # Count the number of non-null data-points for parent dimension
            parent_dim = design_request[0]
            parent_inputs = final_data_points[:, input_set_map[parent_dim.id]]
            parent_mask = parent_inputs > parent_dim.portion_null
            points_to_create = np.sum(parent_mask)

        # addressed fixed dimensions
        active_dims = design_request[1]
        encoded_flag, data_points = base_creator(active_dims, points_to_create)

        if base_level:
            # initalize input set
            for i, dim in enumerate(active_dims):
                column_map[active_index] = dim
                input_set_map[dim.id] = active_index
                final_data_points[:, active_index] = data_points[:, i]

                active_index += 1
                if dim.has_child_dimensions():
                    design_request_stack.append((dim, dim.children))
        else:
            # inject design into input set
            for i, dim in enumerate(active_dims):
                column_map[active_index] = dim
                input_set_map[dim.id] = active_index
                final_data_points[:, active_index][parent_mask] = data_points[:, i]

                if dim.has_child_dimensions():
                    design_request_stack.append((dim, dim.children))
                active_index += 1

    decoded_values = space.decode_zero_one_matrix(final_data_points, input_set_map)

    return DesignOfExperiment(
        input_sets=decoded_values,
        input_set_map=input_set_map,
        encoded_flag=False,
    )
