import numpy as np

from ..spaces import dimensions as d
from ..spaces import root as s
from .doe import DesignOfExperiment


def generate_design(space: s.InputSpace, n_points: int) -> DesignOfExperiment:

    flatted_dimensions = s.create_all_iterable(space.children)
    dim_map = {}
    for i, dim in enumerate(flatted_dimensions):
        dim_map[dim.local_id] = i
    random_x = np.random.rand(n_points, len(dim_map.keys()))
    input_sets = space.decode_zero_one_matrix(
        random_x, dim_map, map_null_to_children_dim=True
    )

    return DesignOfExperiment(input_sets=input_sets, input_set_map=dim_map)
