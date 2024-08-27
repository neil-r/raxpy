""" TODO Explain Module"""

import numpy as np

from ..spaces import dimensions as d
from ..spaces import root as s
from .doe import DesignOfExperiment, EncodingEnum


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

    flatted_dimensions = s.create_all_iterable(space.children)
    dim_map = {}
    for i, dim in enumerate(flatted_dimensions):
        dim_map[dim.local_id] = i
    random_x = np.random.rand(n_points, len(dim_map.keys()))

    return DesignOfExperiment(
        input_sets=random_x,
        input_set_map=dim_map,
        input_space=space,
        encoding=EncodingEnum.ZERO_ONE_RAW_ENCODING,
    )
