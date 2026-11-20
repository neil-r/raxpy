"""
Defines factorial designs for experimental design.
"""

from itertools import product

import numpy as np

import raxpy
from raxpy.spaces import create_all_iterable
from raxpy.spaces.dimensions import Composite, Float, Int, Text, Variant

from .doe import DesignOfExperiment, EncodingEnum


def _get_dimension_levels(
    dimension, n_levels: int, active_dim_ids: list[str]
) -> np.ndarray:
    """Return zero-one encoded levels for an active dimension."""
    if isinstance(dimension, Float) and dimension.value_set is None:
        return (
            np.linspace(0.0, 1.0, n_levels)
            if n_levels > 1
            else np.array([0.5])
        )

    if isinstance(dimension, Composite):
        return np.array([1.0])

    if isinstance(dimension, Variant):
        if dimension.options is None:
            raise ValueError(
                f"Variant dimension '{dimension.id}' has no options"
            )
        for option_index, option in enumerate(dimension.options):
            if option.id in active_dim_ids:
                return np.array([option_index])
        raise ValueError(
            f"Unable to determine the active option for variant "
            f"dimension '{dimension.id}'"
        )
    elif isinstance(dimension, (Int, Text, Float)):
        discrete_values = dimension.get_discrete_values()
        if discrete_values is None:
            raise ValueError(
                f"Unable to determine categorical levels for dimension "
                f"'{dimension.id}'"
            )
        level_count = len(discrete_values)
    else:
        raise ValueError(
            f"Unable to determine categorical levels for dimension "
            f"'{dimension.id}'"
        )

    if level_count == 0:
        raise ValueError(f"Dimension '{dimension.id}' has no levels")
    if n_levels == 1 or level_count == 1:
        return np.array([0.0])
    return np.linspace(0.0, 1.0, level_count)


def generate_full_factorial_design(
    input_space: raxpy.spaces.InputSpace,
    n_levels: int = 2,
    _seed: int | None = None,
) -> DesignOfExperiment:
    """
    Generates a full factorial design with the specified
    number of levels and factors. If n_levels is 1, the
    design will contain a single point for each concept-subspace
    at the center of the sub-space.

    For dimensions that are nullable, the design will include points
    with null values for those dimensions, as well as points with
    non-null values. The number of levels for Float dimensions,
    without value_set specifications,
    is determined by the n_levels parameter, and the null value is
    represented as an additional level. All other dimensions are
    treated as categorical, and the design will include all
    combinations of their levels unless n_levels = 1, then
    the first level of the dimension will be used.
    Variant dimensions use their selected option's integer index in
    zero-one-null encoding, rather than a normalized value between zero and one.



    Arguments
    ---------
    input_space: raxpy.spaces.InputSpace
        The input space defining the factors and their levels.
    n_levels: int
        The number of levels for each continuous factor.
    _seed: int | None
        An optional seed for random number generation, not used in this function
        but included for consistency with other design generation functions.

    Returns
    -------
    DesignOfExperiment
        A design containing all combinations of factor levels.
    """
    if n_levels < 1:
        raise ValueError("n_levels must be greater than zero")

    dimensions = list(create_all_iterable(input_space.dimensions))
    seen_dimension_ids = set()
    dimensions = [
        dimension
        for dimension in dimensions
        if not (
            dimension.only_supports_spec_structure()
            or dimension.id in seen_dimension_ids
            or seen_dimension_ids.add(dimension.id)
        )
    ]
    input_set_map = {
        dimension.id: index for index, dimension in enumerate(dimensions)
    }
    input_sets = []

    for active_dim_ids in input_space.derive_concept_subspaces():
        active_dimensions = [
            dimension
            for dimension in dimensions
            if dimension.id in active_dim_ids
        ]
        active_column_indexes = [
            input_set_map[dimension.id] for dimension in active_dimensions
        ]
        active_levels = [
            _get_dimension_levels(dimension, n_levels, active_dim_ids)
            for dimension in active_dimensions
        ]
        subspace_sets = product(*active_levels)
        for point in subspace_sets:
            row = np.full(len(dimensions), np.nan)
            row[active_column_indexes] = point
            input_sets.append(row)

    input_sets = np.asarray(input_sets, dtype=float)

    return DesignOfExperiment(
        input_space=input_space,
        input_sets=input_sets,
        input_set_map=input_set_map,
        encoding=EncodingEnum.ZERO_ONE_NULL_ENCODING,
    )
