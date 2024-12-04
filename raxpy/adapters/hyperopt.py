from typing import Optional, Any, Tuple, List, Dict
from hyperopt import hp, fmin, tpe
import numpy as np

from raxpy.spaces.complexity import assign_null_portions
from raxpy.spaces.dimensions import convert_values_from_dict
from raxpy.spaces.root import create_level_iterable

from ..annotations import function_spec
from ..does.doe import DesignOfExperiment
from ..spaces import (
    InputSpace,
    OutputSpace,
    Dimension,
    Int,
    Float,
    Variant,
    Composite,
    Bool,
)


def convert_input_space(input_space: InputSpace):
    """
    Converts a raxpy InputSpace to an hyperopt space.
    """

    hp_space = {}

    for dim in input_space.children:
        key, value = _convert_dimension(dim)

        hp_space[key] = value

    return hp_space


def _convert_dimension(dim: Dimension) -> Tuple[str, Any]:
    """
    Converts a raxpy dimension to a hyperopt dimension
    """
    dim_id = dim.local_id
    value = None
    if isinstance(dim, Float):
        if dim.value_set is not None:
            value = hp.choice(dim.id, dim.value_set)
        elif dim.has_tag("log"):
            value = hp.loguniform(dim.id, dim.lb, dim.ub)
        else:
            value = hp.uniform(dim.id, dim.lb, dim.ub)
    elif isinstance(dim, Int):
        if dim.value_set is not None:
            value = hp.choice(dim.id, dim.value_set)
        elif dim.has_tag("log"):
            value = hp.uniformint(dim.id, dim.lb, dim.ub + 1)
        else:
            value = hp.uniformint(dim.id, dim.lb, dim.ub + 1)
    elif isinstance(dim, Composite):
        child_values = {}

        for child_dim in dim.children:

            child_key, child_value = _convert_dimension(child_dim)

            child_values[child_key] = child_value

        value = child_values
    elif isinstance(dim, Variant):
        options = []

        for child_dim in dim.options:

            child_key, child_value = _convert_dimension(child_dim)

            options.append({child_key: child_value})

        value = hp.choice(dim.id, options)

    if dim.nullable and dim.portion_null > 0:
        value = hp.pchoice(
            f"{dim_id}##NULL",
            [(dim.portion_null, None), (1.0 - dim.portion_null, value)],
        )

    return dim_id, value


def convert_to_hp(
    f,
    input_space: Optional[InputSpace] = None,
    output_space: Optional[OutputSpace] = None,
):
    """
    Converts a raxpy annotated function f to a hyper-opt space
    and to a hyper-opt optimizable function.

    """
    if input_space is None:
        input_space = function_spec.extract_input_space(f)

    if output_space is None:
        output_space = function_spec.extract_output_space(f)

    # assign unassigned null poritions using complexity hueristic
    assign_null_portions(create_level_iterable(input_space.children))

    hp_space = convert_input_space(input_space)

    def hp_callable(config):
        args = convert_values_from_dict(input_space.children, config)

        results = f(**args)

        return results

    return hp_space, hp_callable


def convert_design(design: DesignOfExperiment) -> List[Dict]:
    """
    Converts a raxpy design of experiment to a form used by hyperopt.
    """
    hyper_opt_points = []
    dim_map = design.input_space.create_dim_map()
    for point in design.decoded_input_sets:
        h_point = {}
        for dim_id, index in design.input_set_map.items():
            dim = dim_map[dim_id]
            convert_value = False
            if dim.nullable:
                if np.isnan(point[index]):
                    h_point[f"{dim_id}##NULL"] = 0
                else:
                    h_point[f"{dim_id}##NULL"] = 1
                    convert_value = True
            else:
                convert_value = not np.isnan(point[index])

            if convert_value:
                if isinstance(dim, Bool):
                    # must convert Bool dimensions to 0, 1 integers
                    h_point[dim_id] = (
                        1 if dim.convert_to_argument(point[index]) else 0
                    )
                elif isinstance(dim, Int) or isinstance(dim, Float):
                    if dim.value_set is not None:
                        # represented as hp choice, must get index
                        h_point[dim_id] = dim.value_set.index(
                            dim.convert_to_argument(point[index])
                        )
                    else:
                        h_point[dim_id] = dim.convert_to_argument(point[index])
                elif isinstance(dim, Variant):
                    h_point[dim_id] = int(point[index])
                elif isinstance(dim, Composite):
                    pass
                    # h_point[dim_id] = dim.convert_to_argument(point[index])

        hyper_opt_points.append(h_point)

    return hyper_opt_points
