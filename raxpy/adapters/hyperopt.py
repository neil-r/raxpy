from typing import Optional, Any, Tuple, List, Dict
from hyperopt import hp, fmin, tpe

from raxpy.spaces.dimensions import convert_values_from_dict

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
)


def convert_input_space(input_space: InputSpace):
    """ """

    hp_space = {}

    for dim in input_space.children:
        key, value = _convert_dimension(dim)

        hp_space[key] = value

    return hp_space


def _convert_dimension(dim: Dimension) -> Tuple[str, Any]:
    dim_id = dim.local_id
    value = None
    if isinstance(dim, Float):
        if dim.value_set is not None:
            value = hp.choice(dim.local_id, dim.value_set)
        elif dim.has_tag("log"):
            value = hp.loguniform(dim.lb, dim.ub)
        else:
            value = hp.uniform(dim.local_id, dim.lb, dim.ub)
    elif isinstance(dim, Int):
        if dim.value_set is not None:
            value = hp.choice(dim.local_id, dim.value_set)
        elif dim.has_tag("log"):
            value = hp.uniformint(dim.local_id, dim.lb, dim.ub + 1)
        else:
            value = hp.uniformint(dim.lb, dim.ub + 1)
    elif isinstance(dim, Composite):
        options = []

        for child_dim in dim.children:

            child_key, child_value = _convert_dimension(child_dim)

            options.append({child_key: child_value})

        value = hp.choice(dim.local_id, options)

    return dim_id, value


def convert_to_hp(
    f,
    input_space: Optional[InputSpace] = None,
    output_space: Optional[OutputSpace] = None,
):
    """ """
    if input_space is None:
        input_space = function_spec.extract_input_space(f)

    if output_space is None:
        output_space = function_spec.extract_output_space(f)

    hp_space = convert_input_space(input_space)

    def hp_callable(config):
        # TODO implement
        results = f(config["x1"], config["x2"])

        return results

    return hp_space, hp_callable


def convert_design(design: DesignOfExperiment) -> List[Dict]:

    value_dicts = design.input_space.convert_flat_values_to_dict(
        design.decoded_input_sets, design.input_set_map
    )
    return value_dicts
