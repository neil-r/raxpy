"""
Provides adapters to use raxpy with ConfigSpace code and
experiment environments
"""

import ConfigSpace
from ConfigSpace.hyperparameters.categorical import CategoricalHyperparameter
from ConfigSpace.hyperparameters.ordinal import OrdinalHyperparameter
from ConfigSpace.hyperparameters.uniform_float import (
    UniformFloatHyperparameter,
)
from ConfigSpace.hyperparameters.uniform_integer import (
    UniformIntegerHyperparameter,
)

from raxpy.spaces.dimensions import Bool, Float, Int, Text
from raxpy.spaces.root import InputSpace
import raxpy.spaces.dim_tags as dim_tags


def convert_config_space(
    config_space: ConfigSpace.ConfigurationSpace,
) -> InputSpace:
    """
    Converts a input space specified in ConfigSpace to a raxpy input space
    specification

    Arguments
    ---------
    config_space : ConfigSpace.ConfigurationSpace
        The input space specified by ConfigSpace

    Returns
    -------
    InputSpace
        the input space specification as a raxpy data structure
    """
    dimensions = []

    for hp_name in config_space.unconditional_hyperparameters:
        hp = config_space.get(hp_name)
        if isinstance(hp, UniformFloatHyperparameter):
            tags = []
            if hp.log:
                tags.append(dim_tags.LOG)
            dimensions.append(
                Float(id=hp_name, lb=hp.lower, ub=hp.upper, tags=tags)
            )

        elif isinstance(hp, UniformIntegerHyperparameter):
            tags = []
            if hp.log:
                tags.append(dim_tags.LOG)
            dimensions.append(
                Int(id=hp_name, lb=hp.lower, ub=hp.upper, tags=tags)
            )
        elif isinstance(hp, OrdinalHyperparameter):
            tags = [dim_tags.ORDINAL]
            value_set = tuple(x for x in hp.sequence)
            if isinstance(value_set[0], int):
                dimensions.append(
                    Int(
                        id=hp_name,
                        value_set=value_set,
                        tags=tags,
                    )
                )
            elif isinstance(value_set[0], float):
                dimensions.append(
                    Float(
                        id=hp_name,
                        value_set=value_set,
                        tags=tags,
                    )
                )
            else:
                dimensions.append(
                    Text(
                        id=hp_name,
                        value_set=value_set,
                        tags=tags,
                    )
                )
        elif isinstance(hp, CategoricalHyperparameter):
            tags = []
            value_set = tuple(x for x in hp.choices)
            if isinstance(value_set[0], int):
                dimensions.append(
                    Int(
                        id=hp_name,
                        value_set=value_set,
                        tags=tags,
                    )
                )
            elif isinstance(value_set[0], bool):
                dimensions.append(
                    Bool(
                        id=hp_name,
                        value_set=value_set,
                        tags=tags,
                    )
                )
            elif isinstance(value_set[0], float):
                dimensions.append(
                    Float(
                        id=hp_name,
                        value_set=value_set,
                        tags=tags,
                    )
                )
            else:
                dimensions.append(
                    Text(
                        id=hp_name,
                        value_set=value_set,
                        tags=tags,
                    )
                )
        else:
            raise NotImplementedError(
                f"Not implemented, failed to convert configspace paramter of type '{type(hp)}'"
            )
    if len(config_space.conditional_hyperparameters) > 0:
        raise NotImplementedError(
            "Not implemented, failed to convert conditional hyperparmaeters"
        )
    space = InputSpace(dimensions)
    return space
