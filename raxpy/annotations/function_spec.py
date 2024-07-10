from typing import Optional, List, Union, Generic, TypeVar, Any, Callable, Type, Dict
from typing import List

import inspect
from dataclasses import dataclass

from raxpy.does.doe import DesignOfExperiment
from raxpy.spaces import dim_tags, dimensions as dim
from raxpy.spaces.root import InputSpace, OutputSpace
from .type_spec import map_type, UndefinedValue


def _convert_param(name: str, param: inspect.Parameter) -> dim.Dimension:
    if param.annotation is not inspect.Parameter.empty:
        # analyze static type for parameter specification of dimension
        t = param.annotation
        d = map_type(
            name,
            t,
            (
                param.default
                if param.default is not inspect.Parameter.empty
                else UndefinedValue
            ),
        )
    else:
        if param.default is inspect.Parameter.empty:
            # no default value and no static type spec
            d = dim.Float(id=name, nullable=False)
        else:
            # infer type given type of default value
            if param.default is None:
                d = dim.Float(id=name, nullable=True, default_value=None)
            else:
                t = type(param.default)

                d = map_type(name, t, None, param.default)

    return d


def extract_input_space(func: Callable) -> InputSpace:
    """
    Takes a function and dervies the input space of the function from the
    function parameters' static types and annotations.

    Args:
        func (function): The function to introspect.
    """
    input_dimensions: List[dim.Dimension] = []

    params = inspect.signature(func).parameters
    for name, param in params.items():
        d = _convert_param(name, param)
        input_dimensions.append(d)

    input_space = InputSpace(
        dimensions=input_dimensions,
    )

    return input_space


def extract_output_space(func: Callable) -> OutputSpace:
    output_dimensions: List[dim.Dimension] = []
    output_space = OutputSpace(dimensions=output_dimensions)
    return output_space


def map_design_to_function_spec(design: DesignOfExperiment, func: Callable):
    arg_sets = []

    for inputs in design.input_set:
        pass

    return arg_sets
