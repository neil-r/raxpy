""" TODO """

from typing import List, Callable

import inspect

from raxpy.spaces import dimensions as dim
from raxpy.spaces.root import InputSpace, OutputSpace
from .type_spec import map_type, UndefinedValue


def _convert_param(name: str, param: inspect.Parameter) -> dim.Dimension:
    """
    TODO Explain the Function

    Arguments
    ---------
    name : str
        **Explanation**
    param : inspect.Parameter
        **Explanation**

    Returns
    -------
    d : Dimension
        TODO **Map Type**?
    """
    if param.annotation is not inspect.Parameter.empty:
        # analyze static type for parameter specification of dimension
        t = param.annotation
        d = map_type(
            "",
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
            d = dim.Float(id=name, local_id=name, nullable=False)
        else:
            # infer type given type of default value
            if param.default is None:
                d = dim.Float(
                    id=name, local_id=name, nullable=True, default_value=None
                )
            else:
                t = type(param.default)

                d = map_type("", name, t, param.default)

    return d


def extract_input_space(func: Callable) -> InputSpace:
    """
    Takes a function and derives the input space of the function from
    the function parameters' static types and annotations.

    Arguments
    ---------
    func (function) : Callable
        The function to introspect.

    Returns
    -------
    input_space: Type InputSpace
        TODO**What is input Space?**
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
    """
    Takes a function and derives the output space of the function from
    the function parameters' static types and annotations.

    Arguments
    ---------
    func (function) : Callable
        The function to introspect.

    Returns
    -------
    input_space: Type OutputSpace
        TODO **Explanation**
    """
    output_dimensions: List[dim.Dimension] = []
    # TODO implement return type introspection logic
    output_space = OutputSpace(dimensions=output_dimensions)
    return output_space
