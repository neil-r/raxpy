from typing import List, Callable, get_type_hints

import inspect

from raxpy.spaces import dimensions as dim
from raxpy.spaces.root import InputSpace, OutputSpace
from .type_spec import map_type, UndefinedValue


ID_ROOT_RETURN = "y"

def _convert_param(name: str, param: inspect.Parameter) -> dim.Dimension:
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
                d = dim.Float(id=name, local_id=name, nullable=True, default_value=None)
            else:
                t = type(param.default)

                d = map_type("", name, t, param.default)

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
    signature = inspect.signature(func)

    return_annotation = signature.return_annotation

    # TODO remove the following if not needed for older versions of Python    
    # Get the type hints, including the return type
    # type_hints = get_type_hints(func)

    # TODO find way to avoid inspect._empty reference
    if return_annotation is not None and return_annotation != inspect._empty:
        base_output_dim = map_type("",ID_ROOT_RETURN,return_annotation)

        output_dimensions.append(base_output_dim)
    
    return OutputSpace(dimensions=output_dimensions)
