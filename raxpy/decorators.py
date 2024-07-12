from typing import Callable, TypeVar, Any, ParamSpec

from functools import wraps

from raxpy.annotations import function_spec
from raxpy.spaces.root import InputSpace


# Define type variables for parameter and return types
P = ParamSpec("P")
R = TypeVar("R")


def validate_function_inputs(space: InputSpace, args, kwargs):

    for i, dim in enumerate(space.children):
        specified_input = False
        if i < len(args):
            value = args[i]
            specified_input = True
        else:
            if dim.id not in kwargs:
                value = None
            else:
                value = kwargs[dim.id]
                specified_input = True
        dim.validate(value, specified_input)


def validate_at_runtime(check_inputs=True, check_outputs=True):

    def _validate_at_runtime(func: Callable[P, R]) -> Callable[P, R]:
        input_space = function_spec.extract_input_space(func)

        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:

            if check_inputs:
                # validate the inputs
                validate_function_inputs(input_space, args, kwargs)
            outputs = func(*args, **kwargs)
            # You can add post-processing here
            if check_outputs:
                # validate the outputs
                # TODO implement output validation
                raise NotImplementedError("Output validation not implemented")

            return outputs

        return wrapper

    return _validate_at_runtime
