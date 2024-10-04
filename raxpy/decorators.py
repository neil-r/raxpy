""" 
    This module provides support function decorators,
    designed to work with raxpy input space introspection
    features.
"""

import sys

if sys.version_info >= (3, 10):
    from typing import Callable, TypeVar, Any, ParamSpec
else:
    from typing_extensions import Callable, TypeVar, Any, ParamSpec

from functools import wraps

from raxpy.annotations import function_spec
from raxpy.spaces.root import InputSpace


# Define type variables for parameter and return types
P = ParamSpec("P")
R = TypeVar("R")


def validate_function_inputs(space: InputSpace, args, kwargs) -> None:
    """
        Validates args and kwargs given the space constraints.
        Raises an exception if any values are invalid.

    Arguments
    ---------
    space : InputSpace
        The constraint specification of args and kwargs.
    args
        the arguments to validate
    kwargs
        the keyword arguments to validate

    """

    for i, dim in enumerate(space.children):
        specified_input = False
        if i < len(args):
            value = args[i]
            specified_input = True
        else:
            if dim.local_id not in kwargs:
                value = None
            else:
                value = kwargs[dim.local_id]
                specified_input = True
        dim.validate(value, specified_input)


def validate_at_runtime(check_inputs=True, check_outputs=True):
    """
        A function decorator that validates
        a function's inputs at runtime given parameter annotations.

    Arguments
    ---------
    check_inputs=True
        Flag to check the arguments provided to a function
    check_outputs=True
        Flag to check the returned values from a function

    Returns
    -------
    _validate_at_runtime
        a runtime validator function
    """

    def _validate_at_runtime(func: Callable[P, R]) -> Callable[P, R]:
        """
        Wraps func to implement runtime validation logic

        Arguments
        ---------
        func (Function) : Callable[P, R]
            The func to wrap

        Returns
        -------
        wrapper : Callable[P, R]
            the wrapped representation of func
        """
        input_space = function_spec.extract_input_space(func)

        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            """
            Implements runtime validation logic

            Arguments
            ---------
            *args : P.args
                the input arguments to consider
            **kwargs : P.kwargs
                the input keyword arguments to consider

            Returns
            -------
            outputs : R
                the function's returned value

            """

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
