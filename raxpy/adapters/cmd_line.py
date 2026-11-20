"""
Command-line adapter for calling functions.

Inspects the function signature, pulling command line
arguments and converting them to the appropriate types.
Then calls the function with those arguments.
"""

import argparse

import raxpy.annotations.function_spec as fs
from raxpy.spaces.dimensions import (
    convert_values_from_dict,
    Composite,
    Variant,
    List,
)
import raxpy.spaces.root as root_space


def _convert_param_to_arg(dimensions, param_dict):
    """
    Convert a parameter dictionary to command line arguments.

    Parameters
    ----------
    dimensions : list of raxpy.spaces.dimensions.Dimension
        The list of dimensions corresponding to the function parameters.
    param_dict : dict
        A dictionary mapping parameter names to their values.

    Returns
    -------

        A list of command line arguments.
    """
    kw_args = {}
    for dim in dimensions:
        if dim.id in param_dict:
            if isinstance(dim, Composite):
                child_kw_args = _convert_param_to_arg(dim.children, param_dict)
                kw_args[dim.local_id] = dim.type_class(**child_kw_args)
            elif isinstance(dim, Variant):
                option_id = param_dict[dim.id]
                # check if option_id is an int index or a string id
                # try to parse as an int first, if it fails, treat it as a string id
                option_dim = None
                assert dim.options is not None
                try:
                    if option_id is not None:
                        option_id = int(option_id)
                        option_dim = dim.options[option_id]
                except ValueError:
                    for opt in dim.options:
                        if opt.local_id == option_id or (
                            isinstance(opt, Composite)
                            and opt.class_name == option_id
                        ):
                            option_dim = opt
                            break
                if option_dim is not None:

                    if isinstance(option_dim, Composite):
                        child_kw_args = _convert_param_to_arg(
                            option_dim.children, param_dict
                        )
                        assert option_dim.type_class is not None
                        kw_args[dim.local_id] = option_dim.type_class(
                            **child_kw_args
                        )
                    else:
                        kw_args[dim.local_id] = param_dict[dim.id]
                elif option_id is not None:
                    raise ValueError(
                        f"Invalid option id '{option_id}' for variant dimension '{dim.id}'."
                    )
                else:
                    if dim.nullable:
                        kw_args[dim.local_id] = None
                    elif dim.default_value is not None:
                        kw_args[dim.local_id] = dim.default_value
                    else:
                        raise ValueError(
                            f"Variant dimension '{dim.id}' is not nullable and has no default value."
                        )
            elif isinstance(dim, List):
                raise NotImplementedError(
                    "List dimensions are not supported in command line arguments."
                )
            else:
                kw_args[dim.local_id] = param_dict[dim.id]
    return kw_args


def call(f):
    """
    Call a function with arguments extracted from the command line.

    Parameters
    ----------
    f : callable
        The function to call. The function should have type annotations for its parameters.
    Returns
    -------
        The return value of the function f.
    """
    parser = argparse.ArgumentParser()

    input_space = fs.extract_input_space(f)

    for dim in root_space.create_all_iterable(
        input_space.children, skip_structual_dims=True
    ):

        if isinstance(dim, Composite):
            raise ValueError(
                f"Composite dimensions are not supported in command line arguments: {dim}"
            )
        elif isinstance(dim, Variant):
            assert dim.options is not None
            parser.add_argument(
                f"--{dim.id}",
                type=str,
                nargs="?",
                default=dim.default_value,
                help=f"Type: {dim.__class__.__name__}, Options: {[opt.local_id for opt in dim.options] if hasattr(dim, 'options') else 'N/A'}",
            )
        elif isinstance(dim, List):
            raise ValueError(
                f"List dimensions are not supported in command line arguments: {dim}"
            )
        else:
            parser.add_argument(
                f"--{dim.id}",
                type=dim.acceptable_types()[0],
                nargs="?",
                default=dim.default_value,
                help=f"Type: {dim.__class__.__name__}, Bounds: {dim.bounds if hasattr(dim, 'bounds') else 'N/A'}, Value set: {dim.value_set if hasattr(dim, 'value_set') else 'N/A'}",
            )

    args = parser.parse_args()
    kwargs = vars(args)

    decoded_args = _convert_param_to_arg(input_space.children, kwargs)
    return f(**decoded_args)
