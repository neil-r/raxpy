""" 
    This module contains type-hint and annotation conversion
    logic to `raxpy.spaces` objects.
"""

import types
from dataclasses import MISSING, fields
from typing import List, Type, Union, get_args, get_origin

from .. import spaces as s


UndefinedValue = object()

_type_dimension_mapper: Type = {
    int: s.Int,
    float: s.Float,
    str: s.Text,
    bool: s.Bool,
}


def _is_annotated_with_metadata(type_annotation) -> bool:
    """
    Helper function to determine if type is annotated.

    Arguments
    ---------
    type_annotation : Type
        the type to inspect

    Returns
    -------
    bool
        True if type is annotated with meta-data

    """
    return hasattr(type_annotation, "__metadata__")


def list_dataclass_attributes(parent_prefix: str, cls) -> List[s.Dimension]:
    """
    Function to introspect and list the attributes of a dataclass

    Arguments
    ---------
    parent_prefix : str
        value to prefix dimension ids to ensure they are unique
    cls : Type
        the class to analyze

    Returns
    -------
    List[s.Dimension]
        list of dimensions mapped from annotated attributes
        of the data class

    """
    children_dims = []
    if not hasattr(cls, "__dataclass_fields__"):
        raise TypeError(f"{cls.__name__} is not a dataclass")

    attributes = fields(cls)
    for attr in attributes:
        t = attr.type
        child_dim = map_type(
            parent_prefix,
            attr.name,
            t,
            UndefinedValue if attr.default == MISSING else attr.default,
        )
        children_dims.append(child_dim)
    return children_dims


def _map_base_type(parent_prefix: str, t, initialization_values):
    """
    Helper function to map a type to a dimension type.

    Arguments
    ---------
    parent_prefix : str
        value to prefix dimension ids to ensure they are unique
    t : Type
        a variable decoded type
    initialization_values : dict
        A dict to mutate with initalization values for the dimension

    Returns
    -------
    dt : Type
        a type of dimension
    """
    if t in _type_dimension_mapper:
        dt = _type_dimension_mapper[t]
    else:
        if get_origin(t) is get_origin(List):
            dt = s.ListDim
            element_type = None

            a = get_args(t)
            if a is not None and len(a) == 1:
                element_type = map_type(parent_prefix, "_element", a[0])
            else:
                raise NotImplementedError(
                    f"Multiple List args not implemented: {a}"
                )

            initialization_values["element_type"] = element_type
        elif hasattr(t, "__dataclass_fields__"):
            dt = s.Composite
            initialization_values["children"] = list_dataclass_attributes(
                parent_prefix, t
            )
            initialization_values["type_class"] = t
        else:
            raise NotImplementedError(f"Type ({t}) not understood")
    return dt


def map_type(
    parent_prefix: str, name: str, base_type, default_value=UndefinedValue
) -> s.Dimension:
    """
    Maps a Python type to a dimension.

    Arguments
    ---------
    parent_prefix : str
        value to prefix dimension ids to ensure they are unique
    name : str
        the local name of the type
    base_type :
        a variable decoded type
    default_value=UndefinedValue
        a value to specify as a dimension's default value

    Returns
    -------
    d : Dimension
        the derived dimension
    """

    metadata = None
    if _is_annotated_with_metadata(base_type):
        metadata = base_type.__metadata__
        base_type = base_type.__origin__

    if parent_prefix != "":
        id: str = parent_prefix + "##" + name
    else:
        id: str = name
    child_parent_prefix = id
    initalization_values = {
        "local_id": name,
        "id": id,
        "default_value": (
            None if default_value is UndefinedValue else default_value
        ),
        "specified_default": (
            False if default_value is UndefinedValue else True
        ),
        "nullable": True if default_value is None else False,
    }

    dt = s.Float
    o = get_origin(base_type)
    if o is not None and (o is Union or o == types.UnionType):
        args = list(base_type.__args__)
        if type(None) in args:
            initalization_values["nullable"] = True
            args.remove(type(None))
        if len(args) == 1:
            child_type = args[0]
            dt = _map_base_type(id, child_type, initalization_values)
        else:
            dt = s.Variant
            options = []

            for i, a in enumerate(args):

                options.append(map_type(id, f"option_{i}", a))

            initalization_values["options"] = options
    else:
        dt = _map_base_type(
            child_parent_prefix, base_type, initalization_values
        )

    if dt == s.Bool:
        initalization_values["lb"] = 0
        initalization_values["ub"] = 1
    d = dt(**initalization_values)

    if metadata is not None:
        for m in metadata:
            if hasattr(m, "apply_to"):
                m.apply_to(d)
    return d
