""" TODO Explain Module"""

from typing import (
    List,
    Union,
    Type,
    get_origin,
    get_args,
)
from raxpy.spaces import dimensions as dim
from dataclasses import fields, MISSING


UndefinedValue = object()


_type_dimension_mapper: Type = {
    int: dim.Int,
    float: dim.Float,
    str: dim.Text,
    # float:_map_str_parameter,
}


def _is_annotated_with_metadata(type_annotation) -> bool:
    """
    TODO Explain the Function

    Arguments
    ---------
    type_annotation : 
        **Explanation**

    Returns
    -------
    hasattr : bool
        **Explanation**

    """
    return hasattr(type_annotation, "__metadata__")


def list_dataclass_attributes(parent_prefix: str, cls) -> List:
    """
    TODO Function to introspect and list the attributes of a dataclass

    Arguments
    ---------
    parent_prefix : str
        **Explanation**
    cls : 
        **Explanation**
    
    Returns
    -------
    children_dims : List
        **Explanation**

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
    TODO Explain the Function

    Arguments
    ---------
    parent_prefix : str
        **Explanation**
    t : 
        **Explanation**
    initialization_values : 
        **Explanation**

    Returns
    -------
    dt : 
        **Explanation**

    """
    if t in _type_dimension_mapper:
        dt = _type_dimension_mapper[t]
    else:
        if get_origin(t) is get_origin(List):
            dt = dim.Listing
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
            dt = dim.Composite
            initialization_values["children"] = list_dataclass_attributes(
                parent_prefix, t
            )
            initialization_values["type_class"] = t
        else:
            raise NotImplementedError(f"Type ({t}) not understood")
    return dt


def map_type(
    parent_prefix: str, name: str, base_type, default_value=UndefinedValue
) -> dim.Dimension:
    """
    TODO Explain the Function

    Arguments
    ---------
    parent_prefix : str
        **Explanation**
    name : str
        **Explanation**
    base_type : 
        **Explanation**
    default_value=UndefinedValue
        **Explanation**

    Returns
    -------
    d : Dimension
        **Explanation**
    
    """

    metadata = None
    if _is_annotated_with_metadata(base_type):
        metadata = base_type.__metadata__
        base_type = base_type.__origin__

    id: str = parent_prefix + name
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

    dt = dim.Float
    o = get_origin(base_type)
    if o is not None and o is Union:
        args = list(base_type.__args__)
        if type(None) in args:
            initalization_values["nullable"] = True
            args.remove(type(None))
        if len(args) == 1:
            child_type = args[0]
            dt = _map_base_type(id, child_type, initalization_values)
        else:
            dt = dim.Variant
            options = []

            for i, a in enumerate(args):

                options.append(map_type(id, f"option_{i}", a))

            initalization_values["options"] = options
    else:
        dt = _map_base_type(parent_prefix, base_type, initalization_values)

    d = dt(**initalization_values)

    if metadata is not None:
        for m in metadata:
            if hasattr(m, "apply_to"):
                m.apply_to(d)
    return d


def map_primative_input(d: dim.Dimension, input) -> None:
    """
    Explain the Function **Not Implemented**

    Arguments
    ---------
    d : Dimension
        **Explanation**
    input : 
        **Explanation**

    Returns
    -------

    """
    pass
