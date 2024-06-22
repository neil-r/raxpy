import inspect
from typing import Optional, List, Union, Generic, TypeVar, Any, Callable, Type, Dict
from typing import List
from dataclasses import dataclass
from raxpy.spaces import dim_tags, dimensions as dim
from raxpy.spaces.root import InputSpace, OutputSpace



def _is_annotated_with_metadata(param_annotation):
    return hasattr(param_annotation, "__metadata__")

def _map_int_parameter(name:str, param:inspect.Parameter, optional:bool = False) -> dim.Int:


    return dim.Int(
        id=name,
        optional=optional

    )

def _map_union(name:str, param:inspect.Parameter, optional:bool = False) -> dim.Dimension:
    pass


_type_dimension_mapper:Type = {
    int:dim.Int,
    float:dim.Float,
    #float:_map_str_parameter,
}


def _map_type(name:str, base_type, metadata, default_value) -> dim.Dimension:
    initalize_values = {
        "id":name,
        "default_value":default_value,
        "optional": False,
    }

    dt = dim.Float
    if hasattr(base_type, "__origin__") and base_type.__origin__ is Union:
        args = list(base_type.__args__)
        if type(None) in args:
            initalize_values["optional"] = True
            args.remove(type(None))
        if len(args) == 1:
            child_type = args[0]
            if child_type in _type_dimension_mapper:
                dt = _type_dimension_mapper[child_type]
            else:
                raise NotImplementedError(f"Child type ({child_type}) not implemented")
        else:
            raise NotImplementedError("Choice not implemented")
        pass
    else:
        if base_type in _type_dimension_mapper:
            dt = _type_dimension_mapper[base_type]
        else:
            raise NotImplementedError(f"Child type ({base_type}) not implemented")

    d = dt(**initalize_values)

    if metadata is not None:
        for m in metadata:
            if hasattr(m,"apply_to"):
                m.apply_to(d)
    return d


def _convert_param(name:str, param:inspect.Parameter) -> dim.Dimension:
    if param.annotation is not inspect.Parameter.empty:
        t  = param.annotation
        metadata = None
        if _is_annotated_with_metadata(param.annotation):
            t  = param.annotation.__origin__
            metadata = param.annotation.__metadata__
                
        d = _map_type(name, t, metadata, param.default if param.default is not inspect.Parameter.empty else None)
    else:
        if param.default is inspect.Parameter.empty:
            # no default value
            d = dim.Float(id=name, optional=False)
        else:
            if param.default is None:
                d = dim.Float(id=name, optional=True)
            else:
                t = type(param.default)
                
                if isinstance(t,int):
                    d = dim.Int(
                        id=name,
                        optional=False,
                        default_value=param.default,
                        tags=[dim_tags.FIXED]
                    )
                elif isinstance(t,float):
                    d = dim.Float(
                        id=name,
                        optional=False,
                        default_value=param.default,
                        tags=[dim_tags.FIXED]
                    )
                else:
                    raise ValueError(f"Did not under this parameter: {param}")
            print(f"{name}: {param.annotation if param.annotation is not inspect.Parameter.empty else 'No type annotation'} = {param.default}")
    
    return d


def extract_input_space(func: Callable) -> InputSpace:
    """
    Takes a function and dervies the input space of the function from the
    function parameters' static types and annotations.
    
    Args:
        func (function): The function to introspect.
    """
    input_dimensions:List[dim.Dimension] = []
    
    params = inspect.signature(func).parameters
    for name, param in params.items():
        d = _convert_param(name,param)
        input_dimensions.append(d)

    input_space = InputSpace(
        dimensions=input_dimensions,
    )

    return input_space


def extract_output_space(func: Callable) -> OutputSpace:
    output_dimensions:List[dim.Dimension] = []
    output_space = OutputSpace(
        dimensions=output_dimensions
    )
    return output_space
