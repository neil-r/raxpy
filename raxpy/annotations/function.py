import inspect
import typing
from typing import List
import dataclasses
from raxpy.spaces import dimensions as dim
from raxpy.spaces.root import Space

def list_function_parameters(func):
    """
    Takes a function and prints its name along with its parameters and their default values.
    
    Args:
    func (function): The function to introspect.
    """
    input_dimensions:List[dim.Dimension] = []
    print(f"Function: {func.__name__}")
    params = inspect.signature(func).parameters
    for name, param in params.items():
        d:dim.Dimension = dim.Int()

        if param.annotation is not inspect.Parameter.empty:
            pass
        else:
            print(f"{name}: {param.annotation if param.annotation is not inspect.Parameter.empty else 'No type annotation'} (no default value)")

        if param.default is inspect.Parameter.empty:
            # no default value
            pass
        else:
            print(f"{name}: {param.annotation if param.annotation is not inspect.Parameter.empty else 'No type annotation'} = {param.default}")

        input_dimensions.append(d)

    input_space = Space(
        dimensions=input_dimensions,
    )

    output_dimensions:List[dim.Dimension] = []
    output_space = Space(
        dimensions=output_dimensions
    )

    return (input_space, output_space)

class ValueRange:
    pass

@dataclasses.dataclass
class AdvancedType:

    pass

def x(a:float, b:str, c:typing.Annotated[int, dim.Int()]= 123):
    pass

list_function_parameters(x)