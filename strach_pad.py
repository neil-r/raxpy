import inspect

from typing import Annotated, Type, Optional, Union

# Function to check if a parameter is annotated with metadata
def is_annotated_with_metadata(param_annotation):
    return hasattr(param_annotation, "__metadata__")

def get_base_type(param_annotation):
    if hasattr(param_annotation, '__origin__') and param_annotation.__origin__ is Optional:
        return param_annotation.__args__[0]
    return param_annotation

# Function to check if a type is optional
def is_optional_type(type_annotation):
    
    if hasattr(type_annotation, '__origin__'):
        if type_annotation.__origin__ is Optional:
            bt = get_base_type(type_annotation)
            return True, bt
        if type_annotation.__origin__ is Union:
            return type(None) in type_annotation.__args__, type_annotation.__args__[0]

    return False, type_annotation

def f(x1 : Optional[int], x2: Annotated[float,"this is a test"], x3 =4):
  pass

params = inspect.signature(f).parameters
for name, param in params.items():
  print(name)
  if param.default != param.empty:
    print(param.default)
  if param.annotation != param.empty:
    t:Type
    if is_annotated_with_metadata(param.annotation):
      t  = param.annotation.__origin__
      print(param.annotation.__metadata__)
    else:
      t = param.annotation
    print(t)
    print(is_optional_type(t))
  print("-----------------")
