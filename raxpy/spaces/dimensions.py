from typing import Optional, List, Union
from dataclasses import dataclass


@dataclass
class Int:
    name:Optional[str] = None
    label:Optional[str] = None
    lb:Optional[int] = None
    ub:Optional[int] = None
    pass

@dataclass
class Float:
    name:str
    label:Optional[str] = None
    lb:Optional[float] = None
    ub:Optional[float] = None
    
    pass

@dataclass
class Text:
    name:str
    label:Optional[str] = None
    pass


@dataclass
class CategoryValue:
    value:str
    label:str
    pass

@dataclass
class Category:
    name:str
    label:Optional[str]
    values:Optional[List[Union[CategoryValue, str]]]
    pass

Dimension = Union[Category, Int, Float, Text]