from typing import Optional, List, Union, Generic, TypeVar, Set
from dataclasses import dataclass


T = TypeVar('T')


@dataclass
class Dimension(Generic[T]):
    id:str
    optional:bool
    label:Optional[str] = None
    default_value:Optional[T] = None
    tags:Optional[List[str]] = None


@dataclass
class Int(Dimension[int]):
    lb:Optional[int] = None
    ub:Optional[int] = None
    value_set :Optional[Set[int]] = None


@dataclass
class Float(Dimension[float]):
    lb:Optional[float] = None
    ub:Optional[float] = None
    value_set :Optional[Set[float]] = None


@dataclass
class Text(Dimension[str]):
    length_limit:Optional[int] = None


@dataclass
class CategoryValue:
    value:str


@dataclass
class Category(Dimension[str]):
    values:Optional[List[Union[CategoryValue, str]]] = None


@dataclass
class Choice(Dimension):
    options:Optional[List[Dimension]] = None


@dataclass
class Listing(Dimension[List]):
    element_type:Optional[Dimension] = None
    cardinality_lb:Optional[int] = None
    cardinality_ub:Optional[int] = None


@dataclass
class Object(Dimension):
    children:Optional[List[Dimension]] = None
