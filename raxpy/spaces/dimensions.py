from typing import Optional, List, Union, Generic, TypeVar, Set
from dataclasses import dataclass


T = TypeVar('T')


def _map_values(x, value_set, portion_null):
    value_count = len(value_set)
    
    return [value_set[int(((xp - portion_null)/(1.0-portion_null))//value_count)] if xp is not None and xp > portion_null else None for xp in x]


@dataclass
class Dimension(Generic[T]):
    id:str
    nullable:bool
    specified_default:bool = False
    label:Optional[str] = None
    default_value:Optional[T] = None
    tags:Optional[List[str]] = None
    portion_null:Optional[float] = None

    def has_child_dimensions(self) -> bool:
        return False
    
    def only_supports_spec_structure(self)->bool:
        return False
    
    def collapse_uniform(self, x):
        raise NotImplementedError("Abstract method, subclass should implement this method")
    



@dataclass
class Int(Dimension[int]):
    lb:Optional[int] = None
    ub:Optional[int] = None
    value_set :Optional[List[int]] = None

    def collapse_uniform(self, x):
        vs = None
        if self.value_set is not None:
            vs = self.value_set  
        else:
            if self.lb is not None and self.ub is not None:
                vs = list(range(self.lb, self.ub+1))
        
        if vs is not None:
            return _map_values(x, vs, self.portion_null)
        raise ValueError("Unbounded Int dimension cannot transform a uniform 0-1 value")


@dataclass
class Float(Dimension[float]):
    lb:Optional[float] = None
    ub:Optional[float] = None
    value_set :Optional[List[float]] = None

    def collapse_uniform(self, x):
        if self.value_set is not None:
            return _map_values(x, self.value_set, self.portion_null)
        
        if self.lb is not None and self.ub is not None:
            r = self.ub - self.lb
            if self.portion_null is not None:
                return [self.lb + r * ((xp - self.portion_null)/(1.0-self.portion_null)) if xp is not None and xp > self.portion_null else None for xp in x]
            else:
                return [self.lb + r * xp if xp is not None else None for xp in x]
        raise ValueError("Unbounded Float dimension cannot transform a uniform 0-1 value")


@dataclass
class CategoryValue:
    value:str


@dataclass
class Text(Dimension[str]):
    length_limit:Optional[int] = None
    value_set:Optional[List[Union[CategoryValue, str]]] = None

    def collapse_uniform(self, x):
        if self.value_set is not None:
            return _map_values(x, {v.value for v in self.value_set}, self.portion_null)
        raise ValueError("Unbounded Text dimension cannot transform a uniform 0-1 value")
    


@dataclass
class Variant(Dimension):
    options:Optional[List[Dimension]] = None


@dataclass
class Listing(Dimension[List]):
    element_type:Optional[Dimension] = None
    cardinality_lb:Optional[int] = None
    cardinality_ub:Optional[int] = None


@dataclass
class Composite(Dimension):
    class_name:Optional[str] = ""
    children:Optional[List[Dimension]] = None

    def collapse_uniform(self, x):
        return _map_values(x, [1,], self.portion_null)

    def only_supports_spec_structure(self)->bool:
        return not self.nullable

    def has_child_dimensions(self) -> bool:
        return True

    def count_children_dimensions(self) -> int:
        return sum([c.count_children_dimensions() if c.has_child_dimensions() else 1 for c in self.children])
