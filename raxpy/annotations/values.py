from dataclasses import dataclass
from typing import Optional, Set, Union, Tuple, List

from ..spaces import dimensions as d


CategorySpec = Union[str, Tuple[str,str]]


@dataclass
class Base:
    label:Optional[str] = None
    tags:Optional[List[str]] = None

@dataclass
class Categorical(Base):
    value_set:Optional[List[CategorySpec]] = None
    

@dataclass
class Float(Base):
    ub:Optional[float] = None
    lb:Optional[float] = None
    value_set:Optional[Set[float]] = None

    def apply_to(self, d:d.Float):
        d.lb = self.lb
        d.ub = self.ub
        d.value_set = self.value_set


@dataclass
class Integer(Base):
    ub:Optional[int] = None
    lb:Optional[int] = None
    value_set:Optional[Set[int]] = None


    def apply_to(self, d:d.Int):
        d.lb = self.lb
        d.ub = self.ub
        d.value_set = self.value_set


@dataclass
class Binary(Base):

    pass