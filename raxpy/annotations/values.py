from typing import Optional, Set, Union, Tuple, List
from dataclasses import dataclass

from ..spaces import dimensions as d


CategorySpec = Union[str, Tuple[str, str]]


@dataclass(frozen=True, eq=True)
class Base:
    label: Optional[str] = None
    tags: Optional[List[str]] = None


@dataclass(frozen=True, eq=True)
class Categorical(Base):
    value_set: Optional[List[CategorySpec]] = None


@dataclass(frozen=True, eq=True)
class Float(Base):
    ub: Optional[float] = None
    lb: Optional[float] = None
    value_set: Optional[Set[float]] = None

    def apply_to(self, d: d.Float):
        d.lb = self.lb
        d.ub = self.ub
        d.value_set = self.value_set


@dataclass(frozen=True, eq=True)
class Integer(Base):
    ub: Optional[int] = None
    lb: Optional[int] = None
    value_set: Optional[Set[int]] = None

    def apply_to(self, d: d.Int):
        d.lb = self.lb
        d.ub = self.ub
        d.value_set = self.value_set


@dataclass(frozen=True, eq=True)
class Binary(Base):

    pass
