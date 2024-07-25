from typing import Optional, Set, Union, Tuple, List
from dataclasses import dataclass

from ..spaces import dimensions as d


CategorySpec = Union[str, Tuple[str, str]]


@dataclass(frozen=True, eq=True)
class Base:
    label: Optional[str] = None
    tags: Optional[List[str]] = None

    def apply_to(self, d:d.Dimension):
        if self.label is not None:
            d.label = self.label
        if self.tags is not None:
            if d.tags is None:
                d.tags = self.tags
            else:
                d.tags.extend(self.tags)


@dataclass(frozen=True, eq=True)
class Categorical(Base):

    value_set:Optional[List[CategorySpec]] = None

    def apply_to(self, dim:d.Text):
        super().apply_to(dim)
        if self.value_set is not None:
            dim.value_set = [d.CategoryValue(value=v) if isinstance(v,str) else d.CategoryValue(value=v[0], label=v[1])  for v in self.value_set]



@dataclass(frozen=True, eq=True)
class Float(Base):
    ub: Optional[float] = None
    lb: Optional[float] = None
    value_set: Optional[Set[float]] = None


    def apply_to(self, d:d.Float):
        super().apply_to(d)



@dataclass(frozen=True, eq=True)
class Integer(Base):
    ub: Optional[int] = None
    lb: Optional[int] = None
    value_set: Optional[Set[int]] = None


    def apply_to(self, d:d.Int):
        super().apply_to(d)



@dataclass(frozen=True, eq=True)
class Binary(Base):

    pass
