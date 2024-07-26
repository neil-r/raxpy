""" TODO Explain Module"""

from typing import Optional, Set, Union, Tuple, List
from dataclasses import dataclass

from ..spaces import dimensions as d


CategorySpec = Union[str, Tuple[str, str]]


@dataclass(frozen=True, eq=True)
class Base:
    """
    TODO Explain Class
    """

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
    """
    TODO Explain Class
    """

    value_set: Optional[List[CategorySpec]] = None


@dataclass(frozen=True, eq=True)
class Float(Base):
    """
    TODO Explain Class
    """

    ub: Optional[float] = None
    lb: Optional[float] = None
    value_set: Optional[Set[float]] = None


    def apply_to(self, d:d.Float):
        super().apply_to(d)
        d.lb = self.lb
        d.ub = self.ub
        d.value_set = self.value_set



@dataclass(frozen=True, eq=True)
class Integer(Base):
    """
    TODO Explain Class
    """

    ub: Optional[int] = None
    lb: Optional[int] = None
    value_set: Optional[Set[int]] = None

    def apply_to(self, d: d.Int):
        """
        TODO Explain the Function

        Arguments 
        ---------
        self : 
            **Explanation**
        d : d.Int
            **Explanation**

        """
        super().apply_to(d)
        d.lb = self.lb
        d.ub = self.ub
        d.value_set = self.value_set


@dataclass(frozen=True, eq=True)
class Binary(Base):
    """
    TODO Explain Class

    **Not Implemented**

    """

    pass
