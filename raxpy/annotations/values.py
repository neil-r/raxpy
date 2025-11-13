"""
This modules contains classes used to Annotate function
parameters.
"""

from typing import Optional, Set, Union, Tuple, List, cast
from dataclasses import dataclass

from .. import spaces as s


CategorySpec = Union[str, Tuple[str, str]]


@dataclass(frozen=True, eq=True)
class Base:
    """
    Parameter annotation abstract class.
    """

    label: Optional[str] = None
    tags: Optional[List[str]] = None
    portion_null: Optional[float] = None

    def apply_to(self, d: s.Dimension) -> None:
        """
        Applies the annotated attributes to the
        dimension d.

        Arguments
        ---------
        self : Base

        d : d.Dimension
            a dimension to apply annotation attributes onto
        """
        if self.label is not None:
            d.label = self.label
        if self.tags is not None:
            if d.tags is None:
                d.tags = self.tags
            else:
                d.tags.extend(self.tags)
        if self.portion_null is not None:
            d.portion_null = self.portion_null


@dataclass(frozen=True, eq=True)
class Categorical(Base):
    """
    Annotation for a str parameter representing a finite set of values.
    """

    value_set: Optional[Tuple[CategorySpec]] = None


@dataclass(frozen=True, eq=True)
class Float(Base):
    """
    Annotation for a float parameter.
    """

    ub: Optional[float] = None
    lb: Optional[float] = None
    value_set: Optional[Tuple[float]] = None

    def apply_to(self, d: s.Dimension):
        """
        Applies the annotated attributes to the
        dimension d.

        Arguments
        ---------
        self : Base

        d : d.Dimension
            a dimension to apply annotation attributes onto
        """
        super().apply_to(d)
        d = cast(s.Float, d)

        d.lb = self.lb
        d.ub = self.ub
        d.value_set = self.value_set


@dataclass(frozen=True, eq=True)
class Integer(Base):
    """
    Annotation for a int parameter.
    """

    ub: Optional[int] = None
    lb: Optional[int] = None
    value_set: Optional[Tuple[int]] = None

    def apply_to(self, d: s.Dimension):
        """
        Applies the annotated attributes to the
        dimension d.

        Arguments
        ---------
        self : Base

        d : d.Dimension
            a dimension to apply annotation attributes onto
        """
        super().apply_to(d)
        d = cast(s.Int, d)

        d.lb = self.lb
        d.ub = self.ub
        d.value_set = self.value_set


@dataclass(frozen=True, eq=True)
class Binary(Base):
    """
    Annotation for a bool parameter.
    """
