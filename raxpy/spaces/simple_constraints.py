""" TODO Explain Module"""

from typing import Set, Type, Generic, Optional, TypeVar


T = TypeVar("T")


class RangeConstraint(Generic[T]):
    """
    TODO Explain Class

    """

    UpperBound: Optional[T] = None
    LowerBound: Optional[T] = None


class ValueSetConstraint(Generic[T]):
    """
    TODO Explain Class

    """
    AcceptableValues: Set[T]
