from typing import Set, Type, Generic, Optional, TypeVar


T = TypeVar('T')

class RangeConstraint(Generic[T]):
  UpperBound:Optional[T] = None
  LowerBound:Optional[T] = None


class ValueSetConstraint(Generic[T]):
  AcceptableValues:Set[T]
