""" TODO Explain Module """

from dataclasses import dataclass
from typing import (Any, Dict, Generic, List, Optional, Set, Tuple, Type,
                    TypeVar, Union)

import numpy as np

T = TypeVar("T")


def _map_values(x, value_set, portion_null) -> List[Union[int, float]]:
    """
    TODO Explain the Function

    Arguments
    ---------
    x
        **Explanation**
    value_set
        **Explanation**
    portion_null
        **Explanation**

    Returns
    -------
    value_set : List[Union[int, float]]
        **Explanation**

    """
    value_count = len(value_set)
    boundary_size = 1.0 / value_count

    if portion_null is not None:
        return [
            (
                value_set[
                    int(
                        ((xp - portion_null) / (1.0 - portion_null))
                        // boundary_size
                    )
                ]
                if (not np.isnan(xp)) and xp > portion_null
                else np.nan
            )
            for xp in x
        ]
    else:
        return [
            value_set[int(xp // boundary_size)] if not np.isnan(xp) else np.nan
            for xp in x
        ]


def convert_values_from_dict(dimensions, input_value: Dict[str, Any]) -> Dict:
    """
    TODO Explain the Function

    Arguments
    ---------
    dimensions
        **Explanation**
    input_value : Dict[str, Any]
        **Explanation**

    Returns
    -------
    args : Dict
        **Explanation**

    """
    args = {}

    for dim in dimensions:
        if dim.id in input_value:
            child_value = input_value[dim.id]
            if child_value is None:
                if dim.nullable:
                    args[dim.id] = None
                else:
                    # TODO reassess this logic, may want to raise error
                    args[dim.id] = dim.default_value
            else:
                args[dim.id] = dim.convert_to_argument(child_value)
        else:
            args[dim.id] = dim.default_value
    return args


@dataclass
class Dimension(Generic[T]):
    """
    TODO Explain Class
    """

    id: str = ""
    local_id: str = ""
    nullable: bool = False
    specified_default: bool = False
    label: Optional[str] = None
    default_value: Optional[T] = None
    tags: Optional[List[str]] = None
    portion_null: Optional[float] = None

    def __post_init__(self):
        """
        TODO Explain the Function

        Raises
        ------
        ValueError:
            **If value is invalid**

        """
        if self.id == "":
            self.id = self.local_id
        if self.local_id == "":
            self.local_id = self.id
        if self.id == "":
            raise ValueError("Invalid identifier for dimension")

    def has_child_dimensions(self) -> bool:
        """
        TODO Explain the Function **Not Implemented?**
        """
        return False

    def only_supports_spec_structure(self) -> bool:
        """
        TODO Explain the Function **Not Implemented?**
        """
        return False

    def collapse_uniform(self, x, utilize_null_portions=True):
        """
        TODO Explain the Function **Not Implemented?**

        Arguments
        ---------
        self
            **Explanation**
        x
            **Explanation**
        utilize_null_portions=True
            **Explanation**

        """
        raise NotImplementedError(
            "Abstract method, subclass should implement this method"
        )

    def has_tag(self, tag: str) -> bool:
        """
        Function validates if there is a valid tag.

        Arguments
        ---------
        self
            **Explanation**
        tag : str
            **Explanation**

        Returns
        -------
        Returns True if there is a valid tag and
        False if there is not a tag

        """
        return self.tags is not None and tag in self.tags

    def convert_to_argument(self, input_value) -> T:
        """
        TODO Explain the Function **Not Implemented?**

        Arguments
        ---------
        self
            **Explanation**
        input_value
            **Explanation**

        """
        raise NotImplementedError(
            "Abstract method, subclass should implement this method"
        )

    def acceptable_types(self) -> Tuple[Type]:
        """
        TODO Explain the Function**Not Implemented?**

        Arguments
        ---------
        self
            **Explanation**

        """
        raise NotImplementedError(
            "Abstract method, subclass should implement this method"
        )

    def validate(self, input_value, specified_input: bool):
        """
        TODO Explain the Function

        Arguments
        ---------
        self
            **Explanation**
        input_value
            **Explanation**
        specified_input : bool
            **Explanation**

        Returns
        -------
        TODO **Explanation**

        """
        if input_value is None:
            if self.nullable:
                return

            if specified_input:
                raise ValueError(
                    f"Invalid value, dimension '{self.id}' should not be null"
                )
            if not self.specified_default:
                raise ValueError(
                    f"Invalid value, dimension '{self.id}' should be "
                    f"specified, no default provided"
                )

            return

        if not isinstance(input_value, self.acceptable_types()):
            raise ValueError(
                f"Invalid value type, dimension '{self.id}' should be a {T}"
            )


@dataclass
class Int(Dimension[int]):
    """
    TODO Explain Class
    """

    lb: Optional[int] = None
    ub: Optional[int] = None
    value_set: Optional[List[int]] = None

    def convert_to_argument(self, input_value) -> T:
        """
        TODO Explain the Function

        Arguments
        ---------
        self
            **Explanation**
        input_value
            **Explanation**

        Returns
        -------
        Returns input value as an integer
        """
        return int(input_value)

    def collapse_uniform(
        self, x, utilize_null_portions=True
    ) -> List[Union[int, float]]:
        """
        TODO Explain the Function

        Arguments
        ---------
        self
            **Explanation**
        x
            **Explanation**
        utilize_null_portions=True
            **Explanation**

        Returns
        -------
        _map_values : List[Union[int, float]]
            **Explanation**

        Raises
        ------
        ValueError:
            If dimension cannot transform a uniform 0-1 value

        """
        vs = None
        if self.value_set is not None:
            vs = self.value_set
        else:
            if self.lb is not None and self.ub is not None:
                vs = list(range(self.lb, self.ub + 1))

        if vs is not None:
            return _map_values(
                x, vs, self.portion_null if utilize_null_portions else None
            )
        raise ValueError(
            "Unbounded Int dimension cannot transform a uniform 0-1 value"
        )

    def validate(self, input_value, specified_input: bool) -> None:
        """
        TODO Explain the Function

        Arguments
        ---------
        self
            **Explanation**
        input_value
            **Explanation**
        specified_input : bool
            **Explanation**

        Raises
        ------
        ValueError:
            If input value, lower bound and upper bound
            are out of range/not in set

        """
        super().validate(input_value, specified_input)
        if input_value is not None:
            if self.lb is not None and input_value < self.lb:
                raise ValueError(
                    f"Invalid value, the value {input_value} is lower than "
                    f"the lower bound {self.lb}"
                )
            if self.ub is not None and input_value > self.ub:
                raise ValueError(
                    f"Invalid value, the value {input_value} is greater than "
                    f"the upper bound {self.ub}"
                )
            if (
                self.value_set is not None
                and input_value not in self.value_set
            ):
                raise ValueError(
                    f"Invalid value, the value {input_value} is not in the "
                    f"value set {self.value_set}"
                )

    def acceptable_types(self):
        """
        TODO Explain the Function
        """
        return (int,)


@dataclass
class Float(Dimension[float]):
    """
    TODO Explain the Function
    """

    lb: Optional[float] = None
    ub: Optional[float] = None
    value_set: Optional[List[float]] = None

    def convert_to_argument(self, input_value) -> T:
        """
        TODO Explain the Function

        Arguments
        ---------
        self : float
            **Explanation**
        input_value
            **Explanation**

        Returns
        -------
        Returns input value as a float

        """
        return float(input_value)

    def collapse_uniform(self, x, utilize_null_portions=True):
        """
        TODO Explain the Function

        Arguments
        ---------
        self : float
            **Explanation**
        x
            **Explanation**
        utilize_null_portions=True
            **Explanation**

        Returns
        -------
        _map_values : List[Union[int, float]]
            **Explanation**
        TODO **Explain one-line if/for return statement**

        Raises
        ------
        ValueError:
            If dimension cannot transform a uniform 0-1 value

        """
        if self.value_set is not None:
            return _map_values(
                x,
                self.value_set,
                self.portion_null if utilize_null_portions else None,
            )

        if self.lb is not None and self.ub is not None:
            r = self.ub - self.lb
            if self.portion_null is not None and utilize_null_portions:
                return [
                    (
                        self.lb
                        + r
                        * (
                            (xp - self.portion_null)
                            / (1.0 - self.portion_null)
                        )
                        if xp is not None and xp > self.portion_null
                        else None
                    )
                    for xp in x
                ]
            else:
                return [
                    self.lb + r * xp if xp is not None else None for xp in x
                ]
        raise ValueError(
            "Unbounded Float dimension cannot transform a uniform 0-1 value"
        )

    def validate(self, input_value, specified_input: bool) -> None:
        """
        TODO Explain the Function

        Arguments
        ---------
        self : float
            **Explanation**
        input_value
            **Explanation**
        specified_input : bool
            **Explanation**

        Raises
        ------
        ValueError:
            If input value, lower bound and upper bound
            are out of range/not in set

        """
        super().validate(input_value, specified_input)
        if input_value is not None:
            if self.lb is not None and input_value < self.lb:
                raise ValueError(
                    f"Invalid value, the value {input_value} is lower than "
                    f"the lower bound {self.lb}"
                )
            if self.ub is not None and input_value > self.ub:
                raise ValueError(
                    f"Invalid value, the value {input_value} is greater than "
                    f"the upper bound {self.ub}"
                )
            if (
                self.value_set is not None
                and input_value not in self.value_set
            ):
                raise ValueError(
                    f"Invalid value, the value {input_value} is not in the "
                    f"value set {self.value_set}"
                )

    def acceptable_types(self):
        """
        TODO Explain the Function

        """
        return (float,)


@dataclass
class CategoryValue:
    """
    TODO Explain Class
    """

    value: str
    label: Optional[str] = None


@dataclass
class Text(Dimension[str]):
    """
    TODO Explain Class
    """

    length_limit: Optional[int] = None
    value_set: Optional[List[Union[CategoryValue, str]]] = None

    def convert_to_argument(self, input_value) -> T:
        """
        TODO Explain the Function

        Arguments
        ---------
        self : str
            **Explanation**
        input_value
            **Explanation**

        Returns
        -------
        Returns input value as a string

        """
        return str(input_value)

    def collapse_uniform(self, x, utilize_null_portions=True):
        """
        TODO Explain the Function

        Arguments
        ---------
        self : str
            **Explanation**
        x
            **Explanation**

        Returns
        -------
        _map_values : List[Union[int, float]]
            **Explanation**

        Raises
        ------
        ValueError:
            If dimension cannot transform a uniform 0-1 value

        """
        if self.value_set is not None:
            return _map_values(
                x, {v.value for v in self.value_set}, self.portion_null
            )
        raise ValueError(
            "Unbounded Text dimension cannot transform a uniform 0-1 value"
        )

    def validate(self, input_value, specified_input: bool) -> None:
        """
        TODO Explain the Function

        Arguments
        ---------
        self : str
            **Explanation**
        input_value
            **Explanation**
        specified_input : bool
            **Explanation**

        Raises
        ------
        ValueError:
            If input value is not in the value set

        """
        super().validate(input_value, specified_input)
        if input_value is not None:
            if (
                self.value_set is not None
                and input_value not in self.value_set
            ):
                raise ValueError(
                    f"Invalid value, the value {input_value} is not in the "
                    f"value set {self.value_set}"
                )

    def acceptable_types(self):
        """
        TODO Explain the Function
        """
        return (str,)


@dataclass
class Variant(Dimension):
    """
    TODO Explain Class
    """

    options: Optional[List[Dimension]] = None

    @property
    def children(self):
        """
        TODO Explain the Function

        Returns
        -------
        **Explanation**

        """
        return self.options

    def convert_to_argument(self, input_value):
        """
        TODO Explain the Function

        Arguments
        ---------
        self : Variant
            **Explanation**
        input_value
            **Explanation**

        Returns
        -------
        TODO **Explanation**

        """
        option = self.options[input_value.option_index]
        return option.convert_to_argument(input_value.content)

    def collapse_uniform(self, x, utilize_null_portions=True):
        """
        TODO Explain the Function

        Arguments
        ---------
        self : Variant
            **Explanation**
        x
            **Explanation**
        utilize_null_portions=True
            **Explanation**

        Returns
        -------
        _map_values : List[Union[int, float]]
            **Explanation**

        Raises
        ------
        ValueError:
            If dimension cannot transform a uniform 0-1 value

        """
        return _map_values(
            x,
            [i for i in range(len(self.options))],
            self.portion_null if utilize_null_portions else None,
        )

    def only_supports_spec_structure(self) -> bool:
        """
        TODO Explain the Function
        """
        return False

    def has_child_dimensions(self) -> bool:
        """
        TODO Explain the Function
        """
        return True

    def count_children_dimensions(self) -> int:
        """
        TODO Explain the Function
        """
        return sum(
            [
                (
                    c.count_children_dimensions()
                    if c.has_child_dimensions()
                    else 1
                )
                for c in self.options
            ]
        )

    def validate(self, input_value, specified_input: bool) -> None:
        """
        TODO Explain the Function

        Arguments
        ---------
        self : Variant
            **Explanation**
        input_value
            **Explanation**
        specified_input : bool
            **Explanation**
        """
        super().validate(input_value, specified_input)
        if input_value is not None:
            for dim in self.options:
                if isinstance(input_value, dim.acceptable_types()):
                    value = input_value.content
                    dim.validate(value, specified_input)

    def acceptable_types(self):
        """
        TODO Explain the Function

        Arguments
        ---------
        self : Variant
            **Explanation**

        Returns
        -------
        **Explanation**
        """
        at = []
        for option in self.options:
            at += option.acceptable_types()
        return tuple(at)


@dataclass
class Listing(Dimension[List]):
    """
    TODO Explain Class
    """

    element_type: Optional[Dimension] = None
    cardinality_lb: Optional[int] = None
    cardinality_ub: Optional[int] = None


@dataclass
class Composite(Dimension):
    """
    TODO Explain Class
    """

    class_name: Optional[str] = ""
    children: Optional[List[Dimension]] = None
    type_class: Optional[Type] = None

    def convert_to_argument(self, input_value):
        """
        TODO Explain the Function

        Arguments
        ---------
        self : Composite
            **Explanation**
        input_value
            **Explanation**

        Returns
        -------
        Returns input value as a TODO**Explanation**

        """
        args = convert_values_from_dict(self.children, input_value)
        return self.type_class(**args)

    def collapse_uniform(self, x, utilize_null_portions=True):
        """
        TODO Explain the Function

        Arguments
        ---------
        self : Composite
            **Explanation**
        x
            **Explanation**
        utilize_null_portions=True
            **Explanation**

        Returns
        -------
        _map_values : List[Union[int, float]]
            **Explanation**

        """
        return _map_values(
            x,
            [
                1,
            ],
            self.portion_null if utilize_null_portions else None,
        )

    def only_supports_spec_structure(self) -> bool:
        """
        TODO Explain the Function
        """
        return not self.nullable

    def has_child_dimensions(self) -> bool:
        """
        TODO Explain the Function
        """
        return True

    def count_children_dimensions(self) -> int:
        """
        TODO Explain the Function
        """
        return sum(
            [
                (
                    c.count_children_dimensions()
                    if c.has_child_dimensions()
                    else 1
                )
                for c in self.children
            ]
        )

    def validate(self, input_value, specified_input: bool) -> None:
        """
        TODO Explain the Function

        Arguments
        ---------
        self : Composite
            **Explanation**
        input_value
            **Explanation**
        specified_input : bool
            **Explanation**

        """
        super().validate(input_value, specified_input)
        if input_value is not None:
            for dim in self.children:
                specified_child_input = False
                if hasattr(input_value, dim.id):
                    value = getattr(input_value, dim.id)
                    specified_child_input = True
                else:
                    value = None
                dim.validate(value, specified_child_input)

    def acceptable_types(self):
        """
        TODO Explain the Function
        """
        return (self.type_class,)
