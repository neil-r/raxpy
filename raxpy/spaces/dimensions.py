""" 
    This modules provides data structures to support the
    specification of a space's dimensions (or factors).
"""

from dataclasses import dataclass
from typing import (
    Any,
    Dict,
    Generic,
    List,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
)

import numpy as np


T = TypeVar("T")


def _map_values(x, value_set, portion_null) -> List[Union[int, float]]:
    """
    Helper function to map a 0-1 array of values to a
    discrete set of values and a porition of the values
    specified to be null. portion_null is used as a threshold
    for all values under this threshold as being mapped to np.nan.

    Arguments
    ---------
    x
        an array of 0-1 values
    value_set
        a list of values
    portion_null
        threhold of the 0-1 range to map to np.nan

    Returns
    -------
    List[Union[int, float]]
        the transformed array with values from the value_set
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
    Creates a dictionary with a key-value pair for
    each dimension in dimensions.
    If a dimension is not in the input_value list,
    then the default value attribute of the dimension
    is assigned as the value for the dimension.

    Arguments
    ---------
    dimensions
        The dimensions to consider
    input_value : Dict[str, Any]
        the dict values to consider

    Returns
    -------
    args : Dict
        a fully specified dict of key-values
    """
    args = {}

    for dim in dimensions:
        if dim.local_id in input_value:
            child_value = input_value[dim.local_id]
            if child_value is None:
                if dim.nullable:
                    args[dim.local_id] = None
                else:
                    # TODO reassess this logic, may want to raise error
                    args[dim.local_id] = dim.default_value
            else:
                args[dim.local_id] = dim.convert_to_argument(child_value)
        else:
            args[dim.local_id] = dim.default_value
    return args


@dataclass
class Dimension(Generic[T]):
    """
    Abstract class for a specification of a range of values
    within a space.
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
        Ensure id's of dimension are specified

        Raises
        ------
        ValueError:
            if id is not specified

        """
        if self.id == "":
            self.id = self.local_id
        if self.local_id == "":
            self.local_id = self.id
        if self.id == "":
            raise ValueError("Invalid identifier for dimension")

    def has_finite_values(self) -> bool:
        """
        Checks if a dimension represents a finte number of values

        Returns
        -------
            bool: True if the dimension represents a finte number of values
                False otherwise
        """
        return True

    def has_child_dimensions(self) -> bool:
        """
        Checks if the dimension has child dimensions.
        Child dimensions should be accessed with the children
        attribute.

        Returns
        -------
            bool: True if the dimension has child dimensions

        """
        return False

    def only_supports_spec_structure(self) -> bool:
        """
        Checks if the dimension is only for structure

        Returns
        -------
            bool: True if the dimension, itself and not considering children dimensions,
                does represents any variability
        """
        return False

    def collapse_uniform(self, x, utilize_null_portions=True):
        """
        Converts a 0-1 array x to decoded values.

        Arguments
        ---------
        self
            dimension
        x
            0-1 array of values
        utilize_null_portions=True
            to consider the null_portion attribute to assign null values

        Returns
        -------
            list of decoded values
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
            the dimension
        tag : str
            the tag to check if assigned to self

        Returns
        -------
            bool: Returns True if there is a valid tag and
               False if there is not a tag

        """
        return self.tags is not None and tag in self.tags

    def convert_to_argument(self, input_value) -> T:
        """
        Converts a decoded numpy based input_value to a
        form suitable for a Python function.

        Arguments
        ---------
        self
            **Explanation**
        input_value
            the decoded numpy value

        Returns
        -------
            T: a form of the numpy value suitable

        """
        raise NotImplementedError(
            "Abstract method, subclass should implement this method"
        )

    def acceptable_types(self) -> Tuple[Type]:
        """
        Returns a tuple of types acceptable for self

        Arguments
        ---------
        self
            the dimension

        Returns
        -------
            Tuple[Type]: a tuple of the acceptable types

        """
        raise NotImplementedError(
            "Abstract method, subclass should implement this method"
        )

    def validate(self, input_value, specified_input: bool):
        """
        Validates the input_value given self's
        specification of acceptable range of values.

        Arguments
        ---------
        self
            the dimension to consider
        input_value
            the value to check
        specified_input : bool
            flag to indicate whether to ensure not null as well

        Returns
        -------
            bool: True if valid

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
    A range of integer values.
    """

    lb: Optional[int] = None
    ub: Optional[int] = None
    value_set: Optional[List[int]] = None

    def convert_to_argument(self, input_value) -> T:
        """
        Implementation of abstract method. See `Dimension.convert_to_argument`.
        """
        return int(input_value)

    def collapse_uniform(
        self, x, utilize_null_portions=True
    ) -> List[Union[int, float]]:
        """
        Implementation of abstract method. See `Dimension.collapse_uniform`.

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
        Implementation of abstract method. See `Dimension.validate`.

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
        Implementation of abstract method. See `Dimension.acceptable_types`.
        """
        return (int,)


@dataclass
class Bool(Int):
    """
    A range of integer values.
    """

    def validate(self, input_value, specified_input: bool) -> None:
        """
        Implementation of abstract method. See `Dimension.validate`.

        Raises
        ------
        ValueError:
            If input value, lower bound and upper bound
            are out of range/not in set

        """
        Dimension.validate(self, input_value, specified_input)
        if input_value is not None:
            if not isinstance(input_value, bool):
                raise ValueError(
                    f"Invalid value, the value {input_value} is not the right type"
                )

    def convert_to_argument(self, input_value) -> T:
        """
        Implementation of abstract method. See `Dimension.convert_to_argument`.
        """
        return int(input_value) == 1

    def acceptable_types(self):
        """
        Implementation of abstract method. See `Dimension.acceptable_types`.
        """
        return (bool,)


@dataclass
class Float(Dimension[float]):
    """
    A range of float values.
    """

    lb: Optional[float] = None
    ub: Optional[float] = None
    value_set: Optional[List[float]] = None

    def convert_to_argument(self, input_value) -> T:
        """
        Implementation of abstract method. See `Dimension.convert_to_argument`.

        """
        return float(input_value)

    def collapse_uniform(self, x, utilize_null_portions=True):
        """
        Implementation of abstract method. See `Dimension.collapse_uniform`.

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
        Implementation of abstract method. See `Dimension.validate`.

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

    def has_finite_values(self):
        """
        Implementation of abstract method. See `Dimension.has_finite_values`.
        """
        return self.value_set is not None

    def acceptable_types(self):
        """
        Implementation of abstract method. See `Dimension.acceptable_types`.
        """
        return (float,)


@dataclass
class CategoryValue:
    """
    A string value and label for a discrete-category
    within a category dimension.
    """

    value: str
    label: Optional[str] = None


@dataclass
class Text(Dimension[str]):
    """
    A range of string values representing categories.
    """

    length_limit: Optional[int] = None
    value_set: Optional[List[Union[CategoryValue, str]]] = None

    def convert_to_argument(self, input_value) -> T:
        """
        Implementation of abstract method. See `Dimension.convert_to_argument`.
        """
        return str(input_value)

    def collapse_uniform(self, x, utilize_null_portions=True):
        """
        Implementation of abstract method. See `Dimension.collapse_uniform`.

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
        Implementation of abstract method. See `Dimension.validate`.

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
        Implementation of abstract method. See `Dimension.acceptable_types`.
        """
        return (str,)


@dataclass
class Variant(Dimension):
    """
    Represents a dimension that is specified as one of its options.
    """

    options: Optional[List[Dimension]] = None

    @property
    def children(self):
        """
        Lists the children dimensions of this dimension.

        Returns
        -------
            the optional children dimensions
        """
        return self.options

    def convert_to_argument(self, input_value):
        """
        Implementation of abstract method. See `Dimension.convert_to_argument`.
        """
        # find the children
        for option in self.options:
            if option.local_id in input_value:
                return option.convert_to_argument(input_value[option.local_id])

        if self.nullable:
            return None

        raise ValueError(
            f"non-nullable variant trying to convert input_value that does not have an option specified, : {input_value}"
        )

    def collapse_uniform(self, x, utilize_null_portions=True):
        """
        Implementation of abstract method. See `Dimension.collapse_uniform`.

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
        Implementation of abstract method. See `Dimension.only_supports_spec_structure`.
        """
        return False

    def has_child_dimensions(self) -> bool:
        """
        Implementation of abstract method. See `Dimension.has_child_dimensions`.
        """
        return True

    def count_children_dimensions(self) -> int:
        """
        Recursively counts the number of children dimensions

        Returns
        -------
            int: the number of children dimensions
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
        Implementation of abstract method. See `Dimension.validate`.
        """
        super().validate(input_value, specified_input)
        if input_value is not None:
            for dim in self.options:
                if isinstance(input_value, dim.acceptable_types()):
                    value = input_value.content
                    dim.validate(value, specified_input)

    def acceptable_types(self):
        """
        Implementation of abstract method. See `Dimension.acceptable_types`.
        """
        at = []
        for option in self.options:
            at += option.acceptable_types()
        return tuple(at)


@dataclass
class ListDim(Dimension[List]):
    """
    Represents a dimension that is specified as a range of lists.
    """

    element_type: Optional[Dimension] = None
    cardinality_lb: Optional[int] = None
    cardinality_ub: Optional[int] = None

    def convert_to_argument(self, input_value):
        """
        Implementation of abstract method. See `Dimension.convert_to_argument`.
        """
        # TODO implement
        raise NotImplementedError("Not implemented!")

    def only_supports_spec_structure(self) -> bool:
        """
        Implementation of abstract method. See `Dimension.only_supports_spec_structure`.
        """
        return False

    def has_child_dimensions(self) -> bool:
        """
        Implementation of abstract method. See `Dimension.has_child_dimensions`.
        """
        return True

    def collapse_uniform(self, x, utilize_null_portions=True):
        """
        Implementation of abstract method. See `Dimension.collapse_uniform`.
        """
        # TODO implement
        raise NotImplementedError("Not implemented!")

    def acceptable_types(self):
        """
        Implementation of abstract method. See `Dimension.acceptable_types`.
        """
        return (List, Tuple)


@dataclass
class Composite(Dimension):
    """
    Represents a set of dimensions that are specified as a group.
    """

    class_name: Optional[str] = ""
    children: Optional[List[Dimension]] = None
    type_class: Optional[Type] = None

    def convert_to_argument(self, input_value):
        """
        Implementation of abstract method. See `Dimension.convert_to_argument`.
        """
        args = convert_values_from_dict(self.children, input_value)
        return self.type_class(**args)

    def collapse_uniform(self, x, utilize_null_portions=True):
        """
        Implementation of abstract method. See `Dimension.collapse_uniform`.
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
        Implementation of abstract method. See `Dimension.only_supports_spec_structure`.
        """
        return not self.nullable

    def has_child_dimensions(self) -> bool:
        """
        Implementation of abstract method. See `Dimension.has_child_dimensions`.
        """
        return True

    def count_children_dimensions(self) -> int:
        """
        Recursively counts the number of children dimensions

        Returns
        -------
            int: the number of children dimensions
        """
        return sum(
            [
                (
                    1
                    if not c.has_child_dimensions()
                    else (
                        (0 if c.only_supports_spec_structure() else 1)
                        + (c.count_children_dimensions())
                    )
                )
                for c in self.children
            ]
        )

    def validate(self, input_value, specified_input: bool) -> None:
        """
        Implementation of abstract method. See `Dimension.validate`.
        """
        super().validate(input_value, specified_input)
        if input_value is not None:
            for dim in self.children:
                specified_child_input = False
                if hasattr(input_value, dim.local_id):
                    value = getattr(input_value, dim.local_id)
                    specified_child_input = True
                else:
                    value = None
                dim.validate(value, specified_child_input)

    def acceptable_types(self):
        """
        Implementation of abstract method. See `Dimension.acceptable_types`.
        """
        return (self.type_class,)
