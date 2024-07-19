from typing import Optional, List, Tuple, Union, Generic, TypeVar, Set, Type, Dict, Any

import numpy as np

from dataclasses import dataclass


T = TypeVar("T")


def _map_values(x, value_set, portion_null):
    value_count = len(value_set)
    boundary_size = 1.0 / value_count

    if portion_null is not None:
        return [
            (
                value_set[
                    int(((xp - portion_null) / (1.0 - portion_null)) // boundary_size)
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


def convert_values_from_dict(dimensions, input_value: Dict[str, Any]):
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
    id: str = ""
    local_id: str = ""
    nullable: bool = False
    specified_default: bool = False
    label: Optional[str] = None
    default_value: Optional[T] = None
    tags: Optional[List[str]] = None
    portion_null: Optional[float] = None

    def __post_init__(self):
        if self.id == "":
            self.id = self.local_id
        if self.local_id == "":
            self.local_id = self.id
        if self.id == "":
            raise ValueError("Invalid identifier for dimension")

    def has_child_dimensions(self) -> bool:
        return False

    def only_supports_spec_structure(self) -> bool:
        return False

    def collapse_uniform(self, x, utilize_null_portitions=True):
        raise NotImplementedError(
            "Abstract method, subclass should implement this method"
        )

    def has_tag(self, tag: str) -> bool:
        return self.tags is not None and tag in self.tags

    def convert_to_argument(self, input_value) -> T:
        raise NotImplementedError(
            "Abstract method, subclass should implement this method"
        )

    def acceptable_types(self) -> Tuple[Type]:
        raise NotImplementedError(
            "Abstract method, subclass should implement this method"
        )

    def validate(self, input_value, specified_input: bool):
        if input_value is None:
            if self.nullable:
                return

            if specified_input:
                raise ValueError(
                    f"Invalid value, dimension '{self.id}' should not be null"
                )
            if not self.specified_default:
                raise ValueError(
                    f"Invalid value, dimension '{self.id}' should be specified, no default provided"
                )

            return

        if not isinstance(input_value, self.acceptable_types()):
            raise ValueError(
                f"Invalid value type, dimension '{self.id}' should be a {T}"
            )


@dataclass
class Int(Dimension[int]):
    lb: Optional[int] = None
    ub: Optional[int] = None
    value_set: Optional[List[int]] = None

    def convert_to_argument(self, input_value) -> T:
        return int(input_value)

    def collapse_uniform(self, x, utilize_null_portitions=True):
        vs = None
        if self.value_set is not None:
            vs = self.value_set
        else:
            if self.lb is not None and self.ub is not None:
                vs = list(range(self.lb, self.ub + 1))

        if vs is not None:
            return _map_values(
                x, vs, self.portion_null if utilize_null_portitions else None
            )
        raise ValueError("Unbounded Int dimension cannot transform a uniform 0-1 value")

    def validate(self, input_value, specified_input: bool):
        super().validate(input_value, specified_input)
        if input_value is not None:
            if self.lb is not None and input_value < self.lb:
                raise ValueError(
                    f"Invalid value, the value {input_value} is lower than the lower bound {self.lb}"
                )
            if self.ub is not None and input_value > self.ub:
                raise ValueError(
                    f"Invalid value, the value {input_value} is greater than the upper bound {self.ub}"
                )
            if self.value_set is not None and input_value not in self.value_set:
                raise ValueError(
                    f"Invalid value, the value {input_value} is not in the value set {self.value_set}"
                )

    def acceptable_types(self):
        return (int,)


@dataclass
class Float(Dimension[float]):
    lb: Optional[float] = None
    ub: Optional[float] = None
    value_set: Optional[List[float]] = None

    def convert_to_argument(self, input_value) -> T:
        return float(input_value)

    def collapse_uniform(self, x, utilize_null_portitions=True):
        if self.value_set is not None:
            return _map_values(
                x,
                self.value_set,
                self.portion_null if utilize_null_portitions else None,
            )

        if self.lb is not None and self.ub is not None:
            r = self.ub - self.lb
            if self.portion_null is not None and utilize_null_portitions:
                return [
                    (
                        self.lb
                        + r * ((xp - self.portion_null) / (1.0 - self.portion_null))
                        if xp is not None and xp > self.portion_null
                        else None
                    )
                    for xp in x
                ]
            else:
                return [self.lb + r * xp if xp is not None else None for xp in x]
        raise ValueError(
            "Unbounded Float dimension cannot transform a uniform 0-1 value"
        )

    def validate(self, input_value, specified_input: bool):
        super().validate(input_value, specified_input)
        if input_value is not None:
            if self.lb is not None and input_value < self.lb:
                raise ValueError(
                    f"Invalid value, the value {input_value} is lower than the lower bound {self.lb}"
                )
            if self.ub is not None and input_value > self.ub:
                raise ValueError(
                    f"Invalid value, the value {input_value} is greater than the upper bound {self.ub}"
                )
            if self.value_set is not None and input_value not in self.value_set:
                raise ValueError(
                    f"Invalid value, the value {input_value} is not in the value set {self.value_set}"
                )

    def acceptable_types(self):
        return (float,)


@dataclass
class CategoryValue:
    value: str


@dataclass
class Text(Dimension[str]):
    length_limit: Optional[int] = None
    value_set: Optional[List[Union[CategoryValue, str]]] = None

    def convert_to_argument(self, input_value) -> T:
        return str(input_value)

    def collapse_uniform(self, x):
        if self.value_set is not None:
            return _map_values(x, {v.value for v in self.value_set}, self.portion_null)
        raise ValueError(
            "Unbounded Text dimension cannot transform a uniform 0-1 value"
        )

    def validate(self, input_value, specified_input: bool):
        super().validate(input_value, specified_input)
        if input_value is not None:
            if self.value_set is not None and input_value not in self.value_set:
                raise ValueError(
                    f"Invalid value, the value {input_value} is not in the value set {self.value_set}"
                )

    def acceptable_types(self):
        return (str,)


@dataclass
class Variant(Dimension):
    options: Optional[List[Dimension]] = None

    @property
    def children(self):
        return self.options

    def convert_to_argument(self, input_value):
        option = self.options[input_value.option_index]
        return option.convert_to_argument(input_value.content)

    def collapse_uniform(self, x, utilize_null_portitions=True):
        return _map_values(
            x,
            [i for i in range(len(self.options))],
            self.portion_null if utilize_null_portitions else None,
        )

    def only_supports_spec_structure(self) -> bool:
        return False

    def has_child_dimensions(self) -> bool:
        return True

    def count_children_dimensions(self) -> int:
        return sum(
            [
                c.count_children_dimensions() if c.has_child_dimensions() else 1
                for c in self.options
            ]
        )

    def validate(self, input_value, specified_input: bool):
        super().validate(input_value, specified_input)
        if input_value is not None:
            for dim in self.options:
                if isinstance(input_value, dim.acceptable_types()):
                    value = input_value.content
                    dim.validate(value, specified_input)

    def acceptable_types(self):
        at = []
        for option in self.options:
            at += option.acceptable_types()
        return tuple(at)


@dataclass
class Listing(Dimension[List]):
    element_type: Optional[Dimension] = None
    cardinality_lb: Optional[int] = None
    cardinality_ub: Optional[int] = None


@dataclass
class Composite(Dimension):
    class_name: Optional[str] = ""
    children: Optional[List[Dimension]] = None
    type_class: Optional[Type] = None

    def convert_to_argument(self, input_value):
        args = convert_values_from_dict(self.children, input_value)
        return self.type_class(**args)

    def collapse_uniform(self, x, utilize_null_portitions):
        return _map_values(
            x,
            [
                1,
            ],
            self.portion_null if utilize_null_portitions else None,
        )

    def only_supports_spec_structure(self) -> bool:
        return not self.nullable

    def has_child_dimensions(self) -> bool:
        return True

    def count_children_dimensions(self) -> int:
        return sum(
            [
                c.count_children_dimensions() if c.has_child_dimensions() else 1
                for c in self.children
            ]
        )

    def validate(self, input_value, specified_input: bool):
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
        return (self.type_class,)
