""" 
    Unit tests considering the introspection of a function's
    parameters defining an input space.
"""

from typing import Annotated, Optional
from dataclasses import dataclass

import raxpy.annotations.function_spec as fs
import raxpy.spaces as s
import raxpy


def test_no_param_func():
    """
    Tests the introspection of a function that
    has no parameters.

    Asserts
    -------
        InputSpace has no dimensions

    """

    def f():
        """
        function spec to support unit testing
        """
        pass

    input_space = fs.extract_input_space(f)
    assert input_space is not None
    assert input_space.dimensions is not None
    assert len(input_space.dimensions) == 0


def assert_parameters(
    d: s.Dimension,
    t: type,
    id: str,
    default_value,
    lb,
    ub,
    value_set,
    nullable: bool = False,
    tags=None,
    specified_default=False,
):
    """
    Helper function to support the assert dimensions
    have the proper attributes.

    Arguments
    ---------
    d : s.Dimension
        the dimension to analyze
    t
        the type of dimension to check if d is an instance
    id
        the id of dimension to check if d has this id
    default_value
        to check if d has this default_value
    lb
        to check if d has this lower bound
    ub
        to check if d has this upper bound
    value_set
        to check if d has this value set specifed
    nullable=False
        to check if d has this nullable flag
    tags=None
        to check if d has these tags
    specified_default=False
        to check if d has specified a default value

    Asserts
    -------
        d has the attributes provided

    """
    assert isinstance(d, t)
    assert d.id == id
    assert d.default_value == default_value
    assert d.lb == lb
    assert d.ub == ub
    assert d.value_set == value_set
    assert d.nullable == nullable
    assert d.tags == tags
    assert d.specified_default == specified_default


def test_single_param_func():
    """
    Tests the introspection of a function that
    has a single integer parameter.

    Asserts
    -------
        the input space includes a single dimension
        corrosponding to the single parameter
    """

    def f(x: Annotated[int, raxpy.Integer(lb=0, ub=5)] = 2):
        """
        Function supporting the unit test.
        """
        return x * 2

    input_space = fs.extract_input_space(f)
    assert input_space is not None
    assert input_space.dimensions is not None
    assert len(input_space.dimensions) == 1
    assert_parameters(
        input_space.dimensions[0],
        s.Int,
        "x",
        2,
        0,
        5,
        None,
        specified_default=True,
    )


def test_mixed_spec_param_func():
    """
    Tests the introspection of a function that
    has a mutlple number-based parameters.

    Asserts
    -------
        the input space includes a appropriate dimensions
        corrosponding to the parameters
    """

    def f(
        x1: Annotated[int, raxpy.Integer(lb=0, ub=5)],
        _x2: Annotated[float, raxpy.Float(lb=1.7, ub=3.3)],
        _x3: Annotated[int, raxpy.Integer(ub=5)],
        _x4: Annotated[
            Optional[int], raxpy.Integer(value_set={1, 2, 4})
        ] = None,
        _x5: int = 3,
        _x6: Optional[float] = None,
        _x7: bool = True,
    ):
        """
        Function supporting unit testing.
        """
        return x1 * 2

    input_space = fs.extract_input_space(f)
    assert input_space is not None
    assert input_space.dimensions is not None
    assert len(input_space.dimensions) == 7
    assert_parameters(input_space.dimensions[0], s.Int, "x1", None, 0, 5, None)
    assert_parameters(
        input_space.dimensions[1], s.Float, "_x2", None, 1.7, 3.3, None
    )
    assert_parameters(
        input_space.dimensions[2], s.Int, "_x3", None, None, 5, None
    )
    assert_parameters(
        input_space.dimensions[3],
        s.Int,
        "_x4",
        None,
        None,
        None,
        value_set={1, 2, 4},
        nullable=True,
        specified_default=True,
    )
    assert_parameters(
        input_space.dimensions[4],
        s.Int,
        "_x5",
        3,
        None,
        None,
        None,
        nullable=False,
        specified_default=True,
    )
    assert_parameters(
        input_space.dimensions[5],
        s.Float,
        "_x6",
        None,
        None,
        None,
        None,
        nullable=True,
        specified_default=True,
    )
    assert_parameters(
        input_space.dimensions[6],
        s.Bool,
        "_x7",
        True,
        0,
        1,
        None,
        nullable=False,
        specified_default=True,
    )


def test_blank_object_spec_param_func():
    """
    Tests the introspection of a function that
    has a parameter based on a dataclass with
    no attributes.

    Asserts
    -------
        the input space includes a appropriate dimensions
        corrosponding to the dataclass parameter.
    """

    @dataclass
    class CustomObject:
        """
        Dataclass supporting the unit test
        """

    def f(_obj1: CustomObject):
        """
        Function supporting the unit test
        """

    input_space = fs.extract_input_space(f)
    assert input_space is not None
    assert input_space.dimensions is not None
    assert len(input_space.dimensions) == 1


def test_complex_object_spec_param():
    """
    Tests the introspection of a function that
    has a parameter based on a dataclass.

    Asserts
    -------
        the input space includes a appropriate dimensions
        corrosponding to the dataclass parameter.
    """

    @dataclass
    class ChildCustomObject:
        """
        Dataclass supporting unit test
        """

        caf1: float
        caf2: Optional[float]
        cas1: str
        cas2: Annotated[
            str, raxpy.Categorical(value_set={"one", "two", "three"})
        ]

    @dataclass
    class CustomObject:
        """
        Another dataclass supporting unit test
        """

        ao1: ChildCustomObject
        ao2: Optional[ChildCustomObject] = None
        ai2: Optional[int] = None

    def f(_obj1: CustomObject):
        """
        Function supporting unit test introspection
        """

    input_space = fs.extract_input_space(f)
    assert input_space is not None
    assert input_space.dimensions is not None
    assert len(input_space.dimensions) == 1

    actual_composite_dim = input_space.dimensions[0]
    assert isinstance(actual_composite_dim, s.Composite)
    assert len(actual_composite_dim.children) == 3

    # test the conversion of a dictionary representation of the inputs
    co: CustomObject = input_space.dimensions[0].convert_to_argument(
        {
            "ao1": {
                "caf1": 1.0,
                "caf2": 0.5,
                "cas1": "Hello world",
                "cas2": "one",
            }
        }
    )

    assert isinstance(co, CustomObject)
    assert co.ao2 is None
    assert co.ai2 is None
    assert co.ao1.caf1 == 1.0
