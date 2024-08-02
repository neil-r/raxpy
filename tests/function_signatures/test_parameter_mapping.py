""" TODO Explain Module """

from typing import Annotated, Optional
from dataclasses import dataclass

import raxpy.annotations.function_spec as fs
import raxpy.spaces.dimensions as d
import raxpy


def test_no_param_func():
    """
    TODO Explain the Function

    Asserts
    -------
    **Explanation**

    """

    def f():
        """
        TODO Explain the Function **Not Implemented
        """
        pass

    input_space = fs.extract_input_space(f)
    assert input_space is not None
    assert input_space.dimensions is not None
    assert len(input_space.dimensions) == 0


def assert_parameters(
    d,
    t,
    id,
    default_value,
    lb,
    ub,
    value_set,
    nullable=False,
    tags=None,
    specified_default=False,
):
    """
    TODO Explain the Function

    Arguments
    ---------
    d : 
        **Explanation**
    t : 
        **Explanation**
    id : 
        **Explanation**
    default_value : 
        **Explanation**
    lb : 
        **Explanation**
    ub : 
        **Explanation**
    value_set : 
        **Explanation**
    nullable=False
        **Explanation**
    tags=None
        **Explanation**
    specified_default=False
        **Explanation**

    Asserts
    -------
    **Explanation**

    """
    assert type(d) == t
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
    TODO Explain the Function

    Asserts
    -------
    **Explanation**

    """

    def f(x: Annotated[int, raxpy.Integer(lb=0, ub=5)] = 2):
        """
        TODO Explain the Function

        Arguments
        ---------
        x: Annotated[int, raxpy.Integer(lb=0, ub=5)] = 2
            **Explanation**
        
        Returns
        -------
        **Explanation**

        """
        return x * 2

    input_space = fs.extract_input_space(f)
    assert input_space is not None
    assert input_space.dimensions is not None
    assert len(input_space.dimensions) == 1
    assert_parameters(
        input_space.dimensions[0],
        d.Int,
        "x",
        2,
        0,
        5,
        None,
        specified_default=True,
    )


def test_mixed_spec_param_func():
    """
    TODO Explain the Function

    Asserts
    -------
    **Explanation**

    """

    def f(
        x1: Annotated[int, raxpy.Integer(lb=0, ub=5)],
        x2: Annotated[float, raxpy.Float(lb=1.7, ub=3.3)],
        x3: Annotated[int, raxpy.Integer(ub=5)],
        x4: Annotated[
            Optional[int], raxpy.Integer(value_set={1, 2, 4})
        ] = None,
        x5: int = 3,
        x6: Optional[float] = None,
    ):
        """
        TODO Explain the Function

        Arguments
        ---------
        x1 : Annotated[int]
            **Explanation**
        x2 : Annotated[float]
            **Explanation**
        x3 : Annotated[int]
            **Explanation**
        x4 : Annotated[Optional[int]] = None
            **Explanation**
        x5 : int = 3
            **Explanation**
        x6 : Optional[float] = None
            **Explanation**

        Returns
        -------
        **Explanation**
        """
        return x1 * 2

    input_space = fs.extract_input_space(f)
    assert input_space is not None
    assert input_space.dimensions is not None
    assert len(input_space.dimensions) == 6
    assert_parameters(input_space.dimensions[0], d.Int, "x1", None, 0, 5, None)
    assert_parameters(
        input_space.dimensions[1], d.Float, "x2", None, 1.7, 3.3, None
    )
    assert_parameters(
        input_space.dimensions[2], d.Int, "x3", None, None, 5, None
    )
    assert_parameters(
        input_space.dimensions[3],
        d.Int,
        "x4",
        None,
        None,
        None,
        value_set={1, 2, 4},
        nullable=True,
        specified_default=True,
    )
    assert_parameters(
        input_space.dimensions[4],
        d.Int,
        "x5",
        3,
        None,
        None,
        None,
        nullable=False,
        specified_default=True,
    )
    assert_parameters(
        input_space.dimensions[5],
        d.Float,
        "x6",
        None,
        None,
        None,
        None,
        nullable=True,
        specified_default=True,
    )


def test_blank_object_spec_param_func():
    """
    TODO Explain the Function

    Asserts
    -------
    **Explain Asserts**

    """

    @dataclass
    class CustomObject:
        """
        TODO Explain Class
        """
        pass

    def f(obj1: CustomObject):
        pass

    input_space = fs.extract_input_space(f)
    assert input_space is not None
    assert input_space.dimensions is not None
    assert len(input_space.dimensions) == 1


def test_complex_object_spec_param():
    """
    TODO Explain the Function

    Asserts
    -------
    **Explain Asserts**

    """

    @dataclass
    class ChildCustomObject:
        """
        TODO Explain Class
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
        TODO Explain Class
        """

        ao1: ChildCustomObject
        ao2: Optional[ChildCustomObject] = None
        ai2: Optional[int] = None

    def f(obj1: CustomObject):
        """
        TODO Explain the Function **Not Implemented**

        Arguments
        ---------
        obj1 : CustomObject
            **Explanation**
        """
        pass

    input_space = fs.extract_input_space(f)
    assert input_space is not None
    assert input_space.dimensions is not None
    assert len(input_space.dimensions) == 1

    actual_composite_dim = input_space.dimensions[0]
    assert isinstance(actual_composite_dim, d.Composite)
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
