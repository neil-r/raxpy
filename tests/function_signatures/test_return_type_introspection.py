""" TODO Explain Module """

from typing import Annotated, Optional, Union
from dataclasses import dataclass

import raxpy.annotations.function_spec as fs
import raxpy.spaces.dimensions as d
import raxpy


def test_no_return_type_func():
    """
    TODO Explain the Function

    Asserts
    -------
    **Explanation**

    """
    def f():
        """
        TODO Explain the Function
        """
        pass

    output_space = fs.extract_output_space(f)
    assert output_space is not None
    assert output_space.dimensions is not None
    assert len(output_space.dimensions) == 0


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
    label=None,
):
    """
    TODO Explain the Function
    """
    assert type(d) == t
    assert d.id == id
    assert d.default_value == default_value
    assert d.lb == lb
    assert d.ub == ub
    assert d.value_set == value_set
    assert d.nullable == nullable
    assert d.tags == tags
    assert d.label == label


def test_unannotated_single_value_return_type_func():
    """
    TODO Explain the Function
    """
    def f() -> float:
        """
        TODO Explain the Function
        """
        return 0.0
    output_space = fs.extract_output_space(f)
    assert output_space is not None
    assert output_space.dimensions is not None
    assert len(output_space.dimensions) == 1
    assert_parameters(
        output_space.dimensions[0], d.Float, fs.ID_ROOT_RETURN,
        None, None, None, None, nullable=False
    )

    def f2() -> Optional[float]:
        """
        TODO Explain the Function
        """
        return 0.0
    output_space2 = fs.extract_output_space(f2)
    assert output_space2 is not None
    assert output_space2.dimensions is not None
    assert len(output_space2.dimensions) == 1
    assert_parameters(
        output_space2.dimensions[0], d.Float, fs.ID_ROOT_RETURN,
        None, None, None, None, nullable=True
    )

    def f3() -> Union[int,None]:
        """
        TODO Explain the Function
        """
        return 0
    output_space3 = fs.extract_output_space(f3)
    assert output_space3 is not None
    assert output_space3.dimensions is not None
    assert len(output_space3.dimensions) == 1
    assert_parameters(
        output_space3.dimensions[0], d.Int, fs.ID_ROOT_RETURN,
        None, None, None, None, nullable=True
    )

def test_annotated_single_value_return_type_func():
    """
    TODO Explain the Function
    """
    def f() -> Annotated[float,raxpy.Float(label="Zero")]:
        """
        TODO Explain the Function
        """
        return 0.0
    output_space = fs.extract_output_space(f)
    assert output_space is not None
    assert output_space.dimensions is not None
    assert len(output_space.dimensions) == 1
    assert_parameters(
        output_space.dimensions[0], d.Float, fs.ID_ROOT_RETURN,
        None, None, None, None, nullable=False, label="Zero"
    )

    def f2() -> (Annotated[Optional[float],
                 raxpy.Float(label="ZeroOrNone", tags=[raxpy.tags.MAXIMIZE])]):
        """
        TODO Explain the Function
        """
        return 0.0
    output_space2 = fs.extract_output_space(f2)
    assert output_space2 is not None
    assert output_space2.dimensions is not None
    assert len(output_space2.dimensions) == 1
    assert_parameters(
        output_space2.dimensions[0], d.Float, fs.ID_ROOT_RETURN,
        None, None, None, None, nullable=True, tags=[raxpy.tags.MAXIMIZE],
        label="ZeroOrNone"
    )

    def f3() -> Annotated[Union[int,None], raxpy.Integer(label="Int",
                                                         lb=0,ub=4)]:
        return 0
    output_space3 = fs.extract_output_space(f3)
    assert output_space3 is not None
    assert output_space3.dimensions is not None
    assert len(output_space3.dimensions) == 1
    assert_parameters(
        output_space3.dimensions[0], d.Int, fs.ID_ROOT_RETURN,
        None, 0, 4, None, nullable=True, label="Int"
    )
