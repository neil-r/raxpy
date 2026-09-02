"""
Unit tests for the code-intorspection of functions Union parameters
"""

from typing import Annotated, Union
from dataclasses import dataclass
import sys
import pytest

import raxpy
import raxpy.annotations.function_spec as fs
import raxpy.spaces as s


@dataclass
class _CustomCls1:
    """
    Dataclass used for unit testing
    """

    x1: Annotated[float, raxpy.Float(lb=0.0, ub=1.0)]
    x2: Annotated[float, raxpy.Float(lb=0.0, ub=1.0)]

    def execute(self):
        """
        demonstration method that does nothing
        """


@dataclass
class _CustomCls2:
    """
    Dataclass used for unit testing
    """

    x1: Annotated[float, raxpy.Float(lb=0.0, ub=2.0)]
    x2: Annotated[float, raxpy.Float(lb=0.0, ub=2.0)]

    def execute(self):
        """
        demonstration method that does nothing
        """


@dataclass
class _CustomCls3:
    """
    Dataclass used for unit testing
    """

    x1: Annotated[float, raxpy.Float(lb=0.0, ub=2.0)]
    x2: Annotated[float, raxpy.Float(lb=0.0, ub=2.0)]
    x3: Annotated[float, raxpy.Float(lb=0.0, ub=3.0)]

    def execute(self):
        """
        demonstration method that does nothing
        """


@pytest.mark.skipif(
    sys.version_info < (3, 10),
    reason="requires Python 3.10 or higher to use the new Union '|' syntax",
)
def test_union_choice_spec_param_func():
    """
    Tests the union parameters are correctly converted to an
    input space dimensions.

    Asserts
    -------
        The appropriate  dimension with children dimensions
        are derived
    """

    def f(x1: _CustomCls1 | _CustomCls2 | _CustomCls3):
        """
        Function used for unit testing
        """
        x1.execute()

    input_space = fs.extract_input_space(f)
    assert input_space is not None
    assert input_space.dimensions is not None
    assert len(input_space.dimensions) == 1
    dim = input_space.dimensions[0]
    assert isinstance(dim, s.Variant)
    assert len(dim.children) == 3
    assert dim.nullable is False

    def f2(x1: _CustomCls1 | _CustomCls2 | _CustomCls3 | None):
        """
        Function used for unit testing
        """
        x1.execute()

    input_space2 = fs.extract_input_space(f2)
    assert input_space2 is not None
    assert input_space2.dimensions is not None
    assert len(input_space2.dimensions) == 1
    dim2 = input_space2.dimensions[0]
    assert isinstance(dim2, s.Variant)
    assert len(dim2.children) == 3
    assert dim2.nullable is True


def test_union_choice_spec_param_func_long():
    """
    Tests the union parameters, typing.Union, are correctly
    converted to an input space dimensions.

    Asserts
    -------
        The appropriate dimension with children dimensions
        are derived
    """

    def f(x1: Union[_CustomCls1, _CustomCls2, _CustomCls3]):
        """
        Function to support unit testing.

        Arguments
        ---------
        x1 : CustomCls1 | CustomCls2 | CustomCls3
            **Explanation**
        """
        x1.execute()

    input_space = fs.extract_input_space(f)
    assert input_space is not None
    assert input_space.dimensions is not None
    assert len(input_space.dimensions) == 1
    dim = input_space.dimensions[0]
    assert isinstance(dim, s.Variant)
    assert len(dim.children) == 3
    assert dim.nullable is False


def test_shared_nested_factors():
    """
    Tests to ensure ids assigned to shared nested factors are shared across
    the union parameter options
    """

    @dataclass
    class _CustomClsN1:
        """
        Dataclass used for unit testing
        """

        x1: Annotated[float, raxpy.Float(id="x1n", lb=0.0, ub=1.0)]

    @dataclass
    class _CustomClsN2:
        """
        Dataclass used for unit testing
        """

        x1: Annotated[float, raxpy.Float(id="x1n", lb=0.0, ub=1.0)]

    def f(x1: Union[_CustomClsN1, _CustomClsN2]):
        """
        Function to support unit testing.

        Arguments
        ---------
        x1 : CustomClsN1 | CustomClsN2
        """
        pass

    input_space = fs.extract_input_space(f)
    assert input_space is not None
    assert input_space.dimensions is not None
    assert len(input_space.dimensions) == 1
    dim = input_space.dimensions[0]
    assert isinstance(dim, s.Variant)
    assert dim.id == "x1"
    assert len(dim.children) == 2
    assert dim.nullable is False

    assert dim.children[0].children[0].id == "x1n"
    assert dim.children[1].children[0].id == "x1n"
