""" TODO Explain Module """

from typing import Annotated, Optional, Union
from dataclasses import dataclass
import sys
import pytest

import raxpy
import raxpy.annotations.function_spec as fs
import raxpy.spaces.dimensions as d


@dataclass
class _CustomCls1:
    """
    TODO Explain Class
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
    TODO Explain Class

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
    TODO Explain Class
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
    TODO Explain the Function

    Asserts
    -------
    **Explanation**

    """

    def f(x1: _CustomCls1 | _CustomCls2 | _CustomCls3):
        """
        TODO Explain the Function

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
    assert isinstance(dim, d.Variant)
    assert len(dim.children) == 3
    assert dim.nullable is False


def test_union_choice_spec_param_func_long():
    """
    TODO Explain the Function

    Asserts
    -------
    **Explanation**

    """

    def f(x1: Union[_CustomCls1, _CustomCls2, _CustomCls3]):
        """
        TODO Explain the Function

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
    assert isinstance(dim, d.Variant)
    assert len(dim.children) == 3
    assert dim.nullable is False
