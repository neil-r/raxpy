from typing import Annotated, Optional, Union
from dataclasses import dataclass

import raxpy
import raxpy.annotations.function_spec as fs
import raxpy.spaces.dimensions as d


def test_union_choice_spec_param_func():

    @dataclass
    class CustomCls1:
        x1: Annotated[float, raxpy.Float(lb=0.0, ub=1.0)]
        x2: Annotated[float, raxpy.Float(lb=0.0, ub=1.0)]

        def execute(self):
            """
            demonstration method that does nothing"
            """

    @dataclass
    class CustomCls2:
        x1: Annotated[float, raxpy.Float(lb=0.0, ub=2.0)]
        x2: Annotated[float, raxpy.Float(lb=0.0, ub=2.0)]

        def execute(self):
            """
            demonstration method that does nothing"
            """

    @dataclass
    class CustomCls3:
        x1: Annotated[float, raxpy.Float(lb=0.0, ub=2.0)]
        x2: Annotated[float, raxpy.Float(lb=0.0, ub=2.0)]
        x3: Annotated[float, raxpy.Float(lb=0.0, ub=3.0)]

        def execute(self):
            """
            demonstration method that does nothing"
            """

    def f(
        x1: Union[
            CustomCls1,
            CustomCls2,
            CustomCls3,
        ]
    ):
        x1.execute()

    input_space = fs.extract_input_space(f)
    assert input_space is not None
    assert input_space.dimensions is not None
    assert len(input_space.dimensions) == 1
    dim = input_space.dimensions[0]
    assert isinstance(dim, d.Variant)
    assert len(dim.children) == 3
    assert dim.nullable is False
