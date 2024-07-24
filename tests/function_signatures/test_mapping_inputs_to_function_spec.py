from typing import Annotated, Optional
from dataclasses import dataclass

import raxpy


def test_mapping_inputs_to_func_spec():
    @dataclass
    class ChildCustomObject:
        caf1: float
        caf2: Optional[float]
        cas1: str
        cas1: Annotated[
            str, raxpy.Categorical(value_set={"one", "two", "three"})
        ]

    def f():
        pass
