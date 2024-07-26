""" TODO Explain Module"""

from typing import Annotated, Optional
from dataclasses import dataclass

import raxpy


def test_mapping_inputs_to_func_spec():
    """
    TODO Explain the Function
    """

    @dataclass
    class ChildCustomObject:
        """
        TODO Explain Class

        """

        caf1: float
        caf2: Optional[float]
        cas1: str
        cas1: Annotated[
            str, raxpy.Categorical(value_set={"one", "two", "three"})
        ]

    def f():
        """
        TODO Explain the Function **Not Implemented**
        """
        pass
