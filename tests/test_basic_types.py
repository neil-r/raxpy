"""Testing implementation of raxpy"""

from typing import Optional, Annotated

import numpy as np
import pytest

import raxpy


def f_float(
    x1: Annotated[float, raxpy.Float(lb=3.0, ub=4.0)],
    x2: Annotated[Optional[float], raxpy.Float(lb=0.0, ub=3.0)],
) -> float:
    """ Control (float) function"""

    y = 0.4 * x1
    if np.isnan(x2):
        return y
    y += (x2 * 3.0) + (0.7 * x2*x1)
    return y


def f_float_no_optional(
  x1: Annotated[float, raxpy.Float(lb=3.0, ub=4.0)],
  x2: Annotated[float, raxpy.Float(lb=0.0, ub=3.0)],
) -> float:
    """ No Optional args (float) function"""

    y = 0.4 * x1
    if x2 is not None:
        y += (x2 * 3.0) + (0.7 * x2*x1)
    return y


def f_float_6(
  x1: Annotated[float, raxpy.Float(lb=3.0, ub=4.0)],
  x2: Annotated[Optional[float], raxpy.Float(lb=0.0, ub=3.0)],
  x3: Annotated[Optional[float], raxpy.Float(lb=0.0, ub=3.0)],
  x4: Annotated[Optional[float], raxpy.Float(lb=0.0, ub=3.0)],
  x5: Annotated[Optional[float], raxpy.Float(lb=0.0, ub=3.0)],
  x6: Annotated[Optional[float], raxpy.Float(lb=0.0, ub=3.0)],
) -> float:
    """ Float function with large number of optional args"""

    y = 0.4 * x1
    if x2 is not None:
        y += (x2 * 3.0) + (0.7 * x2*x1)
        if x3 is not None:
            y += (x2 * 3.0) + (0.7 * x2*x1) + (0.7 * x2*x3)
            if x4 is not None:
                y += ((x2 * 3.0) + (0.7 * x2*x1)
                      + (0.7 * x2*x3) + (0.7 * x3*x4))
                if x5 is not None:
                    y += ((x2 * 3.0) + (0.7 * x2*x1) + (0.7 * x2*x3)
                          + (0.7 * x3*x4) + (0.7 * x4*x5))
                    if x6 is not None:
                        y += ((x2 * 3.0) + (0.7 * x2*x1) + (0.7 * x2*x3)
                              + (0.7 * x3*x4) + (0.7 * x4*x5) + (0.7 * x5*x6))
    return y


def f_int(
    x1: Annotated[int, raxpy.Integer(lb=3, ub=5)],
    x2: Annotated[Optional[int], raxpy.Integer(lb=0, ub=3)],
) -> int:
    """ Control (Int) Function"""
    y = 4 * x1
    if x2 is not None:
        y += (x2 * 3) + (2 * x2*x1)
    return y


def f_int_with_float_coefficient(
  x1: Annotated[int, raxpy.Integer(lb=3, ub=5)],
  x2: Annotated[Optional[int], raxpy.Integer(lb=0, ub=3)],
) -> float:
    """ Int Function with math using floats"""

    y = 0.4 * x1
    if x2 is not None:
        y += (x2 * 3.0) + (0.7 * x2*x1)
    return y


def test_perform_basic_batch_experiment():
    """ Test that batch experiment output is not NaN (Not a Number)"""
    inputs, outputs = raxpy.perform_batch_experiment(f_float, 100)

    assert any(np.isnan(outputs)) is not True
