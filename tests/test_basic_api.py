""" 
    Tests the high-level perform experiment API and default settings.
"""

from typing import Annotated, Optional
from dataclasses import dataclass
import pytest

import raxpy


@dataclass
class Object:
    x3_1: Annotated[float, raxpy.Float(lb=-4.0, ub=-3.0)]
    x3_2: Annotated[Optional[float], raxpy.Float(lb=0.0, ub=3.0)] = 1.5


@raxpy.validate_at_runtime(check_outputs=False)
def f(
    x1: Annotated[float, raxpy.Float(lb=3.0, ub=4.0)],
    x2: Annotated[Optional[float], raxpy.Float(lb=0.0, ub=3.0)] = 1.5,
    x3: Optional[Object] = None,
):
    """
    A test function to support testing of runtime
    validation.

    Arguments
    ---------
    x1 : Annotated[float]
        input 1
    x2 : Annotated[Optional[float]] = 1.5
        input 2
    x3 : Optional[Object]
        input 3

    Returns
    -------
    y : float
        the test function's return vale
    """

    y = 0.4 * x1
    if x3 is not None:
        y += x3.x3_1 * x1
        if x3.x3_2 is not None:
            y += x3.x3_2
    if x2 is not None:
        y += (x2 * 3.0) + (0.7 * x2 * x1)
    return y


def test_perform_basic_batch_experiment():
    """
    Tests the default perform experiment
    settings

    Asserts
    -------
        Experiment on f is executed without error and
        returns the proper number of inputs and outputs.

    """

    inputs, outputs = raxpy.perform_experiment(f, 10)

    assert inputs is not None
    assert len(inputs) == 10
    assert len(inputs[2]) == 3
    assert len(outputs) == 10
    assert isinstance(outputs[0], float)


def test_validation_decorator():
    """
        Tests the runtime validation decorator

    Asserts
    -------
        Invalid inputs raise a ValueError during runtime
    """
    # violoate lower bound of x1
    with pytest.raises(ValueError):
        f(0.1, 0.4)

    # violoate upper bound of x2
    with pytest.raises(ValueError):
        f(3.0, 11.1)

    # do not violoate any bounds
    f(3.0, 0.1)

    # violoate lower bound of x3_1
    with pytest.raises(ValueError):
        f(3.0, 0.1, Object(-5.0))

    # violoate upper bound of x3_2
    with pytest.raises(ValueError):
        f(3.0, 0.1, Object(-3.5, 3.01))

    # do not violoate any bounds
    f(3.0, 0.1, Object(-3.5, 3.0))
