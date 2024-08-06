""" TODO Explain Module """

from typing import Annotated, Optional

import pytest

import raxpy


@raxpy.validate_at_runtime(check_outputs=False)
def f(
    x1: Annotated[float, raxpy.Float(lb=3.0, ub=4.0)],
    x2: Annotated[Optional[float], raxpy.Float(lb=0.0, ub=3.0)] = 1.5,
):
    """
    The following code should execute the computations with these
    values, such as running a simulation or training a machine
    learning model to keep it simple for this demonstration, we
    simply compute a polynominal.

    Arguments
    ---------
    x1 : Annotated[float]
        **Explanation**
    x2 : Annotated[Optional[float]] = 1.5
        x2 is annotated as Optional. This indicates that this parameter
        is optional (users can call this function with setting x2 to
        None) The function specification also provides a lower and upper
        bound for each float input parameter.

    Returns
    -------
    y : float
        **Explanation**
    """

    y = 0.4 * x1
    if x2 is not None:
        y += (x2 * 3.0) + (0.7 * x2 * x1)
    return y


def test_perform_basic_batch_experiment():
    """
    TODO Explain the Function

    Asserts
    -------
    **Explanation**

    """

    inputs, outputs = raxpy.perform_batch_experiment(f, 10)

    assert inputs is not None
    assert len(inputs) == 10
    assert len(inputs[2]) == 2
    assert len(outputs) == 10
    assert isinstance(outputs[0], float)


def test_validation_decorator():
    """
    TODO Explain the Function
    """
    # violoate lower bound of x1
    with pytest.raises(ValueError):
        f(0.1, 0.4)

    # violoate upper bound of x2
    with pytest.raises(ValueError):
        f(3.0, 11.1)

    # do not violoate any bounds
    f(3.0, 0.1)
