"""
Unit test for the ValuePool data structure.
"""

import numpy as np

from raxpy.does.lhs import ValuePool


def test_value_pool():
    """
    Tests `raxpy.does.lhs.ValuePool`

    Asserts
    -------
        The pull method removes values from the pool
    """
    p = ValuePool(10, outline_mode=False)

    rng = np.random.default_rng(seed=42)

    values_1 = p.pull(3, rng)

    assert len(values_1) == 3

    values_2 = p.pull(4, rng)

    assert len(values_2) == 4

    values_3 = p.pull(1, rng)

    assert len(values_3) == 1

    values_4 = p.pull(2, rng)

    assert len(values_4) == 2

    all_values = values_1 + values_2 + values_3 + values_4

    # ensure no duplicates
    assert 10 == len(set(all_values))

    # ensure every number is in the list
    for i in range(10):
        v_check = i / 10.0 + (1 / 20.0)

        assert v_check in all_values


def test_value_pool_outline_mode():
    """
    Tests `raxpy.does.lhs.ValuePool`

    Asserts
    -------
        The pull method removes values from the pool
    """
    p = ValuePool(10, outline_mode=True)

    rng = np.random.default_rng(seed=42)

    values_1 = p.pull(3, rng)

    assert len(values_1) == 3

    values_2 = p.pull(4, rng)

    assert len(values_2) == 4

    values_3 = p.pull(1, rng)

    assert len(values_3) == 1

    values_4 = p.pull(2, rng)

    assert len(values_4) == 2

    all_values = values_1 + values_2 + values_3 + values_4

    # ensure no duplicates
    assert 10 == len(set(all_values))

    # round to address numerical imprecisions
    all_values = {round(x, 2) for x in all_values}
    # ensure every number is in the list
    for i in range(10):
        v_check = round(i / (10.0 - 1), 2)

        assert v_check in all_values
