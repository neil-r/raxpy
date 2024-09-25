"""
    Unit test for the ValuePool data structure.
"""

from raxpy.does.lhs import ValuePool


def test_value_pool():
    """
    Tests `raxpy.does.lhs.ValuePool`

    Asserts
    -------
        The pull method removes values from the pool
    """
    p = ValuePool(10)

    values_1 = p.pull(3)

    assert len(values_1) == 3

    values_2 = p.pull(4)

    assert len(values_2) == 4

    values_3 = p.pull(1)

    assert len(values_3) == 1

    values_4 = p.pull(2)

    assert len(values_4) == 2

    all_values = values_1 + values_2 + values_3 + values_4

    # ensure no duplicates
    assert 10 == len(set(all_values))

    # ensure every number is in the list
    for i in range(10):
        v_check = i / 10.0 + (1 / 20.0)

        assert v_check in all_values
