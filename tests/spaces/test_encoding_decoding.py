"""
Unit tests for encoding and decoding of dimensions
"""

import raxpy.spaces as s
import numpy as np


def test_variant_type():
    """
    Tests that the encoding and decoding of a variant dimension is consistent
    Asserts
    -------
    The encoding and decoding of a variant dimension is consistent
    """

    dim = s.Variant(
        id="x1",
        nullable=True,
        portion_null=0.2,
        options=[
            s.Int(id="x2", lb=1, ub=2, nullable=False),
            s.Int(id="x3", lb=9, ub=10, nullable=False),
            s.Int(id="x4", lb=12, ub=13, nullable=False),
            s.Int(id="x5", lb=19, ub=20, nullable=False),
        ],
    )
    decoded_data = np.array([1, 2, 0, np.nan, 1])
    encoded_data = dim.reverse_decoding(decoded_data)
    decoded_data1 = dim.collapse_uniform(
        encoded_data, utilize_null_portions=False
    )

    assert np.array_equal(
        encoded_data,
        np.array([1 / 3, 2 / 3, 0.0 / 3, np.nan, 1 / 3]),
        equal_nan=True,
    )

    assert np.array_equal(decoded_data, decoded_data1, equal_nan=True)
