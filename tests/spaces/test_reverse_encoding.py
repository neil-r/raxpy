"""
Tests the dimension reversing encodings and decodings
"""

from raxpy.spaces import dimensions
import numpy as np


def test_reverse_decoding_for_variant():
    """
    Tests that reverse decoding for a Variant dimension correctly encodes
    values and decodes them back to the original values.

    Asserts
    -------
        Encoded values are within the expected range (0 to 1)
        Decoded values match the original values
    """

    dim = dimensions.Variant(
        id="test_dim",
        nullable=True,
        portion_null=0.2,
        options=[
            dimensions.Float(id="option1", lb=0.0, ub=1.0, nullable=False),
            dimensions.Float(id="option2", lb=0.0, ub=1.0, nullable=False),
            dimensions.Float(id="option3", lb=0.0, ub=1.0, nullable=False),
            dimensions.Float(id="option4", lb=0.0, ub=1.0, nullable=False),
        ],
    )

    # Test reverse decoding with a list of values
    values = [0, 3, np.nan, 2, 1]
    encoded_values = dim.reverse_decoding(values)
    expected_encoded_values = [0.0, 1.0, np.nan, (2 / 3), (1 / 3)]

    # Check that the encoded values are within the expected range (0 to 1)
    assert all(0 <= x <= 1 or np.isnan(x) for x in encoded_values)
    assert np.allclose(encoded_values, expected_encoded_values, equal_nan=True)

    decoded_values = dim.collapse_uniform(
        encoded_values, utilize_null_portions=False
    )

    assert np.allclose(decoded_values, values, equal_nan=True)
