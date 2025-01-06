""" 
    Tests numpy to standard Python conversions
"""

import numpy as np
import raxpy.spaces as s
from raxpy.spaces.root import _create_dict_from_flat_values


def test_create_dict_for_nan_and_variant():
    """
    Tests if create_dict_from_flat_values function outputs a NaN value and variants
    children data is converted.
    """
    input_f = [
        s.Float(
            id="x1",
            local_id="x1",
            nullable=False,
            specified_default=False,
            label=None,
            default_value=None,
            tags=None,
            portion_null=0.0,
            lb=3.0,
            ub=4.0,
            value_set=None,
        ),
        s.Float(
            id="x2",
            local_id="outputx2",
            nullable=True,
            specified_default=False,
            label=None,
            default_value=None,
            tags=None,
            portion_null=0.1,
            lb=0.0,
            ub=3.0,
            value_set=None,
        ),
        s.Variant(
            id="x3",
            nullable=False,
            portion_null=0.0,
            options=[
                s.Composite(
                    id="x4",
                    children=[
                        s.Int(id="x5", lb=6, ub=7),
                        s.Float(
                            id="x6",
                            value_set=[0.1, 0.5, 0.9],
                        ),
                    ],
                ),
                s.Float(
                    id="x7",
                    value_set=[0.1, 0.5, 0.9],
                ),
            ],
        ),
    ]
    input_set_map = {"x1": 0, "x2": 1, "x3": 2, "x5": 3, "x6": 4, "x7": 5}
    input_array = [3.8165008, np.nan, 0.0, 6, 0.1, np.nan]

    output = _create_dict_from_flat_values(input_f, input_array, input_set_map)

    for key, value in output.items():
        if value != value:
            assert np.isnan(value) is True

    assert "x3" in output
    assert output["x3"] is not None
    assert "x4" in output["x3"]
    assert "x7" not in output["x3"]
    assert output["x3"]["x4"]["x5"] == 6
    assert output["x3"]["x4"]["x6"] == 0.1

    input_array_2 = [3.8165008, np.nan, 1.0, np.nan, np.nan, 0.9]

    output_2 = _create_dict_from_flat_values(
        input_f, input_array_2, input_set_map
    )

    assert "x3" in output_2
    assert output_2["x3"] is not None
    assert "x4" not in output_2["x3"]
    assert "x7" in output_2["x3"]
    assert output_2["x3"]["x7"] == 0.9
