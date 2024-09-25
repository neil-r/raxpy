""" 
    Tests numpy to standard Python conversions
"""

import numpy as np
import raxpy.spaces.s as d
import raxpy.spaces.root as s


def test_create_dict_for_nan():
    """
    Tests if create_dict_from_flat_values function outputs a NaN value
    """
    input_f = [
        d.Float(
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
        d.Float(
            id="x2",
            local_id="x2",
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
    ]
    input_set_map = {"x1": 0, "x2": 1}
    input_array = [3.8165008, np.nan]

    output = s._create_dict_from_flat_values(
        input_f, input_array, input_set_map
    )

    for key, value in output.items():
        if value != value:
            assert np.isnan(value) is True
