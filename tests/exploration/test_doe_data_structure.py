import numpy as np
import pytest

import raxpy.does.doe as doe


def test_doe_counts():
    design = doe.DesignOfExperiment(
        input_sets=np.array(
            [
                [1, 2, 3],  # point #1
                [4, 5, 6],  # point #2
            ]
        ),
        input_set_map={"x1": 0, "x2": 1, "x3": 2},
    )

    assert design.point_count == 2
    assert design.dim_specification_count == 3


def test_error_init_doe():

    with pytest.raises(ValueError):
        # create a design with one column not specified
        doe.DesignOfExperiment(
            input_sets=np.array(
                [
                    [1, 2, 3],
                    [4, 5, 6],
                ]
            ),
            input_set_map={
                "x1": 0,
                "x2": 1,
            },
        )

    with pytest.raises(ValueError):
        # create a design with one column specification out-of-bounds
        doe.DesignOfExperiment(
            input_sets=np.array(
                [
                    [1, 2, 3],
                    [4, 5, 6],
                ]
            ),
            input_set_map={
                "x1": 0,
                "x2": 1,
                "x3": 10000,
            },
        )

    with pytest.raises(ValueError):
        # create a design with two column specification to the same index
        doe.DesignOfExperiment(
            input_sets=np.array(
                [
                    [1, 2, 3],
                    [4, 5, 6],
                ]
            ),
            input_set_map={
                "x1": 0,
                "x2": 1,
                "x3": 1,
            },
        )
