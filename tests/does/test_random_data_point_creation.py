"""
Tests for the creation of random data points in a design of experiment (DOE).
"""

import numpy as np

import raxpy.spaces as s
import raxpy.does.random as r
from . import test_creation_of_space_filling_doe as sf_doe


def test_creation_of_random_doe():
    """
    Tests the creation of a experiment design using random number generation.

    Asserts
    -------
    design created and every point belongs to a sub-space

    """
    design1 = r.generate_design(sf_doe.SPACE, n_points=10)
    

    assert design1 is not None
    assert design1.point_count == 10
    sf_doe.assert_every_point_in_a_full_sub_space(sf_doe.SUB_SPACES, design1)

    rng = np.random.default_rng(seed=42)
    design = r.generate_design(sf_doe.SPACE, n_points=10, rng=rng)
    assert design is not None
    assert design.point_count == 10
    sf_doe.assert_every_point_in_a_full_sub_space(sf_doe.SUB_SPACES, design)
    rng2 = np.random.default_rng(seed=42)
    design2 = r.generate_design(sf_doe.SPACE, n_points=10, rng=rng2)
    assert design2 is not None
    assert design2.point_count == 10
    sf_doe.assert_every_point_in_a_full_sub_space(sf_doe.SUB_SPACES, design2)
    # make sure the same seed produces the same design
    assert np.all(design.input_sets == design2.input_sets)
    assert np.any(design1.input_sets != design2.input_sets)

def test_collapse_of_random_numbers():
    """
    Tests the collapse of random numbers to a design.

    Asserts
    -------
    the collapse of random numbers to a design is consistent with the design's
    input space

    """
    space = s.Space(
        dimensions=[
            s.Float(id="x1", lb=3.0, ub=5.0, nullable=False),
            s.Float(
                id="x2", lb=-3.0, ub=-5.0, nullable=True, portion_null=0.33
            ),
            s.Composite(
                id="x3",
                nullable=True,
                portion_null=0.33,
                children=[
                    s.Int(id="x4", lb=6, ub=7, nullable=False),
                    s.Float(
                        id="x5",
                        value_set=[0.1, 0.5, 0.9],
                        nullable=True,
                        portion_null=0.33,
                    ),
                ],
            ),
            s.Variant(
                id="x6",
                nullable=True,
                portion_null=0.33,
                options=[
                    s.Float(id="x7", lb=1.0, ub=2.0, nullable=False),
                    s.Float(id="x8", lb=3.0, ub=4.0, nullable=False),
                ],
            ),
            s.Text(id="x9", value_set=("one", "two", "three")),
        ]
    )
    assert space.count_dimensions() == 9
    
    flatted_dimensions = s.create_all_iterable(space.children)
    dim_map = {}
    for i, dim in enumerate(flatted_dimensions):
        dim_map[dim.local_id] = i
    np.random.seed(42)
    random_x = np.random.rand(10, 9)
    collapsed_x = space.decode_zero_one_matrix(
        random_x, dim_map, map_null_to_children_dim=True
    )

    assert collapsed_x is not None

    x3_values = collapsed_x[:, dim_map["x3"]]
    x4_values = collapsed_x[:, dim_map["x4"]]
    # make sure if x3 is null, x4 is also null
    assert np.all(np.isnan(x3_values) == np.isnan(x4_values))

    x6_values = collapsed_x[:, dim_map["x6"]]

    # make sure x7 is only specified when x6 is 0
    x7_values = collapsed_x[:, dim_map["x7"]]
    # make sure x8 is only specified when x6 is 1
    x8_values = collapsed_x[:, dim_map["x8"]]

    for i, x6_value in enumerate(x6_values):
        if np.isnan(x6_value):
            assert np.isnan(x7_values[i])
            assert np.isnan(x8_values[i])
        elif x6_value == 0:
            assert not np.isnan(x7_values[i])
            assert np.isnan(x8_values[i])
        elif x6_value == 1:
            assert np.isnan(x7_values[i])
            assert not np.isnan(x8_values[i])
