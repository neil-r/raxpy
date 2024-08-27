""" TODO Explain Module"""

import numpy as np

import raxpy.spaces.dimensions as d
import raxpy.spaces.root as s
import raxpy.does.lhs as doe


def assert_every_point_in_a_full_sub_space(
    sub_spaces_list, design: doe.DesignOfExperiment
):
    """
    Helper test function to assert that every point is a full sub-space specified

    """

    sub_space_index_map = {}

    for i, sub_space in enumerate(sub_spaces_list):
        sub_space_index_map[sub_space] = i

    # determine the sub-space each data-point belongs to
    def map_point(point):
        active_dim_ids = []

        for dim_id, column_index in design.input_set_map.items():
            if ~np.isnan(point[column_index]):
                active_dim_ids.append(dim_id)

        active_dim_ids.sort()
        assert tuple(a for a in active_dim_ids) in sub_space_index_map

    # compute the subspace each point belongs to
    for point in design.decoded_input_sets:
        map_point(point)


SPACE = s.InputSpace(
    dimensions=[
        d.Float(id="x1", lb=3.0, ub=5.0),
        d.Float(
            id="x2",
            lb=-3.0,
            ub=-5.0,
            nullable=True,
            portion_null=1.0 / 10.0,
        ),
        d.Composite(
            id="x3",
            nullable=True,
            portion_null=1.0 / 7.0,
            children=[
                d.Int(id="x4", lb=6, ub=7),
                d.Float(
                    id="x5",
                    value_set=[0.1, 0.5, 0.9],
                    nullable=True,
                    portion_null=1.0 / 4.0,
                ),
            ],
        ),
        d.Variant(
            id="x6",
            nullable=True,
            portion_null=0.33,
            options=[
                d.Float(id="x7", lb=1.0, ub=2.0),
                d.Float(id="x8", lb=3.0, ub=4.0),
            ],
        ),
    ]
)

SUB_SPACES = (
    ("x1", "x2"),
    ("x1", "x3", "x4", "x5"),
    ("x1", "x3", "x4"),
    ("x1", "x6", "x7"),
    ("x1", "x6", "x8"),
    ("x1", "x2", "x3", "x4", "x5"),
    ("x1", "x2", "x3", "x4"),
    ("x1", "x2", "x6", "x7"),
    ("x1", "x2", "x6", "x8"),
    ("x1", "x3", "x4", "x5", "x6", "x7"),
    ("x1", "x3", "x4", "x6", "x7"),
    ("x1", "x3", "x4", "x5", "x6", "x8"),
    ("x1", "x3", "x4", "x6", "x8"),
    ("x1", "x2", "x3", "x4", "x5", "x6", "x7"),
    ("x1", "x2", "x3", "x4", "x6", "x7"),
    ("x1", "x2", "x3", "x4", "x5", "x6", "x8"),
    ("x1", "x2", "x3", "x4", "x6", "x8"),
    ("x1",),
)


def test_creation_of_space_filling_doe_simple_merge():
    """
    Tests the creation of a space-filling design with a level-by-level method.

    The level-sub-designs are simplyed injected in the order they were generated.

    Asserts
    -------
    Design is not None
    every point in design is in a full-sub-space

    """

    design = doe.generate_design(SPACE, 100, merge_method=doe.MERGE_SIMPLE)
    assert design is not None
    assert design.point_count == 100
    assert_every_point_in_a_full_sub_space(
        sub_spaces_list=SUB_SPACES, design=design
    )


def test_creation_of_space_filling_doe_shadow_merge():
    """
    Tests the creation of a space-filling design with a level-by-level method.

    The level-sub-design points are merged using a shadow merge technique that
    aligns similar partial-points of the shadow design with close partial-
    points of the level-sub-design.

    Asserts
    -------
    Design is not None
    every point in design is in a full-sub-space

    """
    design = doe.generate_design(
        SPACE, 100, merge_method=doe.MERGE_SHADOW_DESIGN
    )
    assert design is not None
    assert design.point_count == 100
    assert_every_point_in_a_full_sub_space(
        sub_spaces_list=SUB_SPACES, design=design
    )


def test_creation_of_space_filling_by_subspaces():
    """
    TODO Explain the Function

    Asserts
    -------
    Design is not None

    """
    design = doe.generate_seperate_designs_by_full_subspace(SPACE, 100)

    assert design is not None
    assert design.point_count == 100
    assert_every_point_in_a_full_sub_space(
        sub_spaces_list=SUB_SPACES, design=design
    )


def test_creation_of_space_filling_by_subspaces_null_fill():
    """
    TODO Explain the Function

    Asserts
    -------
    Design is not None

    """
    space = s.InputSpace(
        dimensions=[
            d.Float(
                id="x1",
                lb=-3.0,
                ub=-5.0,
                nullable=True,
                portion_null=1.0 / 10.0,
            ),
            d.Float(
                id="x2",
                lb=-3.0,
                ub=-5.0,
                nullable=True,
                portion_null=1.0 / 10.0,
            ),
            d.Float(
                id="x3",
                lb=-3.0,
                ub=-5.0,
                nullable=True,
                portion_null=1.0 / 10.0,
            ),
        ]
    )
    design = doe.generate_seperate_designs_by_full_subspace(space, 100)

    assert design is not None
    assert design.point_count == 100
    assert_every_point_in_a_full_sub_space(
        sub_spaces_list=(
            tuple(),
            ("x1",),
            ("x2",),
            ("x1", "x2"),
            ("x3",),
            ("x1", "x3"),
            ("x2", "x3"),
            ("x1", "x2", "x3"),
        ),
        design=design,
    )


def test_creation_with_lhs_projection():
    """
    TODO Explain the Function

    Asserts
    -------
    Design is not None
    """
    design = doe.generate_design_with_projection(SPACE, 100)

    assert design is not None
    assert design.point_count == 100
    assert_every_point_in_a_full_sub_space(
        sub_spaces_list=SUB_SPACES, design=design
    )
