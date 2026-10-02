"""Tests for full factorial designs."""

import numpy as np

import raxpy.spaces as s
from raxpy.does.doe import EncodingEnum
from raxpy.does.factorial import generate_full_factorial_design


def test_full_factorial_design_covers_nested_nullable_subspaces():
    """A factorial design includes every valid nested nullable subspace."""
    space = s.InputSpace(
        dimensions=[
            s.Text(id="category", value_set=("red", "blue")),
            s.Int(id="count", lb=0, ub=4),
            s.Composite(
                id="group",
                nullable=True,
                portion_null=0.25,
                children=[
                    s.Int(id="inner_count", lb=1, ub=2),
                    s.Composite(
                        id="nested_group",
                        nullable=True,
                        portion_null=0.25,
                        children=[s.Text(id="label", value_set=("a", "b"))],
                    ),
                ],
            ),
        ]
    )

    design = generate_full_factorial_design(space, n_levels=2)

    assert design.encoding == EncodingEnum.ZERO_ONE_NULL_ENCODING
    assert design.input_set_map == {
        "category": 0,
        "count": 1,
        "group": 2,
        "inner_count": 3,
        "nested_group": 4,
        "label": 5,
    }

    actual_subspaces = {
        tuple(
            dim_id
            for dim_id, column_index in design.input_set_map.items()
            if not np.isnan(row[column_index])
        )
        for row in design.input_sets
    }
    expected_subspaces = {
        tuple(dim_id for dim_id in subspace if dim_id in design.input_set_map)
        for subspace in space.derive_concept_subspaces()
    }

    assert actual_subspaces == expected_subspaces
    assert np.all(
        np.logical_or(
            np.isnan(design.input_sets),
            (design.input_sets >= 0.0) & (design.input_sets <= 1.0),
        )
    )


def test_full_factorial_design_uses_center_point_for_one_level():
    """A one-level design uses the continuous center and first categories."""
    space = s.InputSpace(
        dimensions=[
            s.Float(id="continuous", lb=0.0, ub=1.0),
            s.Text(id="category", value_set=("red", "blue")),
            s.Int(id="count", lb=0, ub=4),
        ]
    )

    design = generate_full_factorial_design(space, n_levels=1)

    assert design.point_count == 1
    assert np.array_equal(design.input_sets, np.array([[0.5, 0.0, 0.0]]))


def test_full_factorial_design_enumerates_categorical_levels():
    """Discrete dimensions use every level independently of n_levels."""
    space = s.InputSpace(
        dimensions=[
            s.Float(id="continuous", lb=0.0, ub=1.0),
            s.Text(id="category", value_set=("red", "blue")),
            s.Int(id="count", lb=1, ub=3),
        ]
    )

    design = generate_full_factorial_design(space, n_levels=2)

    assert design.point_count == 12
    assert set(design.input_sets[:, 0]) == {0.0, 1.0}
    assert set(design.input_sets[:, 1]) == {0.0, 1.0}
    assert set(design.input_sets[:, 2]) == {0.0, 0.5, 1.0}


def test_full_factorial_design_uses_variant_option_indices():
    """Variant values are option indices and activate only their option."""
    space = s.InputSpace(
        dimensions=[
            s.Variant(
                id="choice",
                options=[
                    s.Text(id="text_option", value_set=("a", "b")),
                    s.Int(id="int_option", lb=1, ub=2),
                ],
            )
        ]
    )

    design = generate_full_factorial_design(space, n_levels=2)
    choice_column = design.input_set_map["choice"]
    text_column = design.input_set_map["text_option"]
    int_column = design.input_set_map["int_option"]

    assert design.point_count == 4
    assert set(design.input_sets[:, choice_column]) == {0.0, 1.0}
    for row in design.input_sets:
        if row[choice_column] == 0:
            assert not np.isnan(row[text_column])
            assert np.isnan(row[int_column])
        else:
            assert np.isnan(row[text_column])
            assert not np.isnan(row[int_column])
