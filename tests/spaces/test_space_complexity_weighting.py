""" TODO Explain Module """

import math
from typing import List, Tuple, Iterable
import raxpy.spaces.dimensions as d
import raxpy.spaces.complexity as c
import raxpy.spaces.root as s


def test_assign_null_portions():
    """
    TODO Explain the Function

    Asserts
    -------
    **Explanation**

    """
    space = s.Space(
        dimensions=[
            d.Float(id="x1", lb=3.0, ub=5.0),
            d.Float(
                id="x2",
                lb=-3.0,
                ub=-5.0,
                nullable=True,
            ),
            d.Composite(
                id="x3",
                nullable=True,
                children=[
                    d.Int(id="x4", lb=6, ub=7),
                    d.Float(
                        id="x5",
                        value_set=[0.1, 0.5, 0.9],
                        nullable=True,
                    ),
                ],
            ),
            d.Variant(
                id="x6",
                nullable=True,
                options=[
                    d.Float(
                        id="x7",
                        value_set=[0.1, 0.5, 0.9],
                    ),
                    d.Float(
                        id="x8",
                        value_set=[0.1, 0.5, 0.9],
                    ),
                ],
            ),
        ]
    )

    # adjust the porition null values
    c.assign_null_portions(s.create_level_iterable(space.children))

    # an dimension that cannot be null should not portion
    # any nulls in data points
    assert space.dimensions[0].portion_null == 0.0
    # assert space.dimensions[0].portion_null is not None
    # the default heuristic for optional floats 1 out of 4
    assert space.dimensions[1].portion_null == 1.0 / 4.0
    # the default heuristic for optional composites is to
    # add the largest, then the root of the next ,etc
    assert space.dimensions[2].portion_null == 1.0 / (
        (4 ** (1 / 1) + 2 ** (1 / 2)) + 1
    )
    # the default heuristic for optional Varient/union is to
    # add the largest, then the root of the next ,etc
    assert space.dimensions[3].portion_null == 1.0 / (
        (3 ** (1 / 1) + 3 ** (1 / 2)) + 1
    )


def test_subspace_portitions_computations():
    """
    TODO Explain the Function

    Asserts
    -------
    **Explanation**
    """
    space = s.Space(
        dimensions=[
            d.Float(id="x1", lb=3.0, ub=5.0),
            d.Float(
                id="x2",
                lb=-3.0,
                ub=-5.0,
                nullable=True,
                portion_null=0.1,
            ),
            d.Composite(
                id="x3",
                nullable=True,
                children=[
                    d.Int(id="x4", lb=6, ub=7),
                    d.Float(
                        id="x5",
                        value_set=[0.1, 0.5, 0.9],
                        nullable=True,
                        portion_null=0.2,
                    ),
                ],
                portion_null=0.5,
            ),
            d.Variant(
                id="x6",
                nullable=True,
                options=[
                    d.Float(
                        id="x7",
                        value_set=[0.1, 0.5, 0.9],
                    ),
                    d.Float(
                        id="x8",
                        value_set=[0.1, 0.5, 0.9],
                    ),
                ],
                portion_null=0.25,
            ),
        ]
    )
    full_subspace_sets = space.derive_full_subspaces()
    portitions = c.compute_subspace_portions(space, full_subspace_sets)

    assert portitions is not None
    assert len(portitions) == len(full_subspace_sets)

    index_of_check = full_subspace_sets.index(
        ["x1", "x3", "x4", "x5", "x6", "x8"]
    )
    p_check = portitions[index_of_check]
    p_expected = (
        1.0 * 0.1 * 0.5 * 0.8 * 0.75 * 0.5
    )  # multipying by the approiate scaler if or if not included
    # x1 * !x2 * ...
    assert p_check == p_expected
    total = sum(portitions)
    assert total == 1.0
