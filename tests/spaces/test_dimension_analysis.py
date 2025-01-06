""" 
    Unit tests for analysis functions of a space's dimensions
"""

import raxpy.spaces as s

from tests.does.test_creation_of_space_filling_doe import SPACE, SUB_SPACES


def test_deriving_subspaces():
    """
    Tests the analysis of a Space with optional dimensions to discover all
    the possible full-sub-spaces.

    Asserts
    -------
        The full-sub-spaces are derived to the proper number and
        delinated with the proper dimension specifications
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
                    s.Composite(
                        # Note that x6 is ignore since it is
                        # just for the specification's structure
                        id="x6",
                        nullable=False,
                        children=[
                            s.Float(
                                id="x5",
                                value_set=[0.1, 0.5, 0.9],
                                nullable=True,
                                portion_null=0.33,
                            )
                        ],
                    ),
                ],
            ),
        ]
    )

    sub_spaces = space.derive_full_subspaces()

    assert sub_spaces is not None
    assert len(sub_spaces) == 6
    assert ["x1", "x2", "x3", "x4", "x5"] in sub_spaces
    assert ["x1", "x2", "x3", "x4"] in sub_spaces
    assert ["x1", "x3", "x4", "x5"] in sub_spaces
    assert ["x1", "x3", "x4"] in sub_spaces
    assert ["x1", "x2"] in sub_spaces
    assert ["x1"] in sub_spaces


def test_deriving_subspaces_from_unions():
    """
    Tests the analysis of a Space with union dimensions to discover all
    the possible full-sub-spaces.

    Asserts
    -------
        The full-sub-spaces are derived to the proper number and
        delinated with the proper dimension specifications
    """

    space = s.Space(
        dimensions=[
            s.Variant(
                id="xb",
                nullable=True,
                portion_null=0.33,
                options=[
                    s.Float(id="x1", lb=1.0, ub=2.0, nullable=False),
                    s.Float(id="x2", lb=3.0, ub=4.0, nullable=False),
                    s.Float(id="x3", lb=5.0, ub=6.0, nullable=False),
                ],
            ),
        ]
    )

    sub_spaces = space.derive_full_subspaces()

    assert sub_spaces is not None
    assert len(sub_spaces) == 4
    assert ["xb", "x1"] in sub_spaces
    assert ["xb", "x2"] in sub_spaces
    assert ["xb", "x3"] in sub_spaces
    assert [] in sub_spaces


def test_deriving_spanning_subspaces():
    """
    Tests the analysis of a Space with dimensions that are only
    used to support the structure of inputs (Composites that have
    nullable=False).

    Asserts
    -------
        The full-sub-spaces are derived to the proper number and
        delinated with the proper dimension specifications
    """
    space = s.Space(
        dimensions=[
            s.Float(id="x1", lb=3.0, ub=5.0, nullable=False),
            s.Float(id="x1-2", lb=3.0, ub=5.0, nullable=False),
            s.Float(
                id="x2", lb=-3.0, ub=-5.0, nullable=True, portion_null=0.33
            ),
            s.Composite(
                id="x3",
                nullable=True,
                portion_null=0.33,
                children=[
                    s.Int(id="x4", lb=6, ub=7, nullable=False),
                    s.Composite(
                        id="x6",  # Note that x6 is ignore since it is
                        # just for the specification's structure
                        nullable=False,
                        children=[
                            s.Float(
                                id="x5",
                                value_set=[0.1, 0.5, 0.9],
                                nullable=True,
                                portion_null=0.33,
                            )
                        ],
                    ),
                ],
            ),
        ]
    )

    subspaces = space.derive_spanning_subspaces()

    assert ["x1", "x1-2"] in subspaces
    assert ["x2"] in subspaces
    assert ["x3", "x4"] in subspaces
    assert ["x5"] in subspaces


def test_deriving_subspaces_from_required_unions():
    """
    Tests the ability to derive full spaces when a
    required Variant dimension is at the root level.

    Asserts
    -------
        The proper number of full subspaces are
        derived
    """
    full_subspaces = SPACE.derive_full_subspaces()

    assert len(full_subspaces) == len(SUB_SPACES)
