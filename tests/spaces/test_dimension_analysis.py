""" TODO Explain Module """

import raxpy.spaces.dimensions as d
import raxpy.spaces.root as s


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
            d.Float(id="x1", lb=3.0, ub=5.0, nullable=False),
            d.Float(
                id="x2", lb=-3.0, ub=-5.0, nullable=True, portion_null=0.33
            ),
            d.Composite(
                id="x3",
                nullable=True,
                portion_null=0.33,
                children=[
                    d.Int(id="x4", lb=6, ub=7, nullable=False),
                    d.Composite(
                        # Note that x6 is ignore since it is
                        # just for the specification's structure
                        id="x6",
                        nullable=False,
                        children=[
                            d.Float(
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
    TODO Explain the Function

    Asserts
    -------
    **Explanation**

    """

    space = s.Space(
        dimensions=[
            d.Variant(
                id="xb",
                nullable=True,
                portion_null=0.33,
                options=[
                    d.Float(id="x1", lb=1.0, ub=2.0, nullable=False),
                    d.Float(id="x2", lb=3.0, ub=4.0, nullable=False),
                    d.Float(id="x3", lb=5.0, ub=6.0, nullable=False),
                ],
            ),
        ]
    )

    sub_spaces = space.derive_full_subspaces()

    assert sub_spaces is not None

    assert ["xb", "x1"] in sub_spaces
    assert ["xb", "x2"] in sub_spaces
    assert ["xb", "x3"] in sub_spaces
    assert [] in sub_spaces


def test_deriving_spanning_subspaces():
    """
    TODO Explain the Function

    Asserts
    -------
    **Explanation**
    """
    space = s.Space(
        dimensions=[
            d.Float(id="x1", lb=3.0, ub=5.0, nullable=False),
            d.Float(id="x1-2", lb=3.0, ub=5.0, nullable=False),
            d.Float(
                id="x2", lb=-3.0, ub=-5.0, nullable=True, portion_null=0.33
            ),
            d.Composite(
                id="x3",
                nullable=True,
                portion_null=0.33,
                children=[
                    d.Int(id="x4", lb=6, ub=7, nullable=False),
                    d.Composite(
                        id="x6",  # Note that x6 is ignore since it is
                        # just for the specification's structure
                        nullable=False,
                        children=[
                            d.Float(
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

    print(subspaces)
