""" TODO Explain Module"""

import raxpy.spaces.dimensions as d
import raxpy.spaces.root as s
import raxpy.does.lhs as doe


def test_creation_of_space_filling_doe():
    """
    TODO Explain the Function

    Asserts
    -------
    Design is not None

    """
    space = s.InputSpace(
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

    design = doe.generate_design(space, 100)
    assert design is not None


def test_creation_of_space_filling_by_subspaces():
    """
    TODO Explain the Function

    Asserts
    -------
    Design is not None

    """
    space = s.InputSpace(
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
    design = doe.generate_seperate_designs_by_full_subspace(space, 100)

    assert design is not None


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


def test_creation_with_lhs_projection():
    """
    TODO Explain the Function

    Asserts
    -------
    Design is not None
    """
    space = s.InputSpace(
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
    design = doe.generate_design_with_projection(space, 100)

    assert design is not None
