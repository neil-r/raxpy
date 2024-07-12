import raxpy.spaces.dimensions as d
import raxpy.spaces.root as s
import raxpy.does.projected_lhs as doe


def test_creation_of_space_filling_doe():
    space = s.InputSpace(
        dimensions=[
            d.Float(id="x1", lb=3.0, ub=5.0, nullable=False),
            d.Float(id="x2", lb=-3.0, ub=-5.0, nullable=True, portion_null=1.0 / 10.0),
            d.Composite(
                id="x3",
                nullable=True,
                portion_null=1.0 / 7.0,
                children=[
                    d.Int(id="x4", lb=6, ub=7, nullable=False),
                    d.Float(
                        id="x5",
                        value_set=[0.1, 0.5, 0.9],
                        nullable=True,
                        portion_null=1.0 / 4.0,
                    ),
                ],
            ),
        ]
    )

    design = doe.generate_design(space, 100)
    assert design is not None
