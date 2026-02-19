"""
Unit tests for converting a Space to and from a JSON friendly dict object
"""

import raxpy.spaces as s


def test_convert_to_json():

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

    json_dict = space.to_json_dict()
    assert isinstance(json_dict, dict)

    assert json_dict["dimensions"][2]["children"][1]["id"] == "x6"
