"""
Tests for counting dimensions in spaces.
"""
import raxpy.spaces as spaces


def test_shared_dim_counting():
    """
    Test that shared dimensions with the same id are counted as one dimension.
    """

    # first test with x2 as the same dimension in both options of the variant
    input_space_1 = spaces.InputSpace(
        dimensions=[
            spaces.Variant(
                id="x1",
                options=[
                    spaces.Composite(
                        id="x1_1",
                        children=[
                            spaces.Int(
                                id="x2",
                                lb=0,
                                ub=10,
                            )
                        ],
                    ),
                    spaces.Int(
                        id="x2",
                        lb=0,
                        ub=10,
                    )
                ],
            ),
            spaces.Float(id="x3", lb=0.0, ub=1.0)
        ]
    )

    assert input_space_1.count_dimensions() == 3

    # now test with x2 as different dimensions in both options of the variant
    input_space_2 = spaces.InputSpace(
        dimensions=[
            spaces.Variant(
                id="x1",
                options=[
                    spaces.Composite(
                        id="x1_1",
                        children=[
                            spaces.Int(
                                id="x2_1",
                                lb=0,
                                ub=10,
                            )
                        ],
                    ),
                    spaces.Int(
                        id="x2_2",
                        lb=0,
                        ub=10,
                    )
                ],
            ),
            spaces.Float(id="x3", lb=0.0, ub=1.0)
        ]
    )

    assert input_space_2.count_dimensions() == 4