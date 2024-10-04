"""
    Tests the maxpro-influenced techniques
"""

import numpy as np

import raxpy.spaces as s
from raxpy.does import doe
from raxpy.does.maxpro import optimize_design_with_sa


def test_maxpro_oh_optimization():
    """
    Tests that an unoptimized design is improved with
    maxpro-oh optimization

    Asserts
    -------
        algorithm modifies design from bad design
    """

    space = s.Space(
        dimensions=[
            s.Float(id="x1", lb=0.0, ub=1.0, nullable=False),
            s.Float(
                id="x2", lb=-0.0, ub=1.0, nullable=True, portion_null=0.33
            ),
            s.Composite(
                id="x3",
                nullable=True,
                portion_null=0.33,
                children=[
                    s.Int(id="x3_1", lb=0, ub=1, nullable=False),
                    s.Float(
                        id="x3_2",
                        value_set=[0.1, 0.5, 0.9],
                        nullable=True,
                        portion_null=0.33,
                    ),
                ],
            ),
            s.Variant(
                id="x4",
                nullable=True,
                portion_null=0.33,
                options=[
                    s.Float(id="x4_1", lb=0.0, ub=1.0, nullable=False),
                    s.Float(id="x4_2", lb=0.0, ub=1.0, nullable=False),
                ],
            ),
        ]
    )

    design = doe.DesignOfExperiment(
        input_space=space,
        input_set_map={
            "x1": 0,
            "x2": 1,
            "x3": 2,
            "x3_1": 3,
            "x3_2": 4,
            "x4": 5,
            "x4_1": 6,
            "x4_2": 7,
        },
        input_sets=np.array(
            [  # x1   x2n   x3n  x3_1i  x3_2vn x4   x4_1   x4_2
                [0.0, 0.1, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                [0.1, 0.2, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                [0.2, 0.3, 1.0, 0, np.nan, np.nan, np.nan, np.nan],
                [0.3, 0.4, 1.0, 0, 0.1, np.nan, np.nan, np.nan],
                [0.4, 0.5, 1.0, 1, 0.5, 0, 0.5, np.nan],
                [0.5, 0.6, 1.0, 1, 0.9, 0, 1.0, np.nan],
                [0.6, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                [0.7, np.nan, np.nan, np.nan, np.nan, 1, np.nan, 0.5],
                [0.8, np.nan, 1.0, 1, 0.9, 1, np.nan, 1.0],
            ]
        ),
        encoding=doe.EncodingEnum.NONE,
    )

    opt_design = optimize_design_with_sa(design, design.encoding, maxiter=10)

    assert np.any(design.input_sets != opt_design.input_sets)
