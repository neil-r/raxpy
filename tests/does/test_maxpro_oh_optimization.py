"""
Tests the maxpro-influenced techniques
"""

import numpy as np

import raxpy.spaces as s
from raxpy.does import doe
from raxpy.does.maxpro import optimize_design_with_sa
from raxpy.does import measure


def test_maxpro_oh_optimization():
    """
    Tests that an unoptimized design is improved with
    maxpro-oh optimization

    Asserts
    -------
        algorithm modifies design from bad design
        algorithm improves design according to maxpro-oh criterion
    """
    gen = np.random.default_rng(seed=0)  # seed for reproducibility

    space = s.Space(
        dimensions=[
            s.Float(id="x1", lb=0.0, ub=10.0, nullable=False),
            s.Float(
                id="x2", lb=-0.0, ub=6.0, nullable=True, portion_null=0.33
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
                [0.0, 1.0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                [1.0, 2.0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                [2.0, 3.0, 1.0, 0, np.nan, np.nan, np.nan, np.nan],
                [3.0, 4.0, 1.0, 0, 0.1, np.nan, np.nan, np.nan],
                [4.0, 5.0, 1.0, 1, 0.5, 0, 0.5, np.nan],
                [5.0, 6.0, 1.0, 1, 0.9, 0, 1.0, np.nan],
                [6.0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                [7.0, np.nan, np.nan, np.nan, np.nan, 1, np.nan, 0.5],
                [8.0, np.nan, 1.0, 1, 0.9, 1, np.nan, 1.0],
            ]
        ),
        encoding=doe.EncodingEnum.NONE,
    )

    opt_design = optimize_design_with_sa(
        design,
        # we generally want to optimize the projections with the encoded values
        # not the decoded values (Decoded values could be on different
        # unit-of-measurement scales)
        encoding=doe.EncodingEnum.ZERO_ONE_NULL_ENCODING,
        maxiter=10,
        rng=gen,
    )

    if opt_design is None:
        # some implementations modify the design in-place and return None
        opt_design = design

    assert np.any(design.input_sets != opt_design.input_sets)
    assert measure.compute_max_pro(opt_design) < measure.compute_max_pro(
        design
    )

    # Ensure first dimenion's values are not changed but just moved around.
    # This also provides a sanity check that encoding and decoding process is
    # working correctly (the optimization is ).
    for org_value in design.decoded_input_sets[:, design.input_set_map["x1"]]:
        assert (
            org_value
            in opt_design.decoded_input_sets[:, opt_design.input_set_map["x1"]]
        )
