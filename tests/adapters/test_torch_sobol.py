import pytest
import numpy as np

import raxpy.spaces as s

try:
    import torch.quasirandom
except ImportError:
    pytest.skip(
        "Torch quasirandom module not available, so skipping adapter tests",
        allow_module_level=True,
    )

import raxpy.adapters.torch_sobol as ts


def test_sobol_design_generations():
    input_space = s.InputSpace(
        dimensions=[
            s.Float(id="x1", lb=3.0, ub=5.0),
            s.Float(
                id="x2",
                lb=-3.0,
                ub=-5.0,
                nullable=True,
                portion_null=1.0 / 10.0,
            ),
            s.Composite(
                id="x3",
                nullable=True,
                portion_null=1.0 / 7.0,
                children=[
                    s.Int(id="x4", lb=6, ub=7),
                    s.Float(
                        id="x5",
                        value_set=[0.1, 0.5, 0.9],
                        nullable=True,
                        portion_null=1.0 / 4.0,
                    ),
                ],
            ),
        ]
    )
    doe1 = ts.design_experiment(input_space, n_points=10)

    assert doe1 is not None
    assert doe1.point_count == 10
