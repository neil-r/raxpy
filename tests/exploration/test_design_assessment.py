import numpy as np

import raxpy.does.assess as a
import raxpy.spaces.root as s
import raxpy.spaces.dimensions as d
import raxpy.does.doe as doe


def test_doe_assessments():
    space = s.InputSpace(
        dimensions=[
            d.Float(id="x1", lb=0.0, ub=1.0, nullable=True),
            d.Float(id="x2", lb=0.0, ub=1.0, nullable=True),
            d.Float(id="x3", lb=0.0, ub=1.0, nullable=True),
        ]
    )

    design = doe.DesignOfExperiment(
        input_set_map={"x1": 0, "x2": 1, "x3": 2},
        input_sets=np.array(
            [
                [np.nan, 0.1, 0.1],
                [np.nan, 0.1, 0.2],
                [np.nan, np.nan, 0.1],
                [0.3, 0.1, 0.7],
                [0.4, 0.2, 0.8],
                [0.5, 0.3, 0.9],
                [0.6, 0.4, 0.6],
                [np.nan, np.nan, np.nan],
            ]
        ),
        encoded_flag=False,
    )

    assessment = a.assess(space, design)

    assert assessment is not None
