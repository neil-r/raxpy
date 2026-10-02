import os
from typing import Annotated

import raxpy
from raxpy.adapters.joblib import perform_experiment


def test_execute_design_persists_and_reuses_results(tmp_path):
    def f(
        x1: Annotated[float, raxpy.Float(lb=0.0, ub=1.0)],
        x2: Annotated[float, raxpy.Float(lb=0.0, ub=1.0)],
    ) -> float:
        return x1 + x2

    design = raxpy.design_experiment(
        f, n_points=3, seed=7, optimize_projections=False
    )
    session_folder = os.fspath(tmp_path / "session")

    _inputs, outputs = perform_experiment(
        design,
        f,
        n_jobs=1,
        session_folder_path=session_folder,
        sub_folders=("x1",),
    )

    result_files = list(tmp_path.glob("session/x1=*/data_point_*.pkl"))
    assert outputs == [
        f(**args)
        for args in design.input_space.convert_flat_values_to_dict(
            design.decoded_input_sets, design.input_set_map
        )
    ]
    assert len(result_files) == 3

    def should_not_run(**kwargs):
        raise AssertionError("completed trial was executed again")

    assert (
        perform_experiment(
            design,
            should_not_run,
            n_jobs=1,
            session_folder_path=session_folder,
            sub_folders=("x1",),
        )[1]
        == outputs
    )
