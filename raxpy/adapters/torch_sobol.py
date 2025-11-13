"""
Provides an adapter to use torch to create a sobol design of experiment
"""

import torch.quasirandom as qr
from raxpy.does.doe import DesignOfExperiment
from raxpy.does.lhs import generate_design_with_projection
from raxpy.spaces.root import InputSpace


def design_experiment(
    space: InputSpace, n_points: int, scramble: bool = True, seed=None
) -> DesignOfExperiment:
    """
    Generates a design of experiment using the Sobol technqiue

    Arguments
    ---------
    space: InputSpace
    n_points: int
    scramble:bool=True
    seed=None

    Returns
    -------
        a design of expriment

    """

    def generate_sobol_design(d, n):
        engine = qr.SobolEngine(d, scramble=scramble, seed=seed)

        return engine.draw(n)

    return generate_design_with_projection(
        space=space, n_points=n_points, base_creator=generate_sobol_design
    )
