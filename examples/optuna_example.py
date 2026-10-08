from typing import Annotated

import optuna
import raxpy
import raxpy.adapters.optuna as optuna_adapter


def f(
    x1: Annotated[float, raxpy.Float(lb=-10.0, ub=10.0)],
    x2: Annotated[float, raxpy.Float(lb=-10.0, ub=10.0)],
):
    return (x1 - 2) ** 2 + (x2 - 3) ** 2


# create an Optuna-compatible function from the annotated function
optuna_f = optuna_adapter.convert_to_optuna(f)

s = optuna.create_study(direction="minimize")
s.optimize(optuna_f, n_trials=100)
