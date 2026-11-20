from typing import Annotated, Optional
from dataclasses import dataclass

from ax.api.client import Client
from ax.generation_strategy.generation_strategy import GenerationStrategy

import raxpy
import raxpy.adapters.ax as ax_adapter
from raxpy.spaces import dim_tags


@dataclass
class Branch:
    n1: Annotated[float, raxpy.Float(lb=-10.0, ub=10.0)] = 0.0
    n2: Annotated[float, raxpy.Float(lb=-10.0, ub=10.0)] = 0.0

    @property
    def value(self):
        return self.n1 - self.n2 + (self.n1 * self.n2)


def f(
    x1: Annotated[float, raxpy.Float(lb=-10.0, ub=10.0)],
    x2: Annotated[float, raxpy.Float(lb=-10.0, ub=10.0)],
    x3: Optional[Branch],
) -> Annotated[float, raxpy.Float(id="y", tags=[dim_tags.MAXIMIZE])]:
    return (
        (x1 - 3) * x2
        + x1
        + x1 * (x2 - 3) * (x3.value if x3 is not None else 1.0)
    )


if __name__ == "__main__":
    doe = raxpy.design_experiment(f, n_points=25)

    ax_f, conf_experiment, conf_optimization = ax_adapter.convert_to_ax(f)
    ax_points = ax_adapter.convert_doe_to_ax_points(doe)

    client = Client()

    client.configure_experiment(parameters=conf_experiment["parameters"])
    client.configure_optimization(objective=conf_optimization["objective"])

    gs = GenerationStrategy(
        name="ExplicitPoints+BoTorch",
        nodes=[
            ax_adapter.ExplicitPointsGenerationNode(
                points=ax_points,
                name="raxpy_exploration_points",
                transition_criteria=None,
            ),
        ],
    )

    client.set_generation_strategy(gs)

    for _ in range(25):  # Run 25 rounds of exploration trials
        trials = client.get_next_trials(max_trials=1)

        for trial_index, parameters in trials.items():
            raw_data = ax_f(parameters)

            client.complete_trial(trial_index=trial_index, raw_data=raw_data)
