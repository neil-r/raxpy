from typing import Optional, List, Dict
from dataclasses import dataclass, field
import numpy as np

from raxpy.does.doe import DesignOfExperiment
from raxpy.spaces.dimensions import convert_values_from_dict


@dataclass
class BatchExperimentTaskProvider:
    design: DesignOfExperiment
    _active_point = 0
    _active_tasks: Dict = field(default_factory=dict)
    results: List = field(default_factory=list)

    def process_task_result(self, task, task_result):
        if len(self.results) == 0:
            # Initalize results with placeholder None values
            self.results = [None for _ in range(self.design.point_count)]
        self.results[task["index"]] = task_result
        del self._active_tasks[task["index"]]

    def next(self) -> Optional[object]:

        if self._active_point < self.design.point_count:
            value_dicts = self.design.input_space.convert_flat_values_to_dict(
                np.array([self.design.decoded_input_sets[self._active_point]]),
                self.design.input_set_map,
            )
            t_index = self._active_point

            arg_sets = list(
                convert_values_from_dict(
                    self.design.input_space.dimensions, value_dict
                )
                for value_dict in value_dicts
            )
            task = {
                "index": t_index,
                "input": arg_sets[0],
            }
            self._active_tasks[t_index] = task
            self._active_point += 1

            return task

        return None

    def is_waiting_for_results(self):
        return len(self._active_tasks.values()) > 0
