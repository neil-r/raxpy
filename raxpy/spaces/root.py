from typing import List, Optional, Dict
from dataclasses import dataclass
import numpy as np

from .dimensions import Dimension


@dataclass
class SubSpace:
    active_dimensions:Dict
    target_sample_count = 0

    pass

def _project_null(x1, x2):
    return [(np.nan if np.isnan(xp2) else xp1) for xp1, xp2 in zip(x1,x2)]

@dataclass
class Space:
    dimensions:List[Dimension]

    def derive_complete_sub_spaces(self, target_sample_count:int):
        sub_spaces = []

        for d in self.dimensions:
            #TODO derive children dimension sub-spaces

            if d.nullable:
                # duplicate sub_spaces
                sub_spaces = sub_spaces + [SubSpace(s).inject() for s in sub_spaces]
                pass
            else:
                pass
        
        return sub_spaces

    def count_dimensions(self):
        return sum([
            1 if not d.has_child_dimensions() else (
                (0 if d.only_supports_spec_structure() else 1) + 
            (d.count_children_dimensions())  ) for d in self.dimensions])

    def collapse(self, x):

        x = np.array(x)

        column_index = 0
        columns = self.dimensions

        while column_index < len(columns):
            active_column = columns[column_index]
            child_dim_count = 0
            if active_column.has_child_dimensions():
                child_dim_count = active_column.count_children_dimensions()
                # inject child columns
                columns = columns[:column_index+1] + active_column.children + columns[column_index+1:]
            print(column_index)
            print("-" * 80)
            x[:,column_index] = active_column.collapse_uniform(x[:,column_index])
            if active_column.portion_null is not None and child_dim_count > 0:
                for i in range(child_dim_count):
                    x[:,i+column_index+1] = _project_null(x[:, column_index], x[:,i+column_index+1])

            column_index += 1
        return x

@dataclass
class InputSpace(Space):
    multi_dim_contraints:Optional[List] = None

@dataclass
class OutputSpace(Space):
    dimensions:List[Dimension]

