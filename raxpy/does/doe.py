from typing import Dict
from dataclasses import dataclass
import numpy as np


@dataclass
class DesignOfExperiment:
  input_sets:np.array
  input_set_map:Dict[str,int]
  encoded_flag:bool = False
