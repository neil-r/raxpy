from typing import List, Optional
from dataclasses import dataclass
from .dimensions import Dimension


@dataclass
class InputSpace:
    dimensions:List[Dimension]
    multi_dim_contraints:Optional[List] = None

@dataclass
class OutputSpace:
    dimensions:List[Dimension]

