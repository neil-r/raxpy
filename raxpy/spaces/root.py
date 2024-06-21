from typing import List, Optional
from dataclasses import dataclass
from .dimensions import Dimension

@dataclass
class Space:
    dimensions:List[Dimension]
    constraints:Optional[List] = None