from dataclasses import dataclass
from . import values


@dataclass
class Mixture:
    label: str
    limit: float = 1.0

    def create_component_meta(self, component_label) -> float:
        return values.Float(component_label, self.label)
