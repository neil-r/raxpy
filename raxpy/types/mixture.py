from dataclasses import dataclass
from . import base

@dataclass
class MixtureComponent(base.Real):
    label: str
    p_label: str

@dataclass
class Mixture:
    label: str
    limit: float = 1.0

    def create_component_meta(self, component_label):
        return MixtureComponent(component_label, self.label)
