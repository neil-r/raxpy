from dataclasses import dataclass
from typing import Annotated, Optional, Union

'''
 - optional parameters
 - hierarchical parameters
 - combination parameters
 - higher-order interaction parameters
'''
@dataclass
class LinearRegressionTrainer:
    pass

@dataclass
class RandomForestTrainer:
    pass


def train(
    technique_conf:Union[RandomForestTrainer, LinearRegressionTrainer],
    preprocessing_conf
):
    
    pass
