from typing import List
from scipy.stats.qmc import LatinHypercube
from ..spaces.dimensions import Dimension


def generate_design(dimesions:List[Dimension],n_points):

  # 
  possible_dimesions = 10
  sampler = LatinHypercube(d=possible_dimesions)

  points = sampler.random(n=n_points)

  # project points to space

  return points