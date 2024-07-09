import math
from typing import List, Tuple, Iterable
import raxpy.spaces.dimensions as d
import raxpy.spaces.complexity as c
import raxpy.spaces.root as s


def test_assign_null_portions():
  space = s.Space(
    dimensions=[
      d.Float(
        id="x1",
        lb=3.0,
        ub=5.0,
        nullable=False
      ),
      d.Float(
        id="x2",
        lb=-3.0,
        ub=-5.0,
        nullable=True,
      ),
      d.Composite(
        id="x3",
        nullable=True,
        children=[
          d.Int(
            id="x4",
            lb=6,
            ub=7,
            nullable=False
          ),
          d.Float(
            id="x5",
            value_set=[0.1,0.5,0.9],
            nullable=True,
          ),  
        ]
      )
    ]
  )

  # adjust the porition null values
  c.assign_null_portions(s.create_level_iterable(space.children))

  # an dimension that cannot be null should not portion any nulls in data points
  assert space.dimensions[0].portion_null == 0.0
  # the default heuristic for optional floats 1 out of 10
  assert space.dimensions[1].portion_null == 1.0/10.0
  # the default heuristic for optional composites is to add the complexity of children and add one
  assert space.dimensions[2].portion_null == 1.0/7.0

