import numpy as np

import raxpy.spaces.dimensions as d
import raxpy.spaces.root as s

def test_collapse_of_random_numbers():
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
        portion_null=0.33
      ),
      d.Composite(
        id="x3",
        nullable=True,
        portion_null=0.33,
        children=[
          d.Float(
            id="x4",
            lb=6.0,
            ub=7.0,
            nullable=False
          ),
          d.Float(
            id="x5",
            value_set=[0.1,0.5,0.9],
            nullable=True,
            portion_null=0.33
          ),  
        ]
      )
    ]
  )
  assert space.count_dimensions() == 5
  random_x = np.random.rand(10, 5)

  collapsed_x = space.collapse(random_x)

  assert collapsed_x is not None

  
