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
            portion_null=0.33
          ),  
        ]
      )
    ]
  )
  assert space.count_dimensions() == 5

  flatted_dimensions = s.create_all_iterable(space.children)
  dim_map = {}
  for i, dim in enumerate(flatted_dimensions):
    dim_map[dim.id] = i
  random_x = np.random.rand(10, 5)
  collapsed_x = space.decode_zero_one_matrix(random_x, dim_map,map_null_to_children_dim=True)

  assert collapsed_x is not None

  x3_values = collapsed_x[:,dim_map["x3"]]
  x4_values = collapsed_x[:,dim_map["x4"]]
  # make sure if x3 is null, x4 is also null
  assert np.all(np.isnan(x3_values) == np.isnan(x4_values))
