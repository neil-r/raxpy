from typing import Annotated, List
import raxpy.annotations as rx

m = rx.Mixture("test")
m1_meta = m.create_component_meta("Juice")
m2_meta = m.create_component_meta("Pop")
m3_meta = m.create_component_meta("Soda")


@rx.spec(constraints=[
    m,
    m1_meta < m2_meta,
    m2_meta < m3_meta
])
def m(
    m1: Annotated[float, m1_meta ],
    m2: Annotated[float, m2_meta ],
    m3: Annotated[float, m3_meta ]
):
    print("INNER")
    return m1 + m2 + m3

print(m(1,2,3))
print(m(4,5,3))
