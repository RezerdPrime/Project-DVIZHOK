#from Engine import *
from Movement import *

camer = Camera(Point(-2, 0, 0), Vector(1, 0, 0), 90, 10)
map1 = Map()

b1 = Sphere(Point(2, 0, 0), Vector(0, 0, 0), 0.8)
b2 = Cube(Point(2, 0, -2), Vector(1, 0.1, 0), 1)
b3 = Plane(Point(0, -2, 0), Vector(0, 1, 0))
b4 = BoundedPlane(Point(2, 0, 2), Vector(0.75, 1, 0), 1, 1)

map1.append(b1, b2, b3, b4)
cons = Console(map1, camer)
cons.draw()

launch(cons)
