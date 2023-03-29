from Engine import *

vec = Vector(Point(1, 2, 3))
#print(vec.norm())

sharik = Sphere(vs.nullpoint, vs.basis1, radius = 2)
#print(sharik.param['radius'])

samolet = Plane(Point(1, 2, 3), vs.basis1)
#print(samolet.contains(vs.nullpoint))

print(samolet.intersect(vec, Point(10, 10, 10)))
