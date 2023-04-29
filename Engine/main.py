from Engine import *

'''vec = Vector(Point(1, 2, 3))
#print(vec.norm())

sharik = Sphere(vs.nullpoint, vs.basis1, 2)

samolet = Plane(Point(1, 2, 3), vs.basis1)
print(samolet.contains(vs.nullpoint))

print(samolet.intersect(Ray(Point(10, 10, 10), vec)))

print(samolet.pos)
samolet.param.move(Point(100,100,100))
samolet.upd()

print(samolet.pos)

print(samolet.rotation)
samolet.param.rotate(0, 45, 0)
samolet.upd()

print(samolet.rotation)'''

#Vector.vs = VectorSpace(Point(0, 0, 0), Vector(1, 1, 1))

cam = Camera(Point(0, 0, -5), Vector(0, 0, 1), 1, 50)
map1 = Map()

a0 = Sphere(Point(0, 0, -2), Vector(0, 0, 1), 0.4)
a1 = Sphere(Point(0.25, 0.25, 16), Vector(0, 0, 1), 0.3)
a2 = Sphere(Point(0.5, 0.5, 32), Vector(0, 0, 1), 0.2)

b = Plane(Point(0, 0, -4), Vector(1, 1, 1))
b.param.rotate(0, 90, 0)
b.upd()

map1.append(a0, a1, a2, b)

cons = Console(map1, cam)
cons.draw()

