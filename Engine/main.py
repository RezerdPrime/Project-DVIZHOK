from Engine import *

vs = VectorSpace(Point(0, 0, 0), Vector(Point(1, 2, 3)))
Vector.vs = vs  # Передача векторного пространства в класс Вектор

print(Point(10, 9, 8).dist(vs.nullpoint))
print(vs.basis1)
print(Vector(Point(3, 5, 6)) ^ Vector(Point(11, -4, 6)))

