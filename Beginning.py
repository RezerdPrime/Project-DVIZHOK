class Point:

    def __init__(self, c1, c2, c3):  # Инициализация точки
        self.c1 = c1
        self.c2 = c2
        self.c3 = c3

    def dist(self, pt):  # Измерение расстояния между точками
        return ((self.c1 - pt.c1) ** 2 + (self.c2 - pt.c2) ** 2 + (self.c3 - pt.c3) ** 2) ** 0.5

    def __str__(self):  # Настройка фомата вывода
        return "({0},{1},{2})".format(self.c1, self.c2, self.c3)

    def __add__(self, pt):  # Перегрузка операторов
        return Point(self.c1 + pt.c1, self.c2 + pt.c2, self.c3 + pt.c3)

    def __sub__(self, pt):
        return Point(self.c1 - pt.c1, self.c2 - pt.c2, self.c3 - pt.c3)

    def __mul__(self, arg):

        if isinstance(arg, Point):
            return Point(self.c1 * arg.c1, self.c2 * arg.c2, self.c3 * arg.c3)

        if isinstance(arg, int):
            return Point(self.c1 * arg, self.c2 * arg, self.c3 * arg)

    def __truediv__(self, pt):
        return Point(self.c1 / pt.c1, self.c2 / pt.c2, self.c3 / pt.c3)


class VectorSpace:
    nullpoint = Point(0, 0, 0)  # Задание дефолтных значений
    basis1 = Point(1, 0, 0)     # атрибутов класса
    basis2 = Point(0, 1, 0)
    basis3 = Point(0, 0, 1)

    def __init__(self, init_pt=nullpoint, dir1=basis1, dir2=basis2, dir3=basis3):
        nullpoint = init_pt  # Инициализация векторного пространства
        basis1 = dir1
        basis2 = dir2
        basis3 = dir3


class Vector:

    def __init__(self, pt1):  # Инициализация вектора
        assert isinstance(pt1, Point), "Only points dude"
        self.pt1 = pt1

    def len(self):  # Вычисление длины вектора
        return self.pt1.dist(VectorSpace.nullpoint)

    def __str__(self):  # Настройка фомата вывода
        return "({0})".format(self.pt1)

    def __add__(self, vec):  # Перегрузка операторов
        return Vector(self.pt1 + vec.pt1)

    def __sub__(self, vec):
        return Vector(self.pt1 - vec.pt1)

    def __mul__(self, arg):

        if isinstance(arg, Vector):  # Скалярное произведение
            return (self.pt1.c1 * arg.pt1.c1) + (self.pt1.c2 * arg.pt1.c2) + (self.pt1.c3 * arg.pt1.c3)

        if isinstance(arg, int):  # Произведение вектора на число
            return Vector(Point(self.pt1.c1 * arg, self.pt1.c2 * arg, self.pt1.c3 * arg))

    def __xor__(self, vec):  # Векторное произведение
        d1 = VectorSpace.basis1.dist(VectorSpace.nullpoint) * (self.pt1.c2 * vec.pt1.c3 - self.pt1.c3 * vec.pt1.c2)
        d2 = VectorSpace.basis2.dist(VectorSpace.nullpoint) * (self.pt1.c1 * vec.pt1.c3 - self.pt1.c3 * vec.pt1.c1) * -1
        d3 = VectorSpace.basis3.dist(VectorSpace.nullpoint) * (self.pt1.c1 * vec.pt1.c2 - self.pt1.c2 * vec.pt1.c1)
        return Vector(Point(d1, d2, d3))


class Camera:
    height = 50
    weight = 70
    
    def __init__(self, pos: Point, look_at: Point, fov, draw_dostance):
        self.pos = pos
        self.look_at = look_at
        self.fov = fov
        vfov = fov * (Camera.height / Camera.weight)
        self.draw_dostance = draw_dostance


class Object:

    def __init__(self, pos, rotation):
        self.pos = pos
        self.rotation = rotation

    """
    def contains(self, pt):
        if f(x, y, z) == pt: return True
        else: return False
    """


VS = VectorSpace()
A = Vector(Point(2, 0, 0))
B = Vector(Point(0, 0, 2))
print(A ^ B)
