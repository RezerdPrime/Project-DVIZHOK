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

        if isinstance(arg, int) or isinstance(arg, float):
            return Point(self.c1 * arg, self.c2 * arg, self.c3 * arg)

    def __truediv__(self, arg):

        if isinstance(arg, Point):
            return Point(self.c1 / arg.c1, self.c2 / arg.c2, self.c3 / arg.c3)

        if isinstance(arg, int) or isinstance(arg, float):
            return Point(self.c1 / arg, self.c2 / arg, self.c3 / arg)


class Vector:

    def __init__(self, pt):  # Инициализация вектора
        assert isinstance(pt, Point), "Only points dude"
        self.pt = pt

    def len(self):  # Вычисление длины вектора
        return self.vs.nullpoint.dist(self.pt)  #self.pt.dist(VectorSpace.nullpoint)

    def __str__(self):  # Настройка фомата вывода
        return "{0}".format(self.pt)

    def __add__(self, vec):  # Перегрузка операторов
        return Vector(self.pt + vec.pt)

    def __sub__(self, vec):
        return Vector(self.pt - vec.pt)

    def __mul__(self, arg):

        if isinstance(arg, Vector):  # Скалярное произведение
            return (self.pt.c1 * arg.pt.c1) + (self.pt.c2 * arg.pt.c2) + (self.pt.c3 * arg.pt.c3)

        if isinstance(arg, int) or isinstance(arg, float):  # Произведение вектора на число
            return Vector(Point(self.pt.c1 * arg, self.pt.c2 * arg, self.pt.c3 * arg))

    def __rmul__(self, num):  # Реализация коммутативности умножения
        return Vector(Point(self.pt.c1 * num, self.pt.c2 * num, self.pt.c3 * num))

    def __xor__(self, vec):  # Векторное произведение
        d1 = self.vs.basis1 * (self.pt.c2 * vec.pt.c3 - self.pt.c3 * vec.pt.c2)
        d2 = self.vs.basis2 * (self.pt.c1 * vec.pt.c3 - self.pt.c3 * vec.pt.c1) * -1
        d3 = self.vs.basis3 * (self.pt.c1 * vec.pt.c2 - self.pt.c2 * vec.pt.c1)
        return d1 + d2 + d3

    def __truediv__(self, num):
        return Vector(self.pt / num)

    def norm(self):
        return Vector(self.pt / self.len())


class VectorSpace:
    nullpoint = Point(0, 0, 0)          # Задание дефолтных значений
    basis1 = Vector(Point(1, 0, 0))     # атрибутов класса
    basis2 = Vector(Point(0, 1, 0))
    basis3 = Vector(Point(0, 0, 1))

    def __init__(self, init_pt: Point = nullpoint,
                 dir1: Vector = basis1,
                 dir2: Vector = basis2,
                 dir3: Vector = basis3):

        VectorSpace.nullpoint = init_pt  # Инициализация векторного пространства
        VectorSpace.basis1 = dir1
        VectorSpace.basis2 = dir2
        VectorSpace.basis3 = dir3


class Camera:

    def __init__(self, pos: Point, look_at: Point, fov, draw_distance):
        self.pos = pos
        self.look_at = look_at
        self.fov = fov
        # vfov = fov * (h / w)
        self.draw_distance = draw_distance


class Object:

    def __init__(self, pos, rotation):
        self.pos = pos
        self.rotation = rotation

    """
    def contains(self, pt):
        if f(x, y, z) == pt: return True
        else: return False
    """
