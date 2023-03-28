import configparser
import math


# =================================================================================================== #
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

    def __rmul__(self, arg):
        return self.__mul__(arg)

    def __truediv__(self, arg):

        if isinstance(arg, Point):
            return Point(self.c1 / arg.c1, self.c2 / arg.c2, self.c3 / arg.c3)

        if isinstance(arg, int) or isinstance(arg, float):
            return Point(self.c1 / arg, self.c2 / arg, self.c3 / arg)


# =================================================================================================== #
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


# =================================================================================================== #
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

vs = VectorSpace(Point(0, 0, 0), Vector(Point(1, 2, 3)))
#vs = VectorSpace()
Vector.vs = vs  # Передача векторного пространства в класс Вектор


# =================================================================================================== #
class Camera:

    def __init__(self, pos: Point, look_at: Point, fov, draw_distance):
        self.pos = pos
        self.look_at = look_at
        self.fov = fov
        self.draw_distance = draw_distance

        config = configparser.ConfigParser()
        config.read("config.cfg")
        width = int(config['SCREEN']['width'])
        hight = int(config['SCREEN']['hight'])

        self.Vfov = fov * hight / width

    def sendrays(self, count): # returns list[Vector]
        pass


# =================================================================================================== #
class Object:

    def __init__(self, pos: Point, rotation: Vector, **parameter):
        self.pos = pos
        self.rotation = rotation
        self.param = parameter

    def contains(self, pt: Point): # returns bool
        pass

    def intersect(self, vec: Vector, vec_begin: Point= Vector.vs.nullpoint): # returns point
        pass

    def nearest_point(self, *pts: list[Point]): # returns point
        pass


# =================================================================================================== #
class Plane(Object):

    def contains(self, pt): # A(x - x0) + B(y - y0) + C(z - z0) = 0
        return ( self.rotation.pt.c1 * (pt.c1 - self.pos.c1) +
                 self.rotation.pt.c2 * (pt.c2 - self.pos.c2) +
                 self.rotation.pt.c3 * (pt.c3 - self.pos.c3) == 0 )

    def intersect(self, vec: Vector, vec_begin: Point = Vector.vs.nullpoint): # returns point

        if self.rotation * vec != 0 and not self.contains(vec_begin): # Пересечение в одной точке
            t0 = (self.rotation * Vector(self.pos) - self.rotation * Vector(vec_begin)) / (self.rotation * vec)

            if 0 <= t0 <= 1:
                return t0 * vec.pt + vec_begin

        elif self.contains(vec_begin):

            zentrum_vec = Vector(self.pos - vec_begin)
            projection = vec * zentrum_vec / zentrum_vec.len()

            if 0 <= projection <= 1:
                return vec.pt * projection + vec_begin

            elif projection > 0:
                return vec.pt

            else: return vec_begin

        return Vector.vs.nullpoint

    def nearest_point(self, *pts: Point):
        dist_min = 10 ** 9
        pt_min = Vector.vs.nullpoint

        for pt in pts:
            dist = abs(self.rotation * Vector(pt - self.pos)) / self.rotation.len()

            if dist < dist_min:
                dist_min = dist
                pt_min = pt

        return pt_min


# =================================================================================================== #
class BoundedPlane(Plane):

    def __init__(self, pos: Point, rotation: Vector, **parameter):
        self.pos = pos
        self.rotation = rotation
        self.param = parameter

        a = self.rotation.pt.c1
        b = self.rotation.pt.c2

        self.vec1 = Vector(Point(b, -a, 0)).norm() # направляющие вектора плоскости, лежащие в ней
        self.vec2 = (self.vec1 ^ self.rotation).norm()

    def in_boundaries(self, pt: Point):
        corner = self.vec1 * self.param['delta_v1'] + self.vec2 * self.param['delta_v2']

        delta_x = corner.pt.c1
        delta_y = corner.pt.c2
        delta_z = corner.pt.c3

        return abs(pt.c1 - self.pos.c1) <= delta_x  \
            and abs(pt.c2 - self.pos.c2) <= delta_y  \
             and abs(pt.c3 - self.pos.c3) <= delta_z #\

    def contains(self, pt): # returns bool
        if self.in_boundaries(pt):
            return self.rotation * Vector(pt - self.pos) == 0

    def intersect(self, vec: Vector, vec_begin: Point = Vector.vs.nullpoint): # returns point\

        if self.rotation * vec != 0 and not self.contains(vec.pt):
            t0 = (self.rotation * Vector(self.pos) - self.rotation * Vector(vec_begin)) / (self.rotation * vec)

            if 0 <= t0 <= 1 and self.in_boundaries((vec * t0).point):
                return vec.pt * t0 + vec_begin

        elif self.rotation * Vector(vec.pt - self.pos) == 0:
            rad_vec_1 = Vector(vec_begin - self.pos) # радиус вектор из центра плоскости к началу вектора
            begin_vec1 = rad_vec_1 * self.vec1 / rad_vec_1.len() # проекции на направляющие вектора
            begin_vec2 = rad_vec_1 * self.vec2 / rad_vec_1.len()

            rad_vec_2 = rad_vec_1 + vec  # радиус вектор из центра плоскости к концу вектора
            end_vec1 = rad_vec_2 * self.vec1 / rad_vec_2.len()  # проекции на направляющие вектора
            end_vec2 = rad_vec_2 * self.vec2 / rad_vec_2.len()

            if begin_vec1 > self.param['delta_v1'] and end_vec1 > self.param['delta_v1'] \
               or begin_vec2 > self.param['delta_v2'] \
               and end_vec1 > self.param['delta_v2']: return Vector.vs.nullpoint # пересечения нет

            def limit(value, lim): # Ограничение вектора плоскостью
                if value < -lim: value = -lim
                elif value > lim: value = lim

                return value

            begin_vec1 = limit(begin_vec1, self.param['delta_v1'])
            begin_vec2 = limit(begin_vec2, self.param['delta_v2'])

            end_vec1 = limit(end_vec1, self.param['delta_v1'])
            end_vec2 = limit(end_vec2, self.param['delta_v2'])

            rad_vec_1 = self.vec1 * begin_vec1 + self.vec2 * begin_vec2 + Vector(self.pos)
            rad_vec_2 = self.vec1 * end_vec1 + self.vec2 * end_vec2 + Vector(self.pos)

            result_vec = rad_vec_2 - rad_vec_1 - Vector(vec_begin)

            return Plane(self.pos, self.rotation).intersect(result_vec, rad_vec_1.pt)

        return Vector.vs.nullpoint


# =================================================================================================== #
class Sphere(Object):

    def contains(self, pt):
        return pt.c1 ** 2 + pt.c2 ** 2 + pt.c3 ** 2 <= self.param['radius']

    def intersect(self, vec: Vector, vec_begin: Point = Vector.vs.nullpoint): # returns point
        pass
