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

    def __neg__(self):
        return Point(-self.c1, -self.c2, -self.c3)


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

    def __neg__(self):
        return self * -1

    def norm(self):
        if self.len() != 0: return Vector(self.pt / self.len())
        else: return self


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
class Parameters:

    def __init__(self, pos: Point, rotation: Vector):
        self.pos = pos
        self.rotation = rotation

    def move(self, pt):
        self.pos = self.pos + pt

    def rotate(self, c1_angle, c2_angle, c3_angle):

        # сделать перевод в радианы

        c1_old = self.rotation.pt.c1
        c2_old = self.rotation.pt.c2
        c3_old = self.rotation.pt.c3

        self.rotation.pt.c2 = c2_old * math.cos(c1_angle) - c3_old * math.sin(c1_angle)
        self.rotation.pt.c3 = c2_old * math.sin(c1_angle) + c3_old * math.cos(c1_angle)

        self.rotation.pt.c1 = c1_old * math.cos(c2_angle) + c3_old * math.sin(c2_angle)
        self.rotation.pt.c3 = -c1_old * math.sin(c2_angle) + c3_old * math.cos(c2_angle)

        self.rotation.pt.c1 = c1_old * math.cos(c3_angle) - c2_old * math.sin(c3_angle)
        self.rotation.pt.c2 = c1_old * math.sin(c3_angle) + c2_old * math.cos(c3_angle)

    def scale(self, scaling_value):
        pass


# =================================================================================================== #
class ParametersPlane(Parameters):
    def __init__(self, pos: Point, rotation: Vector):
        self.pos = pos
        self.rotation = rotation


# =================================================================================================== #
class ParametersBoundedPlane(Parameters):
    def __init__(self, pos: Point, rotation: Vector, v, u, dv, du):
        self.pos = pos
        self.rotation = rotation
        self.v = v
        self.u = u
        self.dv = dv
        self.du = du

    def scale(self, scaling_value):
        self.dv *= scaling_value
        self.du *= scaling_value


# =================================================================================================== #
class ParametersSphere(Parameters):
    def __init__(self, pos: Point, rotation: Vector, *, radius):
        self.pos = pos
        self.rotation = rotation
        self.rd = radius

    def scale(self, scaling_value):
        self.rd *= scaling_value


# =================================================================================================== #
class ParametersCube(Parameters):
    def __init__(self, pos: Point, limit, rotations: [Vector], edges: '[BoundedPlane]'):
        self.pos = pos
        self.rot = rotations[0]
        self.rot2 = rotations[1]
        self.rot3 = rotations[2]
        self.limit = limit
        self.edges = edges

    def move(self, pt):
        self.pos = self.pos + pt

        for edge in self.edges:
            edge.pos = edge.pos + pt

    def scale(self, scaling_value):
        self.rot = self.rot * scaling_value
        self.rot2 = self.rot2 * scaling_value
        self.rot3 = self.rot3 * scaling_value
        array_rots = [self.rot, self.rot2, self.rot3]
        self.limit *= scaling_value

        for i, edge in enumerate(self.edges):
            edge.param.scale(scaling_value)

            if i % 2 == 0: edge.pos = self.pos + array_rots[i // 2].pt
            else: edge.pos = self.pos - array_rots[i // 2].pt

    #def rotate(self, c1_angle, c2_angle, c3_angle):


# =================================================================================================== #
class Map:

    def __init__(self):
        self.list = []

    def append(self, *obj):
        self.list.extend(obj)

    def __getitem__(self, index):
        return self.list[index]

    def __iter__(self):
        return iter(self.list)


# =================================================================================================== #
class Ray:

    def __init__(self, bpt: Point, direct: Vector):
        self.bpt = bpt
        self.direct = direct

    def intersect(self, mappy: Map):
        return [iobj.intersect(self) for iobj in mappy]


# =================================================================================================== #
class Object:

    def __init__(self, pos: Point, rotation: Vector, *parameter):
        self.pos = pos
        self.rotation = rotation
        self.param = parameter

    def contains(self, pt: Point): # returns bool
        pass

    def intersect(self, ray: Ray): # returns float
        pass

    def nearest_point(self, *pts: list[Point]): # returns point
        pass


# =================================================================================================== #
class Plane(Object):

    def __init__(self, pos: Point, rotation: Vector):

        self.param = ParametersPlane(pos, rotation)
        self.pos = pos
        self.rotation = rotation

    def upd(self):
        self.pos = self.param.pos
        self.rotation = self.param.rotation

    def contains(self, pt): # A(x - x0) + B(y - y0) + C(z - z0) = 0
        self.upd()
        return ( self.rotation.pt.c1 * (pt.c1 - self.pos.c1) +
                 self.rotation.pt.c2 * (pt.c2 - self.pos.c2) +
                 self.rotation.pt.c3 * (pt.c3 - self.pos.c3) == 0 )

    def intersect(self, ray: Ray): # returns float
        self.upd()

        if self.rotation * ray.direct != 0 and not self.contains(ray.bpt): # Пересечение в одной точке
            t0 = (self.rotation * Vector(self.pos) - self.rotation * Vector(ray.bpt)) / (self.rotation * ray.direct)

            if 0 <= t0:
                #return (t0 * ray.direct.pt + ray.bpt).dist(ray.bpt)
                return t0 * ray.direct.len()

        elif self.contains(ray.bpt): return 0

        #return Vector.vs.nullpoint

    def nearest_point(self, *pts: Point):
        self.upd()
        dist_min = 2 ** 63 - 1
        pt_min = Vector.vs.nullpoint

        for pt in pts:
            dist = abs(self.rotation * Vector(pt - self.pos)) / self.rotation.len()

            if dist < dist_min:
                dist_min = dist
                pt_min = pt

        return pt_min


# =================================================================================================== #
class BoundedPlane(Plane):

    # def __init__(self, pos: Point, rotation: Vector, *, dv1, dv2):
    #     self.pos = pos
    #     self.rotation = rotation
    #     self.dv1 = dv1
    #     self.dv2 = dv2
    #
    #     a = self.rotation.pt.c1
    #     b = self.rotation.pt.c2
    #
    #     self.vec1 = Vector(Point(b, -a, 0)).norm() # направляющие вектора плоскости, лежащие в ней
    #     self.vec2 = (self.vec1 ^ self.rotation).norm()

    def __init__(self, pos: Point, rotation: Vector, dv, du):

        self.dv = dv
        self.du = du

        if abs(rotation.pt.c1) < abs(rotation.pt.c2):
            help_vec = Vector.vs.basis1
        else: help_vec = Vector.vs.basis2

        self.u = (self.rotation ^ help_vec).norm()
        self.v = (self.rotation ^ self.u).norm()

        self.param = ParametersBoundedPlane(pos, rotation, self.u, self.v, self.du, self.dv)

        self.pos = self.param.pos
        self.rotation = self.param.rotation
        self.u = self.param.u
        self.v = self.param.v
        self.dv = self.param.dv
        self.du = self.param.du

    def upd(self):
        self.pos = self.param.pos
        self.rotation = self.param.rotation
        self.u = self.param.u
        self.v = self.param.v
        self.dv = self.param.dv
        self.du = self.param.du

    def in_boundaries(self, pt: Point):
        self.upd()

        corner = self.u * self.du + self.v * self.dv

        delta_x = corner.pt.c1
        delta_y = corner.pt.c2
        delta_z = corner.pt.c3

        return abs(pt.c1 - self.pos.c1) <= delta_x  \
            and abs(pt.c2 - self.pos.c2) <= delta_y  \
             and abs(pt.c3 - self.pos.c3) <= delta_z #\

    def contains(self, pt): # returns bool
        self.upd()
        if self.in_boundaries(pt):
            return self.rotation * Vector(pt - self.pos) == 0

        return False

    def intersect(self, ray: Ray): # returns float\
        self.upd()

        if self.rotation * ray.direct != 0 and not self.contains(ray.direct.pt):
            t0 = (self.rotation * Vector(self.pos) - self.rotation * Vector(ray.bpt)) / (self.rotation * ray.direct)
            cond_pt = ray.direct.pt * t0 + ray.bpt

            if 0 <= t0 and self.in_boundaries(cond_pt):
                return cond_pt.dist(ray.bpt)

        elif self.rotation * Vector(ray.direct.pt - self.pos) == 0 \
                and self.rotation * Vector(ray.bpt - self.pos):

            rad_vec_1 = Vector(ray.bpt - self.pos) # радиус вектор из центра плоскости к началу вектора
            if rad_vec_1.len() == 0: return 0

            begin_vec1 = rad_vec_1 * self.u * self.du / rad_vec_1.len() # проекции на направляющие вектора
            begin_vec2 = rad_vec_1 * self.v * self.dv / rad_vec_1.len()

            if abs(begin_vec1) <= 1 and abs(begin_vec2) <= 1: return 0

            rad_vec_2 = rad_vec_1 + ray.direct  # радиус вектор из центра плоскости к концу вектора

            if rad_vec_2.len() == 0:

                if abs(begin_vec1) > 1 and abs(begin_vec2) > 1:

                    if begin_vec1 > 1: begin_vec1 -= 1
                    elif begin_vec1 < 1: begin_vec1 += 1

                    if begin_vec2 > 1: begin_vec2 -= 1
                    elif begin_vec2 < 1: begin_vec2 += 1

                    return (begin_vec1 * self.du * self.u + begin_vec2 * self.dv * self.v).len()

            end_vec1 = rad_vec_2 * self.u * self.du / rad_vec_2.len()  # проекции на направляющие вектора
            end_vec2 = rad_vec_2 * self.v * self.dv / rad_vec_2.len()

            def solution(ray1: Ray, ray2: Ray):
                pass


            if begin_vec1 > 1:
                if self.u * ray.direct == 0:
                    return

            elif begin_vec1 < -1:
                if self.u * ray.direct == 0:
                    return

            elif begin_vec2 > 1:
                if self.v * ray.direct == 0:
                    return

            elif begin_vec2 < -1:
                if self.v * ray.direct == 0:
                    return

            '''
            def limit(value, lim): # Ограничение вектора плоскостью
                if value < -lim: value = -lim
                elif value > lim: value = lim

                return value

            begin_vec1 = limit(begin_vec1, self.du)
            begin_vec2 = limit(begin_vec2, self.dv)

            end_vec1 = limit(end_vec1, self.du)
            end_vec2 = limit(end_vec2, self.dv)

            rad_vec_1 = self.u * begin_vec1 + self.v * begin_vec2 + Vector(self.pos)
            rad_vec_2 = self.u * end_vec1 + self.v * end_vec2 + Vector(self.pos)

            result_vec = rad_vec_2 - rad_vec_1 - Vector(ray.bpt)

            return Plane(self.pos, self.rotation).intersect(result_vec, rad_vec_1.pt)

        return Vector.vs.nullpoint '''

    def nearest_point(self, *pts: Point):
        self.upd()
        dist_min = 2 ** 63 - 1
        pt_min = Vector.vs.nullpoint
        dist = 0

        for pt in pts:
            rad_vec = Vector(pt - self.pos)

            if rad_vec.len() == 0: return pt

            proj1 = rad_vec * self.rotation / rad_vec.len()
            proj2 = rad_vec * self.u * self.du / rad_vec.len()
            proj3 = rad_vec * self.v * self.dv / rad_vec.len()

            sign = lambda x: 1 if x > 0 else -1

            if abs(proj2) <= 1 and abs(proj3) <= 1:
                dist = proj1 * self.rotation.len()

            elif abs(proj2) > 1 and abs(proj3) > 1:
                new_pr2 = proj2 - sign(proj2)
                new_pr3 = proj3 - sign(proj3)

                dist = self.rotation * -proj1 + self.u * new_pr2 + self.v * new_pr3 + Vector(pt)
                dist = dist.len()

            elif abs(proj2) > 1:
                new_pr2 = proj2 - sign(proj2)

                dist = self.rotation * -proj1 + self.u * new_pr2 + Vector(pt)
                dist = dist.len()

            elif abs(proj3) > 1:
                new_pr3 = proj3 - sign(proj3)

                dist = self.rotation * -proj1 + self.v * new_pr3 + Vector(pt)
                dist = dist.len()

            if dist < dist_min:
                dist_min = dist
                pt_min = pt

        return pt_min


# =================================================================================================== #
class Sphere(Object):

    def __init__(self, pos: Point, rotation: Vector, *, radius):
        self.rd = radius
        self.rotation = rotation.norm() * self.rd

    def contains(self, pt):
        return pt.c1 ** 2 + pt.c2 ** 2 + pt.c3 ** 2 <= self.rd

    def intersect(self, vec: Vector, vec_begin: Point = Vector.vs.nullpoint): # returns point

        a = vec * vec
        b = 2 * vec * Vector(vec_begin - self.pos)
        c = Vector(self.pos) * Vector(self.pos) + \
            Vector(vec_begin) * Vector(vec_begin) - \
            2 * Vector(self.pos) * Vector(vec_begin) - self.rd ** 2

        D = b ** 2 - 4 * a * c

        if D > 0:
            t1 = (-b + D ** 0.5) / (2 * a)
            t2 = (-b - D ** 0.5) / (2 * a)

            if 0 <= t1 <= 1: return vec.pt * t1 + vec_begin
            elif 0 <= t2 <= 1: return vec.pt * t2 + vec_begin

            if (0 <= t1) != (0 <= t2) and (t1 <= 1) != (t2 <= 1):
                rad_vec = Vector(self.pos - vec_begin)

                if rad_vec.len() == 0: return self.pos
                proj = vec * rad_vec / rad_vec.len()

                if 0 <= proj <= 1: return proj * vec.pt + vec_begin
                elif proj > 1: return vec.pt
                else: return vec_begin

        elif D == 0:
            t0 = -b / (2 * a)

            if 0 <= t0 <= 1: return vec.pt * t0 + vec_begin

        return Vector.vs.nullpoint


    def nearest_point(self, *pts: Point):
        dist_min = 2 ** 63 - 1
        pt_min = Vector.vs.nullpoint

        for pt in pts:
            dist = self.pos.dist(pt)

            if dist < dist_min:
                dist_min = dist
                pt_min = pt

        return pt_min


# =================================================================================================== #
class Cube(Object):

    def __init__(self, pos: Point, rotation: Vector):
        self.rot = rotation
        self.pos = pos
        self.limit = self.rot.len()
        a = self.rot.pt.c1
        b = self.rot.pt.c2

        self.rot2 = Vector(Point(b, -a, 0)).norm() * self.limit
        self.rot3 = (self.rot2 ^ self.rot).norm() * self.limit
        self.edges = []

        for vec in self.rot, self.rot2, self.rot3:
            self.edges.append(BoundedPlane(vec.pt + self.pos, vec, self.limit, self.limit))
            self.edges.append(BoundedPlane(-vec.pt - self.pos, -vec, self.limit, self.limit))

    def contains(self, pt: Point):
        rad_vec = Vector(pt - self.pos)

        if rad_vec.len() == 0: return True

        rot1_proj = self.rot * rad_vec / rad_vec.len()
        rot2_proj = self.rot2 * rad_vec / rad_vec.len()
        rot3_proj = self.rot3 * rad_vec / rad_vec.len()

        return all(abs(pr) <= 1 for pr in (rot1_proj, rot2_proj, rot3_proj))

    def intersect(self, vec: Vector, vec_begin: Point= Vector.vs.nullpoint):
        # Искать пересечение с лучом, которого еще нет
        pass

    def nearest_point(self, *pts: list[Point]):
        pass

    def rotate(self, c1_angle, c2_angle, c3_angle):
        pass

    def scale(self, scaling_value):
        pass


# =================================================================================================== #
