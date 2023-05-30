import configparser
import numpy as n
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
        return "({0}, {1}, {2})".format(self.c1, self.c2, self.c3)

    def __add__(self, pt):  # Перегрузка операторовw
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

    def __init__(self, *args):
        if len(args) == 1:
            assert isinstance(args[0], Point)
            self.pt = args[0]  # Point(x, y, z)
        elif len(args) == 3:
            assert all(map(isinstance, args, [(int, float)] * 3))
            self.pt = Point(*args)

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
        '''d1 = self.vs.basis1 * (self.pt.c2 * vec.pt.c3 - self.pt.c3 * vec.pt.c2)
        d2 = self.vs.basis2 * (self.pt.c1 * vec.pt.c3 - self.pt.c3 * vec.pt.c1) * -1
        d3 = self.vs.basis3 * (self.pt.c1 * vec.pt.c2 - self.pt.c2 * vec.pt.c1)'''
        x1 = self.pt.c1
        y1 = self.pt.c2
        z1 = self.pt.c3
        x2 = vec.pt.c1
        y2 = vec.pt.c2
        z2 = vec.pt.c3

        d1 = self.vs.basis1 * (y1 * z2 - y2 * z1)
        d2 = self.vs.basis2 * -(x1 * z2 - x2 * z1)
        d3 = self.vs.basis3 * (y2 * x1 - y1 * x2)

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

#vs = VectorSpace(Point(0, 0, 0), Vector(Point(1, 2, 3)))
#vs = VectorSpace()

Vector.vs = VectorSpace()  # Передача векторного пространства в класс Вектор


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

    def intersect(self, mappy: Map): # Schweintersekt
        return [iobj.intersect(self) for iobj in mappy]


# =================================================================================================== #
class Camera:
    config = configparser.ConfigParser()
    config.read("config.cfg")

    '''self.width = int(config['SCREEN']['width']) #self.hight
    self.hight = int(config['SCREEN']['hight'])

    self.ratio = self.hight / self.width'''

    hight = int(config['SCREEN']['hight']) #int(config['SCREEN']['width'])
    width = int(config['SCREEN']['width']) #int(config['SCREEN']['hight'])
    ratio = width / hight

    def __init__(self, pos: Point, look_dir: Vector, fov, draw_distance):
        self.pos = pos
        self.look_dir = look_dir.norm()
        self.fov = (fov / 180 * math.pi) / 2
        self.draw_distance = draw_distance

        self.Vfov = self.fov / self.ratio

        self.screen = BoundedPlane(self.pos + self.look_dir.pt / math.tan(self.fov),
                                   self.look_dir,
                                   math.tan(self.fov),
                                   math.tan(self.Vfov))


    def sendrays(self) -> list[list[Ray]]:
        rays = []

        for i, s in enumerate(n.linspace(-self.screen.dv,
                                           self.screen.dv,
                                           self.hight)): # width
            rays.append([])

            for t in n.linspace(-self.screen.du,
                                self.screen.du,
                                self.width): # hight

                direction = Vector(self.screen.pos) + self.screen.v * s + self.screen.u * t

                direction = direction - Vector(self.pos)
                #direction.pt.c1 *= 50 / 169 #20 / 64
                #direction.pt.c1 /= 14 / 24
                #direction.pt.c2 /= 14 / 44
                direction.pt.c2 *= 400 / 925
                #direction.pt.c1 *= 3 / 2
                #direction.pt.c1 /= 20 / 64
                #direction.pt.c2 *= 15 / 48
                rays[i].append(Ray(self.pos, direction.norm()))

        #print(*[rj.bpt for ri in rays for rj in ri])
        return rays

    def rotate(self, c1_angle, c2_angle, c3_angle):

        alf = c1_angle / 180 * math.pi
        bet = c2_angle / 180 * math.pi
        gam = c3_angle / 180 * math.pi

        '''vec1 = n.array([[self.screen.rotation.pt.c1],
                        [self.screen.rotation.pt.c2],
                        [self.screen.rotation.pt.c3]])'''

        vec2 = n.array([[self.look_dir.pt.c1],
                        [self.look_dir.pt.c2],
                        [self.look_dir.pt.c3]])

        M_x = [[ 1, 0            ,  0             ],
               [ 0, math.cos(alf), -math.sin(alf) ],
               [ 0, math.sin(alf),  math.cos(alf) ]]

        M_y = [[ math.cos(bet), 0, math.sin(bet) ],
               [ 0            , 1, 0             ],
               [-math.sin(bet), 0, math.cos(bet) ]]

        M_z = [[ math.cos(gam), -math.sin(gam), 0 ],
               [ math.sin(gam),  math.cos(gam), 0 ],
               [ 0            ,  0            , 1 ]]

        #REZ1 = n.dot(n.dot(n.dot(M_x, M_y), M_z), vec1)
        REZ2 = n.dot(n.dot(n.dot(M_x, M_y), M_z), vec2)
        self.screen.param.rotate(c1_angle, c2_angle, c3_angle)
        #self.screen.rotation = Vector(REZ1[0][0], REZ1[1][0], REZ1[2][0])
        self.look_dir = Vector(REZ2[0][0], REZ2[1][0], REZ2[2][0])

# =================================================================================================== #
class Object:

    def __init__(self, pos: Point, rotation: Vector):
        self.pos = pos
        self.rotation = rotation

    def contains(self, pt: Point): # returns bool
        return False

    def intersect(self, ray: Ray): # returns float
        return None

    def nearest_point(self, *pts: list[Point]): # returns point
        pass


# =================================================================================================== #
class Parameters:

    def __init__(self, pos: Point, rotation: Vector):
        self.pos = pos
        self.rotation = rotation

    def move(self, pt):
        self.pos = self.pos + pt

    def rotate(self, c1_angle, c2_angle, c3_angle):

        alf = c1_angle / 180 * math.pi
        bet = c2_angle / 180 * math.pi
        gam = c3_angle / 180 * math.pi

        vec = n.array([[self.rotation.pt.c1],
                       [self.rotation.pt.c2],
                       [self.rotation.pt.c3]])

        M_x = [[ 1, 0            ,  0             ],
               [ 0, math.cos(alf), -math.sin(alf) ],
               [ 0, math.sin(alf),  math.cos(alf) ]]

        M_y = [[ math.cos(bet), 0, math.sin(bet) ],
               [ 0            , 1, 0             ],
               [-math.sin(bet), 0, math.cos(bet) ]]

        M_z = [[ math.cos(gam), -math.sin(gam), 0 ],
               [ math.sin(gam),  math.cos(gam), 0 ],
               [ 0            ,  0            , 1 ]]

        REZ = n.dot(n.dot(n.dot(M_x, M_y), M_z), vec)

        self.rotation = Vector(REZ[0][0], REZ[1][0], REZ[2][0])

        '''if c1_angle == c2_angle == c3_angle == 0: return

        x_angle = c1_angle / 180 * math.pi
        y_angle = c2_angle / 180 * math.pi
        z_angle = c3_angle / 180 * math.pi

        # Поворот вокруг оси Ox
        y_old = self.rotation.pt.c2
        z_old = self.rotation.pt.c3
        self.rotation.pt.c2 = y_old * math.cos(x_angle) \
                                   - z_old * math.sin(x_angle)
        self.rotation.pt.c3 = y_old * math.sin(x_angle) \
                                   + z_old * math.cos(x_angle)

        # Поворот вокруг оси Oy
        x_old = self.rotation.pt.c1
        z_old = self.rotation.pt.c3
        self.rotation.pt.c1 = x_old * math.cos(y_angle) \
                                   + z_old * math.sin(y_angle)
        self.rotation.pt.c2 = x_old * -math.sin(y_angle) \
                                   + z_old * math.cos(y_angle)

        # Поворот вокруг оси Oz
        x_old = self.rotation.pt.c1
        y_old = self.rotation.pt.c2
        self.rotation.pt.c1 = x_old * math.cos(z_angle) \
                                   - y_old * math.sin(z_angle)
        self.rotation.pt.c2 = x_old * math.sin(z_angle) \
                                   + y_old * math.cos(z_angle)'''

    def scale(self, scaling_value):
        pass


# =================================================================================================== #
'''class ParametersPlane(Parameters):
    def __init__(self, pos: Point, rotation: Vector):
        self.pos = pos
        self.rotation = rotation'''


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
    def __init__(self, pos: Point, rotation: Vector, radius):
        self.pos = pos
        self.rotation = rotation
        self.rd = radius

    def scale(self, scaling_value):
        self.rd *= scaling_value


# =================================================================================================== #
class ParametersCube(Parameters):
    def __init__(self, pos: Point, limit, rotations: [Vector], edges: '[BoundedPlane]'):
        self.pos = pos
        self.rotation = rotations[0]
        self.rot2 = rotations[1]
        self.rot3 = rotations[2]
        self.limit = limit
        self.edges = edges

    def move(self, pt):
        self.pos = self.pos + pt

        for edge in self.edges:
            edge.pos = edge.pos + pt

    def scale(self, scaling_value):
        self.rotation = self.rotation * scaling_value
        self.rot2 = self.rot2 * scaling_value
        self.rot3 = self.rot3 * scaling_value
        array_rots = [self.rotation, self.rot2, self.rot3]
        self.limit *= scaling_value

        for i, edge in enumerate(self.edges):
            edge.param.scale(scaling_value)

            if i % 2 == 0: edge.pos = self.pos + array_rots[i // 2].pt
            else: edge.pos = self.pos - array_rots[i // 2].pt

    def rotate(self, x_angle, y_angle, z_angle):
        tmp = Parameters(self.pos, self.rotation)
        tmp.rotate(x_angle, y_angle, z_angle)
        self.rotation = tmp.rotation

        tmp.rotation = self.rot2
        tmp.rotate(x_angle, y_angle, z_angle)
        self.rot2 = tmp.rotation

        tmp.rot = self.rot3
        tmp.rotate(x_angle, y_angle, z_angle)
        self.rot3 = tmp.rotation

        rotations = [self.rotation, self.rot2, self.rot3]
        for i, edge in enumerate(self.edges):
            if i % 2 == 0:
                edge.pos = self.pos + rotations[i // 2].pt
            else:
                edge.pos = self.pos - rotations[i // 2].pt

            edge.param.rotate(x_angle, y_angle, z_angle)

# =================================================================================================== #
class Plane(Object):

    def __init__(self, pos: Point, rotation: Vector):

        self.pos = pos
        self.rotation = rotation
        self.param = Parameters(self.pos, self.rotation)

    def upd(self):
        self.pos = self.param.pos
        self.rotation = self.param.rotation

    def contains(self, pt): # A(x - x0) + B(y - y0) + C(z - z0) = 0
        self.upd()
        return abs(self.rotation * Vector(pt - self.pos)) < 1e-6

    def intersect(self, ray: Ray): # returns float
        self.upd()

        if self.rotation * ray.direct != 0 and not (self.contains(ray.bpt) and self.contains(ray.direct.pt)): # Пересечение в одной точке
            t0 = (self.rotation * Vector(self.pos) - self.rotation * Vector(ray.bpt)) / (self.rotation * ray.direct)

            if 0 >= t0:
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
        self.rotation = rotation
        self.pos = pos

        '''if abs(self.rotation.pt.c1) < abs(self.rotation.pt.c2):
            help_vec = Vector.vs.basis1
        else: help_vec = Vector.vs.basis2'''

        help_vec = Vector.vs.basis2

        if self.rotation.pt == help_vec.pt or \
                self.rotation.pt == -1 * help_vec.pt:
            help_vec = Vector.vs.basis1

        '''if self.rotation.pt == help_vec.pt:
            help_vec = Vector.vs.basis3'''

        self.u = (self.rotation ^ help_vec).norm()
        #self.u = (help_vec ^ self.rotation).norm()
        #self.v = (self.rotation ^ self.u).norm()
        self.v = (self.u ^ self.rotation).norm()

        # self.u = (help_vec ^ self.rotation).norm()
        # self.v = (self.rotation ^ self.u).norm()

        self.param = ParametersBoundedPlane(self.pos, self.rotation, self.u, self.v, self.du, self.dv)

        '''self.pos = self.param.pos
        self.rotation = self.param.rotation
        self.u = self.param.u
        self.v = self.param.v
        self.dv = self.param.dv
        self.du = self.param.du'''

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

        return abs(pt.c1 - self.pos.c1) <= abs(delta_x)  \
            and abs(pt.c2 - self.pos.c2) <= abs(delta_y)  \
             and abs(pt.c3 - self.pos.c3) <= abs(delta_z) #\

    def contains(self, pt): # returns bool
        self.upd()
        if self.in_boundaries(pt):
            return abs(self.rotation * Vector(pt - self.pos)) < 1e-6

        return False

    def intersect(self, ray: Ray): # returns float\
        self.upd()

        #print(self.dv * self.v.pt + self.pos, type(self.dv * self.v.pt + self.pos))

        '''if self.rotation * ray.direct != 0 and \
                not (self.rotation * Vector(ray.bpt - self.pos) == 0
                     and self.rotation * Vector(ray.direct.pt + ray.bpt
                                           - self.pos) == 0):
            if self.contains(ray.bpt):
                return 0'''

        if self.rotation * ray.direct != 0:
            if self.contains(ray.bpt):
                return 0

            t0 = (self.rotation * Vector(self.pos) -
                  self.rotation * Vector(ray.bpt)) / (self.rotation * ray.direct)
            int_pt = ray.direct.pt * t0 + ray.bpt

            if t0 >= 0 and self.in_boundaries(int_pt):
                return int_pt.dist(ray.bpt)

        elif self.rotation * Vector(
                ray.direct.pt + ray.bpt - self.pos) == 0:
            # Проекции вектора из точки центра плоскости
            # к точке начала вектора v на направляющие вектора плоскости
            r_begin = Vector(ray.bpt - self.pos)
            # Если начало вектора совпадает с центром плоскости
            if r_begin.len() == 0:
                return 0

            begin_pr1 = r_begin * self.u * self.du / r_begin.len()
            begin_pr2 = r_begin * self.v * self.dv / r_begin.len()
            if abs(begin_pr1) <= 1 and abs(begin_pr2) <= 1:
                return 0

            # Проекции вектора из точки центра плоскости
            # к точке конца вектора v на направляющие вектора плоскости
            r_end = r_begin + ray.direct
            if r_end.len() == 0:
                if abs(begin_pr1) > 1 or abs(begin_pr2) > 1:
                    if begin_pr1 > 1:
                        begin_pr1 -= 1
                    elif begin_pr1 < -1:
                        begin_pr1 += 1

                    if begin_pr2 > 1:
                        begin_pr2 -= 1
                    elif begin_pr2 < -1:
                        begin_pr2 += 1

                    return begin_pr1 * self.du + begin_pr2 * self.dv

                return 0

            def find_point(ray1: Ray, ray2: Ray):

                if ray1.direct.pt.c1 != 0:
                    x0 = ray1.bpt.c1
                    y0 = ray1.bpt.c2
                    xr = ray2.bpt.c1
                    yr = ray2.bpt.c2
                    vx = ray1.direct.pt.c1
                    vy = ray1.direct.pt.c2
                    ux = ray2.direct.pt.c1
                    uy = ray2.direct.pt.c2

                    '''print(x0, y0, xr, yr,
                    vx,
                    vy,
                    ux,
                    uy)'''

                    if uy == ux * vy / vx:
                        return 0, (x0 - xr) / ux

                    t1 = ((x0 - xr) * vy / vx + yr - y0) \
                         / (uy - ux * vy / vx)
                    s1 = (t1 * ux + x0 - xr) / vx
                    return t1, s1

                elif ray1.direct.pt.c2 != 0:
                    x0 = ray1.bpt.c1
                    y0 = ray1.bpt.c2
                    xr = ray2.bpt.c1
                    yr = ray2.bpt.c2
                    vx = ray1.direct.pt.c1
                    vy = ray1.direct.pt.c2
                    ux = ray2.direct.pt.c1
                    uy = ray2.direct.pt.c2
                    t1 = ((y0 - yr) * vx / vy + xr - x0) \
                         / (ux - uy * vx / vy)
                    s1 = (t0 * uy + y0 - yr) / vy
                    return t1, s1

                elif ray1.direct.pt.c3 != 0:
                    z0 = ray1.bpt.c3
                    y0 = ray1.bpt.c2
                    zr = ray2.bpt.c3
                    yr = ray2.bpt.c2
                    vz = ray1.direct.pt.c3
                    vy = ray1.direct.pt.c2
                    uz = ray2.direct.pt.c3
                    uy = ray2.direct.pt.c2
                    t1 = ((z0 - zr) * vy / vz + yr - y0) / (
                            uy - uz * vy / vz)
                    s1 = (t0 * uz + z0 - zr) / vz
                    return t1, s1

            if abs(begin_pr1) > self.du:
                if self.u * ray.direct == 0:
                    return

                sign = 1 if begin_pr1 > 0 else -1
                t0, s0 = find_point(
                    Ray(sign * self.du * self.u.pt + self.pos,
                        self.dv * self.v), ray)
                if s0 >= 0 and abs(t0) <= 1:
                    return s0 * ray.direct.len()

            elif abs(begin_pr2) > self.dv:
                if self.v * ray.direct == 0:
                    return

                sign = 1 if begin_pr2 > 0 else -1

                t0, s0 = find_point(
                    Ray(sign * self.dv * self.v.pt + self.pos,
                        self.du * self.u), ray)
                if s0 >= 0 and abs(t0) <= 1:
                    return s0 * ray.direct.len()

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

    def __init__(self, pos: Point, rotation: Vector, radius):
        self.rd = radius
        self.pos = pos
        self.rotation = rotation.norm() * self.rd
        self.param = ParametersSphere(self.pos, self.rotation, radius)

    def upd(self):
        self.pos = self.param.pos
        self.rotation = self.param.rotation
        self.rd = self.param.rd

    def contains(self, pt):
        self.upd()
        return self.pos.dist(pt) - self.rd <= 1e-6
        #return pt.c1 ** 2 + pt.c2 ** 2 + pt.c3 ** 2 <= self.rd

    def intersect(self, ray: Ray): # returns float
        self.upd()

        a = ray.direct * ray.direct
        b = 2 * ray.direct * Vector(ray.bpt - self.pos)
        c = Vector(self.pos) * Vector(self.pos) + \
            Vector(ray.bpt) * Vector(ray.bpt) \
            - 2 * Vector(self.pos) * Vector(ray.bpt) - self.rd ** 2

        d = b ** 2 - 4 * a * c
        if d > 0:
            t1 = (-b + math.sqrt(d)) / (2 * a)
            t2 = (-b - math.sqrt(d)) / (2 * a)
            # Смотрим пересечения с поверхностью сферы
            if t2 < 0 <= t1 or 0 < t1 <= t2:
                return t1 * ray.direct.len()
            elif t1 < 0 <= t2 or 0 < t2 <= t1:
                return t2 * ray.direct.len()

        elif d == 0:
            t0 = -b / (2 * a)
            if t0 >= 0:
                return t0 * ray.direct.len()


    def nearest_point(self, *pts: Point):
        dist_min = 2 ** 63 - 1
        pt_min = Vector.vs.nullpoint

        for pt in pts:
            dist = self.pos.dist(pt)

            if dist == 0:
                return pt

            if dist < dist_min:
                dist_min = dist
                pt_min = pt

        return pt_min


# =================================================================================================== #
class Cube(Object):

    def __init__(self, pos: Point, rotation: Vector):
        self.rotation = rotation
        self.pos = pos
        self.limit = self.rotation.len()

        if abs(self.rotation.pt.c1) < abs(self.rotation.pt.c2):
            x_dir = Vector.vs.basis1
        else: x_dir = Vector.vs.basis1

        self.rot2 = (self.rotation ^ x_dir).norm()
        self.rot3 = (self.rot2 ^ x_dir).norm()
        self.edges = []

        for vec in self.rotation, self.rot2, self.rot3:
            self.edges.append(BoundedPlane(vec.pt + self.pos, vec, self.limit, self.limit))
            self.edges.append(BoundedPlane(-vec.pt - self.pos, -vec, self.limit, self.limit))

        self.param = ParametersCube(self.pos, self.limit, [self.rotation, self.rot2, self.rot3], self.edges)

    def upd(self):
        self.pos = self.param.pos
        self.rotation = self.param.rotation
        self.rot2 = self.param.rot2
        self.rot3 = self.param.rot3
        self.limit = self.param.limit
        self.edges = self.param.edges

    def contains(self, pt: Point):
        self.upd()
        rad_vec = Vector(pt - self.pos)

        if rad_vec.len() == 0: return True

        rot1_proj = self.rotation * rad_vec / rad_vec.len()
        rot2_proj = self.rot2 * rad_vec / rad_vec.len()
        rot3_proj = self.rot3 * rad_vec / rad_vec.len()

        return all(abs(abs(pr) - 1) <= 1e-6 for pr in (rot1_proj, rot2_proj, rot3_proj))

    def intersect(self, ray: Ray):
        self.upd()

        int_pts = []
        for edge in self.edges:
            r = edge.intersect(ray)
            if r is not None:
                int_pts.append(r)

        if len(int_pts):
            return min(int_pts)

    def nearest_point(self, *pts: Point):
        self.upd()

        r_min = 2 ** 63 - 1
        min_pt = Vector.vs.nullpoint
        r = 0
        nearest = [edge.nearest_point(*pts) for edge in self.edges]
        print(*nearest)

        for i, near_pt in enumerate(nearest):
            r_begin = Vector(near_pt - self.edges[i].pos)
            # Если начало вектора совпадает с центром плоскости
            if r_begin.len() == 0:
                return near_pt

            projection1 = r_begin * self.edges[i].rotation / r_begin.len()
            projection2 = r_begin * self.edges[i].u * self.edges[i].du \
                          / r_begin.len()
            projection3 = r_begin * self.edges[i].v * self.edges[i].dv \
                          / r_begin.len()

            sign = lambda x: 1 if x > 0 else -1
            if abs(projection2) <= 1 and abs(projection3) <= 1:
                r = projection1 * self.edges[i].rotation.len()

            elif abs(projection2) > 1 and abs(projection3) > 1:
                proj2 = projection2 - sign(projection2)
                proj3 = projection3 - sign(projection3)

                r = self.edges[i].rotation * -projection1 \
                    + self.edges[i].u * proj2 \
                    + self.edges[i].v * proj3 + Vector(near_pt)
                r = r.len()

            elif abs(projection2) > 1:
                proj2 = projection2 - sign(projection2)
                r = self.edges[i].rotation * -projection1 \
                    + self.edges[i].u * proj2 + Vector(near_pt)
                r = r.len()

            elif abs(projection3) > 1:
                proj3 = projection3 - sign(projection3)
                r = self.edges[i].rotation * -projection1 \
                    + self.edges[i].v * proj3 + Vector(near_pt)
                r = r.len()

            if r < r_min:
                r_min = r
                min_pt = near_pt

        return min_pt


# =================================================================================================== #
#symbols = " .:!/r(l1Z4H9W8$@"
symbols = " .-:+=*ZH&#$%@W"
#(' ', '.', ',', '-', '+', '=', '*', '#', '%', '&', '$', '@')
#symbols = (' ', '$', '&', '%', '#', '*', '=', '+', '-', ',', '.', '@')
#symbols = (' ', '.', ',', '-', '+', '=', '*', '#', '%', '&')

class Canvas:

    def __init__(self, map_: Map, cam: Camera):
        self.map = map_
        self.cam = cam

    def update(self):
        rays = self.cam.sendrays()
        dist_matrix = []

        for i in range(self.cam.hight): # width
            dist_matrix.append([])

            for j in range(self.cam.width): # hight
                #print(len(rays[i]))
                distances = rays[i][j].intersect(self.map)

                if all(d is None or d > self.cam.draw_distance
                       for d in distances):
                    dist_matrix[i].append(None)

                else:
                    dist_matrix[i].append(
                        min(filter(lambda x: x is not None, distances)))

        return dist_matrix


# =================================================================================================== #
class Console(Canvas):

    def draw(self):
        dist_matrix = self.update()
        output = ""

        for y in range(len(dist_matrix)):

            for x in range(len(dist_matrix[y])):
                if dist_matrix[y][x] is None:
                    #print(symbols[0], end='')
                    output += symbols[0]
                    continue

                gradient = dist_matrix[y][x] / self.cam.draw_distance * (len(symbols) - 1)
                #print(dist_matrix[y][x], gradient)
                #print(symbols[len(symbols) - round(gradient) - 1], end='')
                output += symbols[len(symbols) - round(gradient) - 1]

            output += "\n"

        print(output)

# =================================================================================================== #
