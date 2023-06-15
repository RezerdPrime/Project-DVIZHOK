import configparser
import math
import numpy as np
import sys

# ==================================================================================================================== #
class Point:
    def __init__(self, x, y, z):
        self.c1 = x
        self.c2 = y
        self.c3 = z

    def __add__(self, pt):
        return Point(self.c1 + pt.c1, self.c2 + pt.c2, self.c3 + pt.c3)

    def __mul__(self, arg: [int, float]):
        return Point(self.c1 * arg, self.c2 * arg, self.c3 * arg)

    def __sub__(self, pt):
        return self.__add__(-1 * pt)

    def __rmul__(self, pt):
        return self.__mul__(pt)

    def __truediv__(self, arg: [int, float]):
        assert arg != 0
        return self.__mul__(1 / arg)

    def dist(self, pt):
        return ((self.c1 - pt.c1) ** 2 + (self.c2 - pt.c2) ** 2 + (self.c3 - pt.c3) ** 2) ** 0.5


# =================================================================================================== #
class Vector:
    def __init__(self, *args):

        if len(args) == 1:
            assert isinstance(args[0], Point)
            self.pt = args[0]

        elif len(args) == 3:
            assert all(map(isinstance, args, [(int, float)] * 3))
            self.pt = Point(*args)

    def len(self):
        return self.vs.nullpt.dist(self.pt)

    def norm(self):
        if self.len() == 0:
            return self

        return Vector(self.pt / self.len())

    def __add__(self, vec: "Vector"):
        return Vector(self.pt + vec.pt)

    def __sub__(self, vec: "Vector"):
        return Vector(self.pt - vec.pt)

    def __mul__(self, arg):
        if isinstance(arg, Vector):
            return (self.pt.c1 * arg.pt.c1) + (self.pt.c2 * arg.pt.c2) + (self.pt.c3 * arg.pt.c3)

        else:
            return Vector(self.pt * arg)

    def __rmul__(self, arg: [int, float]):
        return Vector(self.pt * arg)

    def __truediv__(self, arg: [int, float]):
        return Vector(self.pt / arg)

    def __xor__(self, vec):

        x1 = self.pt.c1
        y1 = self.pt.c2
        z1 = self.pt.c3
        x2 = vec.pt.c1
        y2 = vec.pt.c2
        z2 = vec.pt.c3

        x = self.vs.basis[0] * (y1 * z2 - y2 * z1)
        y = self.vs.basis[1] * -(x1 * z2 - x2 * z1)
        z = self.vs.basis[2] * (y2 * x1 - y1 * x2)

        return x + y + z

    def rotate(self, x_angle: float = 0, y_angle: float = 0,
               z_angle: float = 0):

        x_angle = math.pi * x_angle / 360
        y_angle = math.pi * y_angle / 360
        z_angle = math.pi * z_angle / 360

        # Поворот вокруг оси Ox
        y_old = self.pt.c2
        z_old = self.pt.c3
        self.pt.c2 = y_old * math.cos(x_angle) \
                               - z_old * math.sin(x_angle)
        self.pt.c3 = y_old * math.sin(x_angle) \
                               + z_old * math.cos(x_angle)

        # Поворот вокруг оси Oy
        x_old = self.pt.c1
        z_old = self.pt.c3
        self.pt.c1 = x_old * math.cos(y_angle) \
                               + z_old * math.sin(y_angle)
        self.pt.c3 = x_old * -math.sin(y_angle) \
                               + z_old * math.cos(y_angle)

        # Поворот вокруг оси Oz
        x_old = self.pt.c1
        y_old = self.pt.c2
        self.pt.c1 = x_old * math.cos(z_angle) \
                               - y_old * math.sin(z_angle)
        self.pt.c2 = x_old * math.sin(z_angle) \
                               + y_old * math.cos(z_angle)


# =================================================================================================== #
class VectorSpace:
    nullpt = Point(0, 0, 0)
    basis = [Vector(1, 0, 0), Vector(0, 1, 0), Vector(0, 0, 1)]

    '''basis1 = Vector(1, 0, 0)
    basis2 = Vector(0, 1, 0)
    basis3 = Vector(0, 0, 1)'''

    def __init__(self, nullpt: Point = nullpt, dir1: Vector = None,
                 dir2: Vector = None, dir3: Vector = None):
        self.nullpt = nullpt
        for i, d in enumerate((dir1, dir2, dir3)):
            if d is not None:
                VectorSpace.basis[i] = d.norm()

Vector.vs = VectorSpace()


# =================================================================================================== #
class Map:
    def __init__(self):
        self.listobj = []

    def append(self, *objs):
        self.listobj.extend(objs)

    def __getitem__(self, item):
        return self.listobj[item]

    def __iter__(self):
        return iter(self.listobj)


# ==================================================================================================================== #
class Ray:
    def __init__(self, pt: Point, direction: Vector):
        self.pt = pt
        self.dir = direction

    def intersect(self, mapping: Map) -> list[float]:
        return [objt.intersect(self) for objt in mapping]


# ==================================================================================================================== #
class Camera:
    config = configparser.ConfigParser()
    config.read("config.cfg")
    hight = int(config['SCREEN']['hight'])
    width = int(config['SCREEN']['width'])
    ratio = width / hight

    def __init__(self, position: Point, look_dir: Vector,
                 fov, draw_dist):

        self.pos = position
        self.look_dir = look_dir.norm()
        self.draw_dist = draw_dist
        self.fov = (fov / 180 * math.pi) / 2
        self.vfov = self.fov / self.ratio

        self.screen = BoundedPlane(
            self.pos + self.look_dir.pt / math.tan(self.fov),
            self.look_dir, math.tan(self.fov), math.tan(self.vfov))

    def send_rays(self) -> list[list[Ray]]:
        rays = []

        for i, s in enumerate(np.linspace(-self.screen.dv, self.screen.dv, self.hight)):
            rays.append([])

            for t in np.linspace(-self.screen.du, self.screen.du, self.width):
                direction = Vector(self.screen.pos) \
                            + self.screen.v * s + self.screen.u * t

                direction = direction - Vector(self.pos)
                direction.pt.c2 /= 15 / 48
                rays[i].append(Ray(self.pos, direction.norm()))

        return rays

    def rotate(self, x_angle=0, y_angle=0, z_angle=0):
        self.look_dir.rotate(x_angle, y_angle, z_angle)
        self.screen.pr.rotate(x_angle, y_angle, z_angle)
        self.screen.pr.pos = self.pos + self.look_dir.pt
        self.screen._update()


# ==================================================================================================================== #
class Object:
    def __init__(self, pos: Point, rotation: Vector):
        self.pos = pos
        self.rot = rotation

    def contains(self, pt: Point, eps=1e-6) -> bool:
        return False

    def intersect(self, ray: Ray) -> float or None:
        return None

    def nearest_point(self, *pts: list[Point]) -> Point:
        pass


# ==================================================================================================================== #
class Parameters:
    def __init__(self, pos: Point, rotation: Vector):
        self.pos = pos
        self.rot = rotation

    def move(self, pt: Point):
        self.pos = self.pos + pt

    def scaling(self, value):
        pass

    def rotate(self, x_angle: float = 0, y_angle: float = 0, z_angle: float = 0):
        self.rot.rotate(x_angle, y_angle, z_angle)


# ==================================================================================================================== #
class ParametersBoundedPlane(Parameters):
    def __init__(self, pos: Point, rotation: Vector, u, v, du, dv):

        super().__init__(pos, rotation)
        self.u = u
        self.v = v
        self.du = du
        self.dv = dv

    def scaling(self, arg: [int, float]):
        self.du = self.du * arg
        self.dv = self.dv * arg

    def rotate(self, x_angle=0, y_angle=0, z_angle=0):
        self.rot.rotate(x_angle, y_angle, z_angle)
        self.u.rotate(x_angle, y_angle, z_angle)
        self.v.rotate(x_angle, y_angle, z_angle)


# ==================================================================================================================== #
class ParametersSphere(Parameters):
    def __init__(self, pos: Point, rotation: Vector, radius):
        super().__init__(pos, rotation)
        self.r = radius

    def scaling(self, arg: [int, float]):
        self.r = self.r * arg


# ==================================================================================================================== #
class ParametersCube(Parameters):
    def __init__(self, pos: Point, limit, rotations: [Vector], edges: '[BoundedPlane]'):
        super().__init__(pos, rotations[0])
        self.rot2, self.rot3 = rotations[1:]
        self.limit = limit
        self.edges = edges

    def move(self, pt: Point):
        self.pos = self.pos + pt

        for edge in self.edges:
            edge.pos = edge.pos + pt

    def scaling(self, arg: [int, float]):
        self.rot = self.rot * arg
        self.rot2 = self.rot2 * arg
        self.rot3 = self.rot3 * arg
        rotations = [self.rot, self.rot2, self.rot3]
        self.limit *= arg

        for i, edge in enumerate(self.edges):
            edge.pr.scaling(arg)
            if i % 2 == 0:
                edge.pr.pos = self.pos + rotations[i // 2].pt
                edge._update()

            else:
                edge.pr.pos = self.pos - rotations[i // 2].pt
                edge._update()

    def rotate(self, x_angle=0, y_angle=0, z_angle=0):
        self.rot.rotate(x_angle, y_angle, z_angle)
        self.rot2.rotate(x_angle, y_angle, z_angle)
        self.rot3.rotate(x_angle, y_angle, z_angle)

        rotations = [self.rot, self.rot2, self.rot3]
        for i, edge in enumerate(self.edges):
            if i % 2 == 0:
                edge.pr.pos = self.pos + rotations[i // 2].pt

            else:
                edge.pr.pos = self.pos - rotations[i // 2].pt

            edge.pr.rotate(x_angle, y_angle, z_angle)


# ==================================================================================================================== #
class Plane(Object):

    def __init__(self, position, rotation):
        super().__init__(position, rotation)
        self.pr = Parameters(self.pos, self.rot)

    def _update(self):
        self.pos = self.pr.pos
        self.rot = self.pr.rot

    def contains(self, pt: Point, eps=1e-6) -> bool:
        self._update()
        return abs(self.rot * Vector(pt - self.pos)) < eps

    def intersect(self, ray: Ray) -> float:
        self._update()
        if self.rot * ray.dir != 0 and \
                not (self.contains(ray.pt)
                     and self.contains( ray.dir.pt)):

            t0 = (self.rot * Vector(self.pos) - self.rot * Vector(ray.pt)) / (self.rot * ray.dir)
            if t0 >= 0: return t0 * ray.dir.len()

        elif self.contains(ray.pt): return 0

    def nearest_point(self, *pts: Point) -> Point:
        self._update()
        r_min = sys.maxsize
        min_pt = Vector.vs.nullpt

        for pt in pts:
            r = abs(self.rot * Vector(pt - self.pos)) / self.rot.len()
            if r == 0: return pt

            if r < r_min:
                r_min = r
                min_pt = pt

        return min_pt


# ==================================================================================================================== #
class BoundedPlane(Plane):

    def __init__(self, pos: Point, rotation: Vector, du, dv):

        super().__init__(pos, rotation)
        self.du = du
        self.dv = dv

        y_dir = Vector.vs.basis[1]
        if self.rot.pt == y_dir.pt or self.rot.pt == -1 * y_dir.pt:
            y_dir = Vector.vs.basis[0]

        self.u = (self.rot ^ y_dir).norm()
        self.v = (self.rot ^ self.u).norm()

        self.pr = ParametersBoundedPlane(self.pos, self.rot, self.u, self.v, self.du, self.dv)

    def _update(self):
        self.pos = self.pr.pos
        self.rot = self.pr.rot
        self.u = self.pr.u
        self.v = self.pr.v
        self.du = self.pr.du
        self.dv = self.pr.dv

    def in_boundaries(self, pt: Point) -> bool:

        self._update()
        corner = self.u * self.du + self.v * self.dv
        delta_x = corner.pt.c1
        delta_y = corner.pt.c2
        delta_z = corner.pt.c3

        return abs(pt.c1 - self.pos.c1) <= abs(delta_x) \
            and abs(pt.c2 - self.pos.c2) <= abs(delta_y) \
            and abs(pt.c3 - self.pos.c3) <= abs(delta_z)

    def contains(self, pt: Point, eps=1e-6) -> bool:
        self._update()
        if self.in_boundaries(pt):
            return abs(self.rot * Vector(pt - self.pos)) < eps

        return False

    def intersect(self, ray: Ray) -> float or None:

        self._update()
        if self.rot * ray.dir != 0:
            if self.contains(ray.pt): return 0

            t0 = (self.rot * Vector(self.pos) -
                  self.rot * Vector(ray.pt)) / (self.rot * ray.dir)

            pti = ray.dir.pt * t0 + ray.pt
            if t0 >= 0 and self.in_boundaries(pti): return pti.dist(ray.pt)

        elif self.rot * Vector(ray.dir.pt + ray.pt - self.pos) == 0:

            r_begin = Vector(ray.pt - self.pos)
            if r_begin.len() == 0: return 0

            begin_pr1 = r_begin * self.u * self.du / r_begin.len()
            begin_pr2 = r_begin * self.v * self.dv / r_begin.len()
            if abs(begin_pr1) <= 1 and abs(begin_pr2) <= 1: return 0

            r_end = r_begin + ray.dir
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

                if ray1.dir.pt.c1 != 0:
                    x0 = ray1.pt.c1
                    y0 = ray1.pt.c2
                    xr = ray2.pt.c1
                    yr = ray2.pt.c2
                    vx = ray1.dir.pt.c1
                    vy = ray1.dir.pt.c2
                    ux = ray2.dir.pt.c1
                    uy = ray2.dir.pt.c2

                    t1 = ((x0 - xr) * vy / vx + yr - y0) / (uy - ux * vy / vx)
                    s1 = (t1 * ux + x0 - xr) / vx
                    return t1, s1

                elif ray1.dir.pt.c2 != 0:
                    x0 = ray1.pt.c1
                    y0 = ray1.pt.c2
                    xr = ray2.pt.c1
                    yr = ray2.pt.c2
                    vx = ray1.dir.pt.c1
                    vy = ray1.dir.pt.c2
                    ux = ray2.dir.pt.c1
                    uy = ray2.dir.pt.c2

                    t1 = ((y0 - yr) * vx / vy + xr - x0) / (ux - uy * vx / vy)
                    s1 = (t0 * uy + y0 - yr) / vy
                    return t1, s1

                elif ray1.dir.pt.c3 != 0:
                    z0 = ray1.pt.c3
                    y0 = ray1.pt.c2
                    zr = ray2.pt.c3
                    yr = ray2.pt.c2
                    vz = ray1.dir.pt.c3
                    vy = ray1.dir.pt.c2
                    uz = ray2.dir.pt.c3
                    uy = ray2.dir.pt.c2

                    t1 = ((z0 - zr) * vy / vz + yr - y0) / (uy - uz * vy / vz)
                    s1 = (t0 * uz + z0 - zr) / vz
                    return t1, s1

            if abs(begin_pr1) > self.du:
                if self.u * ray.dir == 0: return None

                sign = 1 if begin_pr1 > 0 else -1
                t0, s0 = find_point(Ray(sign * self.du * self.u.point + self.pos, self.dv * self.v), ray)

                if s0 >= 0 and abs(t0) <= 1:
                    return s0 * ray.dir.len()

            elif abs(begin_pr2) > self.dv:
                if self.v * ray.dir == 0: return None

                sign = 1 if begin_pr2 > 0 else -1
                t0, s0 = find_point(Ray(sign * self.dv * self.v.point + self.pos, self.du * self.u), ray)

                if s0 >= 0 and abs(t0) <= 1:
                    return s0 * ray.dir.len()
    def nearest_point(self, *pts: Point) -> Point:

        self._update()
        r_min = sys.maxsize
        min_pt = Vector.vs.nullpt
        r = 0

        for pt in pts:
            r_begin = Vector(pt - self.pos)
            if r_begin.len() == 0:
                return pt

            projection1 = r_begin * self.rot / r_begin.len()
            projection2 = r_begin * self.u * self.du / r_begin.len()
            projection3 = r_begin * self.v * self.dv / r_begin.len()
            sign = lambda x: 1 if x > 0 else -1

            if abs(projection2) <= 1 and abs(projection3) <= 1: r = projection1 * self.rot.len()

            elif abs(projection2) > 1 and abs(projection3) > 1:
                proj2 = projection2 - sign(projection2)
                proj3 = projection3 - sign(projection3)
                r = self.rot * -projection1 + self.u * proj2 + self.v * proj3 + Vector(pt)
                r = r.len()

            elif abs(projection2) > 1:
                proj2 = projection2 - sign(projection2)
                r = self.rot * -projection1 + self.u * proj2 + Vector(pt)
                r = r.len()

            elif abs(projection3) > 1:
                proj3 = projection3 - sign(projection3)
                r = self.rot * -projection1 + self.v * proj3 + Vector(pt)
                r = r.len()

            if r < r_min:
                r_min = r
                min_pt = pt

        return min_pt


# ==================================================================================================================== #
class Sphere(Object):
    def __init__(self, pos: Point, rotation: Vector, radius):
        super().__init__(pos, rotation)
        self.pr = ParametersSphere(self.pos, self.rot.norm() * radius, radius)

    def _update(self):
        self.pos = self.pr.pos
        self.rot = self.pr.rot
        self.r = self.pr.r

    def contains(self, pt: Point, eps=1e-6) -> bool:
        self._update()
        return self.pos.dist(pt) - self.r <= eps

    def intersect(self, ray: Ray) -> float or None:

        self._update()
        a = ray.dir * ray.dir
        b = 2 * ray.dir * Vector(ray.pt - self.pos)
        c = Vector(self.pos) * Vector(self.pos) + Vector(ray.pt) * Vector(ray.pt) - 2 * Vector(self.pos) * Vector(ray.pt) - self.r ** 2

        d = b ** 2 - 4 * a * c
        if d > 0:
            t1 = (-b + math.sqrt(d)) / (2 * a)
            t2 = (-b - math.sqrt(d)) / (2 * a)

            if t2 < 0 <= t1 or 0 < t1 <= t2:
                return t1 * ray.dir.len()

            elif t1 < 0 <= t2 or 0 < t2 <= t1:
                return t2 * ray.dir.len()

        elif d == 0:
            t0 = -b / (2 * a)
            if t0 >= 0: return t0 * ray.dir.len()

    def nearest_point(self, *pts: Point) -> Point:
        self._update()
        r_min = sys.maxsize
        min_pt = Vector.vs.nullpt

        for pt in pts:
            r = self.pos.dist(pt)
            if r == 0: return pt

            if r < r_min:
                r_min = r
                min_pt = pt

        return min_pt


# ==================================================================================================================== #
class Cube(Object):

    def __init__(self, pos: Point, rotation: Vector, size: float):
        super().__init__(pos, rotation)
        self.limit = size / 2
        self.rot = rotation.norm() * self.limit

        x_dir = Vector.vs.basis[0]
        if self.rot.pt == x_dir.pt or self.rot.pt == -1 * x_dir.pt:
            x_dir = Vector.vs.basis[1]

        self.rot2 = (x_dir ^ self.rot).norm() * self.limit
        self.rot3 = (self.rot2 ^ self.rot).norm() * self.limit

        self.edges = []
        for v in self.rot, self.rot2, self.rot3:
            self.edges.append(BoundedPlane(v.pt + self.pos, v, du=self.limit, dv=self.limit))
            self.edges.append(BoundedPlane(-1 * v.pt + self.pos, -1 * v, du=self.limit, dv=self.limit))

        self.pr = ParametersCube(self.pos, self.limit,
                                [self.rot, self.rot2, self.rot3],
                                 self.edges)

    def _update(self):
        self.pos = self.pr.pos
        self.rot = self.pr.rot
        self.rot2 = self.pr.rot2
        self.rot3 = self.pr.rot3
        self.limit = self.pr.limit
        self.edges = self.pr.edges

    def contains(self, pt: Point, eps=1e-6) -> bool:
        self._update()
        v_tmp = Vector(pt - self.pos)
        if v_tmp.len() == 0: return True

        rot1_pr = self.rot * v_tmp / v_tmp.len()
        rot2_pr = self.rot2 * v_tmp / v_tmp.len()
        rot3_pr = self.rot3 * v_tmp / v_tmp.len()
        return all(abs(abs(pr) - 1) <= eps for pr in (rot1_pr, rot2_pr, rot3_pr))

    def intersect(self, ray: Ray, eps=1e-6) -> float or None:
        self._update()

        pts = []
        for edge in self.edges:
            r = edge.intersect(ray)
            if r is not None: pts.append(r)

        if len(pts): return min(pts)

    def nearest_point(self, *pts: Point) -> Point:
        r_min = sys.maxsize
        min_pt = Vector.vs.nullpt
        r = 0

        nearest = [edge.nearest_point(*pts) for edge in self.edges]
        print(*nearest)

        for i, near_pt in enumerate(nearest):
            r_begin = Vector(near_pt - self.edges[i].pos)

            if r_begin.len() == 0: return near_pt

            projection1 = r_begin * self.edges[i].rot / r_begin.len()
            projection2 = r_begin * self.edges[i].u * self.edges[i].du / r_begin.len()
            projection3 = r_begin * self.edges[i].v * self.edges[i].dv / r_begin.len()
            sign = lambda x: 1 if x > 0 else -1

            if abs(projection2) <= 1 and abs(projection3) <= 1:
                r = projection1 * self.edges[i].rot.len()

            elif abs(projection2) > 1 and abs(projection3) > 1:
                proj2 = projection2 - sign(projection2)
                proj3 = projection3 - sign(projection3)
                r = self.edges[i].rot * -projection1 + self.edges[i].u * proj2 + self.edges[i].v * proj3 + Vector(near_pt)
                r = r.len()

            elif abs(projection2) > 1:
                proj2 = projection2 - sign(projection2)
                r = self.edges[i].rot * -projection1 + self.edges[i].u * proj2 + Vector(near_pt)
                r = r.len()

            elif abs(projection3) > 1:
                proj3 = projection3 - sign(projection3)
                r = self.edges[i].rot * -projection1 + self.edges[i].v * proj3 + Vector(near_pt)
                r = r.len()

            if r < r_min:
                r_min = r
                min_pt = near_pt

        return min_pt


cnfg = open("config.cfg").readlines()
symbols = cnfg[3].split('"')[1]


# ==================================================================================================================== #
class Canvas:

    def __init__(self, objmap: Map, camera: Camera):
        self.map = objmap
        self.cam = camera

    def update(self):
        rays = self.cam.send_rays()
        dist_matrix = []

        for i in range(self.cam.hight):
            dist_matrix.append([])
            for j in range(self.cam.width):

                distances = rays[i][j].intersect(self.map)
                if all(d is None or d > self.cam.draw_dist for d in distances):
                    dist_matrix[i].append(None)

                else: dist_matrix[i].append(min(filter(lambda x: x is not None, distances)))

        return dist_matrix


# ==================================================================================================================== #
class Console(Canvas):

    def draw(self):
        dist_matrix = self.update()
        output= ''

        for y in range(len(dist_matrix)):

            for x in range(len(dist_matrix[y])):
                if dist_matrix[y][x] is None:
                    output += symbols[0]
                    continue

                gradient = dist_matrix[y][x] / self.cam.draw_dist * (len(symbols) - 1)
                output += symbols[len(symbols) - round(gradient) - 1]

            output += '\n'

        print(output)


# ==================================================================================================================== #
class Events:
    data = {}

    @classmethod
    def add(cls, event: str):
        cls.data[event] = []

    @classmethod
    def handle(cls, event: str, func: type(add)):
        cls.data[event].append(func)

    @classmethod
    def remove(cls, event: str, func: type(add)):
        for i in range(len(cls.data[event])):
            if cls.data[event][i] is func:
                del cls.data[event][i]
                break

    @classmethod
    def __getitem__(cls, item):
        return cls.data[item]

    @classmethod
    def __iter__(cls):
        return iter(cls.data.keys())

    @classmethod
    def trigger(cls, event, *args):
        try:
            for i in range(len(cls.data[event])):
                called = cls.data[event][i](*args)
                if called is not None:
                    return called
        except Exception:
            print("ERROR")
            print(cls.data[event])

# ==================================================================================================================== #
