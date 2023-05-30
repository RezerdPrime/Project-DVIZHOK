from Engine import *
from Movement import *

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

'''for i in range(90):
    map1[0].param.rotate(0, i, 0)
    cons.draw()'''


'''cam.rotate(30, 30, 30)
#cam.rotate(30, 30, 0)
cons = Console(map1, cam)
cons.draw()'''

'''for i in range(90):
    cam = Camera(cam.pos, r(cam.look_dir, 0, 90, 0), cam.fov, cam.draw_distance)
    # r(cam.look_dir, 30, 30, 0)
    cons = Console(map1, cam)
    cons.draw()'''

#print(cam.look_dir, r(cam.look_dir, 15, -30, 45))


'''map1[0].param.rotate(0, 0, 45)
cons.draw()'''

'''for i in range(100):
    map1[0].param.rotate(30, 30, 0)
    #map1[0].upd()
    cons.draw()'''

'''for i in range(50):
    map1[0].param.rotate(-10, 0, 0)
    # map1[0].upd()
    cons.draw()'''

'''a0 = Sphere(Point(0, 0, -3), Vector(0, 0, 1), 1)
a1 = Sphere(Point(0.75, 1, -5), Vector(0, 0, 1), 1)
a2 = Sphere(Point(1.5, 2, -7), Vector(0, 0, 1), 1)'''

'''b = Plane(Point(0, 0, -4), Vector(1, 1, 1))
b.param.rotate(0, 90, 0)
b.upd()'''

#map1.append(a0)

#cons = Console(map1, cam)
#cons.draw()

'''def rut(self, c1_angle, c2_angle, c3_angle):
    alf = c1_angle / 180 * math.pi
    bet = c2_angle / 180 * math.pi
    gam = c3_angle / 180 * math.pi

    vec = n.array([[self.pt.c1],
                   [self.pt.c2],
                   [self.pt.c3]])

    M_x = [[1, 0, 0],
           [0, math.cos(alf), -math.sin(alf)],
           [0, math.sin(alf), math.cos(alf)]]

    M_y = [[math.cos(bet), 0, math.sin(bet)],
           [0, 1, 0],
           [-math.sin(bet), 0, math.cos(bet)]]

    M_z = [[math.cos(gam), -math.sin(gam), 0],
           [math.sin(gam), math.cos(gam), 0],
           [0, 0, 1]]

    REZ = n.dot(n.dot(n.dot(M_x, M_y), M_z), vec)

    if REZ[0][0] == 0: REZ[0][0] -= 1e-6
    if REZ[1][0] == 0: REZ[1][0] -= 1e-6
    if REZ[0][0] == 0: REZ[0][0] -= 1e-6

    return Vector(REZ[0][0], REZ[1][0], REZ[2][0])'''



camer = Camera(Point(-2, 0, 0), Vector(1, 0, 0), 90, 10)
map1 = Map()

'''a = Sphere(Point(1, 0, 0), Vector(0, 0, 0), 0.8)
map1.append(a)
'''
b1 = Sphere(Point(1, 0.5, 0), Vector(0, 0, 0), 0.8)
b2 = Sphere(Point(1, 0.5, -2), Vector(0, 0, 0), 0.8)
b3 = Sphere(Point(1, 0.5, -4), Vector(0, 0, 0), 0.8)
b = BoundedPlane(Point(1, -1, 0), Vector(0.75, 1, 0), 1, 1)
map1.append(b, b1, b2, b3)

cons = Console(map1, camer)
cons.draw()


char = '0'
while char != 'kill ':
    char = input() + ' '

    if char[0] == 'w':
        cons.cam = forward(camer)
        print(cons.cam.pos, cons.cam.screen.v, cons.cam.screen.u)
        cons.draw()

    if char[0] == 's':
        cons.cam = backward(camer)
        print(cons.cam.pos, cons.cam.screen.v, cons.cam.screen.u)
        cons.draw()

    if char[0] == 'd':
        cons.cam = right(camer)
        print(cons.cam.pos, cons.cam.screen.v, cons.cam.screen.u)
        cons.draw()

    if char[0] == 'a':
        cons.cam = left(camer)
        print(cons.cam.pos, cons.cam.screen.v, cons.cam.screen.u)
        cons.draw()

'''cons.cam.pos = cons.cam.pos + Point(0,0,-0.5)
cons.draw()'''

# char = '0'
# while char != 'kill ':
#     char = input() + ' '
#     v = cons.cam.look_dir.norm().pt
#
#     if char[0] == 'q':
#         cons.cam = Camera(cons.cam.pos, rut(cons.cam.look_dir, 0, -5 * char.count('q'), 0), cons.cam.fov, cons.cam.draw_distance)
#         cons.draw()
#
#     if char[0] == 'e':
#         cons.cam = Camera(cons.cam.pos, rut(cons.cam.look_dir, 0, 5 * char.count('e'), 0), cons.cam.fov, cons.cam.draw_distance)
#         cons.draw()
#
#     if char[0] == 'w':
#         cons.cam.pos = cons.cam.pos - v * char.count('w')
#         cons.draw()
#
#     if char[0] == 's':
#         cons.cam.pos = cons.cam.pos + v * char.count('s')
#         cons.draw()
