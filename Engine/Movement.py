from Engine import *


def forward(cam: Camera, speed=1):
    if cam.look_dir != Vector(0, 1, 0) \
            and cam.look_dir != Vector(0, -1, 0):
        y = cam.look_dir.pt.c2
        cam.look_dir.pt.c2 = 0
        cam.pos = cam.pos + cam.look_dir.pt * speed
        cam.look_dir.pt.c2 = y
        cam.screen.pos = cam.pos + cam.look_dir.pt
    else:
        cam.pos = cam.pos - cam.screen.u.pt * speed
        cam.screen.pos = cam.pos + cam.look_dir.pt

    return cam


def backward(cam: Camera, speed=1):
    if cam.look_dir != Vector(0, 1, 0) \
            and cam.look_dir != Vector(0, -1, 0):
        y = cam.look_dir.pt.c2
        cam.look_dir.pt.c2 = 0
        cam.pos = cam.pos - cam.look_dir.pt * speed
        cam.look_dir.pt.c2 = y
        cam.screen.pos = cam.pos + cam.look_dir.pt
    else:
        cam.pos = cam.pos + cam.screen.v.pt * speed
        cam.screen.pos = cam.pos + cam.look_dir.pt

    return cam


def left(cam: Camera, speed=1):
    y = cam.screen.v.pt.c1
    cam.screen.v.pt.c1 = 0
    cam.pos = cam.pos + cam.screen.u.pt * speed
    cam.screen.v.pt.c1 = y

    return cam


def right(cam: Camera, speed=1):
    y = cam.screen.v.pt.c1
    cam.screen.v.pt.c1 = 0
    cam.pos = cam.pos - cam.screen.u.pt * speed
    cam.screen.v.pt.c1 = y

    return cam


def toward(cam: Camera, speed=1):
    cam.pos = cam.pos + cam.look_dir.pt * speed
    return cam


def fromward(cam: Camera, speed=1):
    cam.pos = cam.pos - cam.look_dir.point * speed
    return cam


