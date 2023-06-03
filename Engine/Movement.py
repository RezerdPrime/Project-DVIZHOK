import keyboard
import pyautogui as pag
from Engine import *


def move_forward(cam: Camera, speed=1):
    if cam.look_dir != Vector(0, 1, 0) \
            and cam.look_dir != Vector(0, -1, 0):
        y = cam.look_dir.pt.c2
        cam.look_dir.pt.c2 = 0
        cam.pos = cam.pos + cam.look_dir.pt * speed
        cam.look_dir.pt.c2 = y
        cam.screen.pos = cam.pos + cam.look_dir.pt
    else:
        cam.pos = cam.pos - cam.screen.u.pt * speed
        cam.screen.pos = cam.pos + cam.look_dir.p

    return cam

def move_backward(cam: Camera, speed=1):
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

def move_left(cam: Camera, speed=1):
    y = cam.screen.v.pt.c2
    cam.screen.v.pt.c2 = 0
    cam.pos = cam.pos - cam.screen.u.pt * speed
    cam.screen.pos = cam.screen.pos - cam.screen.u.pt * speed
    cam.screen.v.pt.c2 = y
    return cam

def move_right(cam: Camera, speed=1):
    return move_left(cam, -speed)

def move_to_viewpoint(cam: Camera, speed=1):
    cam.pos = cam.pos + cam.look_dir.pt * speed
    return cam


def move_from_viewpoint(cam: Camera, speed=1):
    cam.pos = cam.pos - cam.look_dir.pt * speed
    return cam


Events.add("w")
Events.add("s")
Events.add('a')
Events.add('d')
Events.handle('w', move_forward)
Events.handle('s', move_backward)
Events.handle('a', move_left)
Events.handle('d', move_right)


class Spectator(Camera):
    Events.add('shift + w')
    Events.add('shift + s')
    Events.handle('shift + w', move_to_viewpoint)
    Events.handle('shift + s', move_from_viewpoint)

#class Player(Camera):
def launch(console: Console, camera_type: str = 'spectator', sensitivity=1, move_speed=1):

    #global exit_cond
    exit_cond = True

    assert camera_type in ['spectator', 'player']
    if camera_type == 'spectator':
        tmp = console.cam
        console.cam = Spectator(tmp.pos, tmp.look_dir, tmp.fov * 360 / math.pi, tmp.draw_dist)

        def close_console():
            nonlocal exit_cond
            exit_cond = False
            print("Exit: Successful")

        def act(action: str):
            console.cam = Events.trigger(action, console.cam, move_speed)
            console.draw()

        keyboard.add_hotkey('ctrl+q', close_console)
        keyboard.add_hotkey('w', lambda: act('w'))
        keyboard.add_hotkey('s', lambda: act('s'))
        keyboard.add_hotkey('a', lambda: act('a'))
        keyboard.add_hotkey('d', lambda: act('d'))
        keyboard.add_hotkey('shift+w', lambda: act('shift + w'))
        keyboard.add_hotkey('shift+s', lambda: act('shift + s'))

        curr_pos = pag.position()
        #pag.moveTo(pag.size()[0] // 1, pag.size()[1] // 2)
        pag.moveTo(pag.size()[0] * 0.1, pag.size()[1] * 0.9)
        pag.click()

        while exit_cond:
            something_happened = False
            new_pos = pag.position()

            if new_pos != curr_pos:
                something_happened = True
                difference = [(new_pos[0] - curr_pos[0]) * sensitivity, (new_pos[1] - curr_pos[1]) * sensitivity]
                difference[0] /= (pag.size()[0] // 2)
                difference[1] /= (pag.size()[1] // 2)
                t, s = difference

                console.cam.look_dir = t * console.cam.screen.u + s * console.cam.screen.v + Vector(console.cam.screen.pos) - Vector(console.cam.pos)
                console.cam.look_dir = console.cam.look_dir.norm()

                console.cam.screen = BoundedPlane(console.cam.pos + console.cam.look_dir.pt,
                                                  console.cam.look_dir, console.cam.screen.du, console.cam.screen.dv)
                curr_pos = new_pos
                pag.PAUSE = 0.1

            if something_happened:
                console.draw()

    else:
        # Ich weiss nicht, wie man.
        pass
