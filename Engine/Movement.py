import keyboard
import pyautogui as pag
from Engine import *

config = configparser.ConfigParser()
config.read("config.cfg")
speed_cfg = float(config['MANAGING']['speed'])
dd_diff = abs(float(config['MANAGING']['draw_dist']))
sens = float(config['MANAGING']['sensitivity'])

def move_forward(cam: Camera, speed=speed_cfg):
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

def move_backward(cam: Camera, speed=speed_cfg):
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

def move_left(cam: Camera, speed=speed_cfg):
    y = cam.screen.v.pt.c2
    cam.screen.v.pt.c2 = 0
    cam.pos = cam.pos - cam.screen.u.pt * speed
    cam.screen.pos = cam.screen.pos - cam.screen.u.pt * speed
    cam.screen.v.pt.c2 = y
    return cam

def move_right(cam: Camera, speed=speed_cfg):
    return move_left(cam, -speed)

def move_up(cam: Camera, speed=speed_cfg):
    cam.pos = cam.pos + Point(0, speed, 0) #cam.look_dir.pt * speed
    cam.screen.pos = cam.screen.pos + Point(0, speed, 0)
    cam.look_dir = cam.look_dir + Vector(0, speed, 0)
    return cam


def move_down(cam: Camera, speed=speed_cfg):
    cam.pos = cam.pos - Point(0, speed, 0) #cam.look_dir.pt * speed
    cam.screen.pos = cam.screen.pos - Point(0, speed, 0)
    cam.look_dir = cam.look_dir - Vector(0, speed, 0)
    return cam

def draw_dist_up(cam: Camera, val=dd_diff):
    cam.draw_dist += val
    if cam.draw_dist > 250: cam.draw_dist = 250
    return cam

def draw_dist_down(cam: Camera, val=dd_diff):
    cam.draw_dist -= val
    if cam.draw_dist < 0: cam.draw_dist = 0
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
    Events.add('shift + z')
    Events.add('shift + x')
    Events.handle('shift + w', move_up)
    Events.handle('shift + s', move_down)
    Events.handle('shift + z', draw_dist_up)
    Events.handle('shift + x', draw_dist_down)

#class Player(Camera):
def launch(console: Console, camera_type: str = 'spectator', sensitivity=sens, speed=speed_cfg):

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
            console.cam = Events.trigger(action, console.cam, speed)
            console.draw()

        keyboard.add_hotkey('ctrl+q', close_console)
        keyboard.add_hotkey('w', lambda: act('w'))
        keyboard.add_hotkey('s', lambda: act('s'))
        keyboard.add_hotkey('a', lambda: act('a'))
        keyboard.add_hotkey('d', lambda: act('d'))
        keyboard.add_hotkey('shift+w', lambda: act('shift + w'))
        keyboard.add_hotkey('shift+s', lambda: act('shift + s'))
        keyboard.add_hotkey('shift+z', lambda: act('shift + z'))
        keyboard.add_hotkey('shift+x', lambda: act('shift + x'))

        curr_pos = pag.position()
        pag.moveTo(pag.size()[0] // 2, pag.size()[1] // 2)
        #pag.moveTo(pag.size()[0] * 0.1, pag.size()[1] * 0.9)
        pag.click()

        while exit_cond:
            something_happened = False
            new_pos = pag.position()

            if new_pos != curr_pos:
                something_happened = True
                difference = [(new_pos[0] - curr_pos[0]) * sensitivity, (new_pos[1] - curr_pos[1]) * sensitivity / 5]
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

            if new_pos[0] in (0, 1, 1919, 1920) or new_pos[1] in (0, 1, 1079, 1080):
                pag.moveTo(pag.size()[0] // 2, pag.size()[1] // 2)
                curr_pos = pag.position()

    else:
        # Ich weiss nicht, wie man.
        pass
