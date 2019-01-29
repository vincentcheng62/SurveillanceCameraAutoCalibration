import ctypes
import sys
import pyglet
from pyglet.gl import *
from pywavefront import visualization
import pywavefront

meshes = pywavefront.Wavefront('agv.obj')
window = pyglet.window.Window()
lightfv = ctypes.c_float * 4


@window.event
def on_resize(width, height):
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(60., float(width)/height, 1., 100.)
    glMatrixMode(GL_MODELVIEW)
    return True


@window.event
def on_draw():
    window.clear()
    glLoadIdentity()

    glLightfv(GL_LIGHT0, GL_POSITION, lightfv(-1.0, 1.0, 1.0, 0.0))
    glEnable(GL_LIGHT0)

    glTranslated(0.0, 0.0, -4.0)
    glRotatef(0, 0.0, 0.0, 0.0)

    glEnable(GL_LIGHTING)
    visualization.draw(meshes)

pyglet.app.run()
