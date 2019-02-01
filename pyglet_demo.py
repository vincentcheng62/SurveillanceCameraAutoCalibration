import ctypes
import sys
import pyglet
from pyglet.gl import *
from pywavefront import visualization
import pywavefront
import cv2
import numpy
import time, datetime

start = time.time()
#meshes = pywavefront.Wavefront('agv.obj')
meshes = pywavefront.Wavefront('Tile_+531_+423_export2_crop.obj')
#meshes = pywavefront.Wavefront('testmodel.obj')
#meshes = pywavefront.Wavefront('earth.obj')
#meshes.parse()
print("pywavefront.Wavefront took {} seconds.".format(time.time() - start))

#for name, material in meshes.materials.items():
    #print(material.vertices, "\n")

window = pyglet.window.Window(1024, 720, caption='Demo', resizable=True)
lightfv = ctypes.c_float * 4


@window.event
def on_resize(width, height):
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    # fovy, aspect ratio, zNear, zFar
    gluPerspective(60., float(width)/height, 1, 200)
    glEnable(GL_DEPTH_TEST)
    glMatrixMode(GL_MODELVIEW)
    return True


@window.event
def on_draw():
    window.clear()
    glLoadIdentity()

#GL_POSITION
#params contains four integer or floating-point values that specify the position of the light
# in homogeneous object coordinates. Both integer and floating-point values are mapped directly.
#  Neither integer nor floating-point values are clamped.

#The position is transformed by the modelview matrix when glLight is called (just as if it were a point),
#  and it is stored in eye coordinates. If the w component of the position is 0, the light is treated as a
#  directional source. Diffuse and specular lighting calculations take the light's direction, but not its
#  actual position, into account, and attenuation is disabled. Otherwise, diffuse and specular lighting
#  calculations are based on the actual location of the light in eye coordinates, and attenuation is enabled
# . The initial position is (0, 0, 1, 0); thus, the initial light source is directional, parallel to,
#  and in the direction of the - z axis.

    #glLightfv(GL_LIGHT0, GL_POSITION, lightfv(0, 0, 1, 0.0))
    #glLightfv(GL_LIGHT0, GL_AMBIENT, lightfv(0.5, 0.5, 0.5, 1))
    # glLightfv(GL_LIGHT0, GL_POSITION, lightfv(-40.0, 200.0, 100.0, 0.0))
    # glLightfv(GL_LIGHT0, GL_AMBIENT, lightfv(0.2, 0.2, 0.2, 1.0))
    # glLightfv(GL_LIGHT0, GL_DIFFUSE, lightfv(0.5, 0.5, 0.5, 1.0))
    # glEnable(GL_LIGHT0)
    # glEnable(GL_LIGHTING)
    # glEnable(GL_COLOR_MATERIAL)
    # glEnable(GL_DEPTH_TEST)
    # glShadeModel(GL_SMOOTH)
    # glMatrixMode(GL_MODELVIEW)

    glRotatef(-42.07855, 1, 0, 0)
    glRotatef(-34.662579, 0, 1, 0)
    glRotatef(-55.69070, 0, 0, 1)
    #glTranslated(-78.4485, -9.73987, -100)
    glTranslated(-137.13, 28.8626, -18.6586)
    #glTranslated(0, 0, -10)
    #glTranslated(0, 0, -4)


    visualization.draw(meshes)
    #pyglet.image.get_buffer_manager().get_color_buffer().save('screenshot.png')
    #kitten = pyglet.image.get_buffer_manager().get_color_buffer().get_image_data()
    #data = kitten.get_data('RGB', kitten.width * 3)
    

pyglet.app.run()
