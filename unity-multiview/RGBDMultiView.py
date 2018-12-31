
import sys
import os
from os import listdir
from os.path import isfile, join
from stl import mesh

import re

import cv2
import numpy as np

from vispy import gloo
from vispy import app
from vispy.util.transforms import perspective, translate, rotate


from vispy.visuals import CubeVisual, transforms
from vispy.util.quaternion import Quaternion

from open3d import *

from vispy_shaders import *

from utils import *
from utils_camera import *
from utils_rgbd_images import *
from open3d import Vector3dVector

def _arcball(x, y, w, h):
    """
        Convert x,y coordinates to w,x,y,z Quaternion parameters

        Adapted from:

        linalg library

        Copyright (c) 2010-2015, Renaud Blanch <rndblnch at gmail dot com>
        Licence at your convenience:
        GPLv3 or higher <http://www.gnu.org/licenses/gpl.html>
        BSD new <http://opensource.org/licenses/BSD-3-Clause>
    """
    r = (w + h) / 2.
    x, y = -(2. * x - w) / r, -(2. * y - h) / r
    h = np.sqrt(x*x + y*y)
    return (0., x/h, y/h, 0.) if h > 1. else (0., x, y, np.sqrt(1. - h*h))

def rotationMatrix(theta):
    """ Return the general rotation matrix (4x4) when given the three euler angles """
    tx,ty,tz = theta
    Rx = np.array([[1,0,0,0], [0, np.cos(tx), -np.sin(tx), 0], [0, np.sin(tx), np.cos(tx), 0],[0,0,0,1]])
    Ry = np.array([[np.cos(ty), 0, -np.sin(ty), 0], [0, 1, 0, 0], [np.sin(ty), 0, np.cos(ty),0], [0,0,0,1]])
    Rz = np.array([[np.cos(tz), -np.sin(tz), 0, 0], [np.sin(tz), np.cos(tz), 0, 0], [0,0,1,0], [0,0,0,1]])
    return np.dot(Rx, np.dot(Ry, Rz))

def quaternion_from_matrix(matrix, isprecise=False):
    M = np.array(matrix, dtype=np.float64, copy=False)[:4, :4]
    if isprecise:
        q = np.empty((4, ))
        t = np.trace(M)
        if t > M[3, 3]:
            q[0] = t
            q[3] = M[1, 0] - M[0, 1]
            q[2] = M[0, 2] - M[2, 0]
            q[1] = M[2, 1] - M[1, 2]
        else:
            i, j, k = 0, 1, 2
            if M[1, 1] > M[0, 0]:
                i, j, k = 1, 2, 0
            if M[2, 2] > M[i, i]:
                i, j, k = 2, 0, 1
            t = M[i, i] - (M[j, j] + M[k, k]) + M[3, 3]
            q[i] = t
            q[j] = M[i, j] + M[j, i]
            q[k] = M[k, i] + M[i, k]
            q[3] = M[k, j] - M[j, k]
            q = q[[3, 0, 1, 2]]
        q *= 0.5 / math.sqrt(t * M[3, 3])
    else:
        m00 = M[0, 0]
        m01 = M[0, 1]
        m02 = M[0, 2]
        m10 = M[1, 0]
        m11 = M[1, 1]
        m12 = M[1, 2]
        m20 = M[2, 0]
        m21 = M[2, 1]
        m22 = M[2, 2]
        # symmetric matrix K
        K = np.array([[m00-m11-m22, 0.0,         0.0,         0.0],
                         [m01+m10,     m11-m00-m22, 0.0,         0.0],
                         [m02+m20,     m12+m21,     m22-m00-m11, 0.0],
                         [m21-m12,     m02-m20,     m10-m01,     m00+m11+m22]])
        K /= 3.0
        # quaternion is eigenvector of K that corresponds to largest eigenvalue
        w, V = np.linalg.eigh(K)
        q = V[[3, 0, 1, 2], np.argmax(w)]
    if q[0] < 0.0:
        np.negative(q, q)
    return q




def transform_pointcloud_vectorized(ptcloud, rotMat, transMat, scaleFactor=1000):
    xyz, rgb = np.hsplit(ptcloud, 2)

    # rotatedXYZ = np.matmul(xyz, rotMat)
    rotatedXYZ = np.matmul(xyz, rotMat[0:3, 0:3])
    rotatedXYZ[..., 0] -= (transMat[0] * scaleFactor)
    rotatedXYZ[..., 1] += (transMat[1] * scaleFactor)
    rotatedXYZ[..., 2] += (transMat[2] * scaleFactor)

    transformedPtcloud = np.hstack([rotatedXYZ, rgb])

    return transformedPtcloud


class RGBDMultiView(app.Canvas):
    """ """

    def __init__(self, sensor_type="OrbbecAstra", sensor_scale=1.0):
        app.Canvas.__init__(self, keys='interactive', size=(1280, 1024))
        self.title = 'RGBDMultiView'
        self.quaternion = Quaternion() # for mouse movement

        self.cam_params = GetCameraParameters(sensor_type, sensor_scale)
  
        self.fov = 30
        ps = self.pixel_scale
        self.playbackSpeed = 1

        # view and model control variables
        self.tshift = 0.05 # the amount to translate by
        self.rshift = 0.05 # amount to increase rotation by
        self.scaleFactor = 1000 # the amount to scale the input model by
        self.default_camera_view()

        self.view = np.dot(self.rotMat, self.transMat)
        self.model = np.eye(4, dtype=np.float32)
        self.projection = np.eye(4, dtype=np.float32)

        self.GL_progs = {} # name : data for a single frame

        self.apply_zoom()

        gloo.set_state('translucent', clear_color='gray')

        self.timer = app.Timer('auto', connect=self.on_timer, start=False)


    def on_timer(self, event):
        if self.nframe_curr < self.nframe_total-self.playbackSpeed: # if we still got images, keep loading them
            self.nframe_curr += self.playbackSpeed
            self.set_frame(self.nframe_curr)
        else:
            # no more images means we're at the end so we should stop the timer allowing the video to
            # be reverse or restarted by through the spacebar
            if self.timer.running: self.timer.stop()

        self.update()


    def on_mouse_move(self, event):
        if event.button == 1 and event.last_event is not None:
            x0, y0 = event.last_event.pos
            x1, y1 = event.pos
            w, h = self.size
            self.quaternion = (self.quaternion *
                               Quaternion(*_arcball(x0, y0, w, h)) *
                               Quaternion(*_arcball(x1, y1, w, h)))

            self.rotMat = self.quaternion.get_matrix()
            self.view = np.dot(self.rotMat, self.transMat)
            for k, prog in self.GL_progs.items():
                prog['u_view'] = self.view
                prog['u_model'] = self.model
            self.update()

    def default_camera_view(self):
        """ Set the view matrics to their default values """
        self.offX = 0.0 # actual amounts translated thus far
        self.offY = -0.6
        self.zoomAmount = 2.5
        self.centerCoord = (0, 0, 0) # center of the current point cloud frame
        self.rotX = 9.3
        self.rotY = 6.3
        self.rotZ = 0.0
        self.transMat = translate((self.offX, self.offY, -self.zoomAmount))
        self.rotMat = rotationMatrix((self.rotX, self.rotY, self.rotZ))

        q = quaternion_from_matrix(self.rotMat) ## just for mouse arc ball
        self.quaternion.w = q[0]
        self.quaternion.x = q[1]
        self.quaternion.y = q[2]
        self.quaternion.z = q[3]

    def print_camera_view(self):
        print(self.offX, self.offY, self.zoomAmount)
        print(self.rotX, self.rotY, self.rotZ)
        print()


    def on_key_press(self, event):
        ### pause and play
        if event.text == ' ':
            if self.timer.running: self.timer.stop()
            else: self.timer.start()

        ### manually advance through the frames
        elif event.text == ',' and self.nframe_curr >= self.playbackSpeed:
            self.nframe_curr -= self.playbackSpeed
            self.set_frame(self.nframe_curr)
            self.update()
        elif event.text == '.':
            if self.nframe_curr < self.nframe_total-self.playbackSpeed:
                self.nframe_curr += self.playbackSpeed
            elif self.nframe_curr < self.nframe_total-1:
                self.nframe_curr += 1

            self.set_frame(self.nframe_curr)
            self.update()

        ### camera manipulation
        if event.text in ['w', 'a', 's', 'd']: # planar translations
            if event.text == 'w': self.offY += self.tshift
            elif event.text == 'a': self.offX -= self.tshift
            elif event.text == 's': self.offY -= self.tshift
            elif event.text == 'd': self.offX += self.tshift
            self.transMat = translate((self.offX, self.offY, -self.zoomAmount))

        if event.text in ['q', 'e', 'z', 'c', 'r', 'v']: # rotation about the axis
            if event.text == 'q': self.rotX += self.rshift
            elif event.text == 'e': self.rotX -= self.rshift
            elif event.text == 'z': self.rotY += self.rshift
            elif event.text == 'c': self.rotY -= self.rshift
            elif event.text == 'r': self.rotZ += self.rshift
            elif event.text == 'v': self.rotZ -= self.rshift
            self.rotMat = rotationMatrix((self.rotX, self.rotY, self.rotZ))
            
        ### reset camera to original default orientation
        if event.text == 'x': 
            self.print_camera_view()
            self.default_camera_view()

        if event.text == 'p':
            strTopFolder = "F:/data/deep-learning/additive_depth/98_vr_calib/"
            strFolderName = "test01_img_seq"
            strMultiviewFolder = os.path.join(strTopFolder, strFolderName)
            strPosesFile = os.path.join(strMultiviewFolder, "poses.txt")
            self.sensor_props = ReadManualCalibPoses(strPosesFile)
            for s in sensor_props: s.load_image_files(strMultiviewFolder)

        self.view = np.dot(self.rotMat, self.transMat)
        for k, prog in self.GL_progs.items():
            prog['u_view'] = self.view
            prog['u_model'] = self.model

        self.update()

    def on_resize(self, event):
        self.apply_zoom()

    def on_mouse_wheel(self, event):
        self.zoomAmount -= event.delta[1]/2.0
        self.transMat = translate((self.offX, self.offY, -self.zoomAmount))
        self.view = np.dot(self.rotMat, self.transMat)
        for k, prog in self.GL_progs.items(): prog['u_view'] = self.view
        self.update()

    def on_draw(self, event):
        gloo.clear()
        for k, prog in self.GL_progs.items(): prog.draw('points')

    def apply_zoom(self):
        gloo.set_viewport(0, 0, self.physical_size[0], self.physical_size[1])
        self.projection = perspective(self.fov, self.size[0] / float(self.size[1]), 0.01, 10000.0)
        for k, prog in self.GL_progs.items(): prog['u_projection'] = self.projection


    def load_frame_data(self, frame_n):
        '''
        load_frame_data

        Arguments:
            frame_n: int obtain the frame number for the stream
        
        Return:
            frame_data: array contain three camera and contain [numVertice , xyz , rgb]
        '''
        frame_data = {}
        for s in self.sensor_props:
            depthImg = cv2.imread(s.imgs_depth[frame_n], -1) ## load frames and compute point cloud
            clrImg = cv2.imread(s.imgs_color[frame_n], -1)
            ptcloud, nVertices = image_fusion(self.cam_params, depthImg, clrImg)

            transformedPtcloud = transform_pointcloud_vectorized(ptcloud, s.rotationMatrix(), s.translationMatrix())
            xyz, rgb = np.hsplit(transformedPtcloud, 2)

            ## TODO: extra post-processing to convert points close to the camera to a specific color

            frame_data[s.name] = [nVertices, xyz, rgb]

        return frame_data
    
    def load_frame_to_stl(self, frame_n):
        '''
        load_frame_data

        Arguments:
            frame_n: int obtain the frame number for the stream
        
        Return:
            frame_data: array contain three camera and contain [numVertice , xyz , rgb]
        '''
        data = np.zeros(6,dtype=mesh.Mesh.dtype)
        
        frame_data = {}
        
        xyzpcd , rgbpcd = [] , [] 
    
        for  i , s in enumerate(self.sensor_props):
            depthImg = cv2.imread(s.imgs_depth[frame_n], -1) ## load frames and compute point cloud
            clrImg = cv2.imread(s.imgs_color[frame_n], -1)
            ptcloud, nVertices = image_fusion(self.cam_params, depthImg, clrImg)
            
            transformedPtcloud = transform_pointcloud_vectorized(ptcloud, s.rotationMatrix(), s.translationMatrix())
            xyz, rgb = np.hsplit(transformedPtcloud, 2)
            
            
            xyzpcd.insert(i, xyz)
            rgbpcd.insert(i, rgb )
            ## TODO: extra post-processing to convert points close to the camera to a specific color
            
            frame_data[s.name] = [nVertices, xyz, rgb]
        
        rgbpcd = np.vstack((rgbpcd[0],rgbpcd[1],rgbpcd[2]))
        xyzpcd = np.vstack((xyzpcd[0],xyzpcd[1],xyzpcd[2]))
        
        print(rgbpcd.shape,xyzpcd.shape)

        return xyzpcd , rgbpcd , frame_data

    def set_frame(self, frame_number):
        self.nframe_curr = frame_number
        frame_data = self.load_frame_data(self.nframe_curr)

        for k, prog in self.GL_progs.items():
            nVertices, xyz, rgb = frame_data[k]

            ## load and bind the current frame data to the GL program
            data = np.zeros(nVertices, [('a_position', np.float32, 3),
                            ('a_bg_color', np.float32, 4),
                            ('a_fg_color', np.float32, 4),
                            ('a_size', np.float32, 1)])
            data['a_position'] = np.array(xyz/self.scaleFactor)
            data['a_bg_color'] = np.c_[rgb/255, np.ones(nVertices)] # make sure rgb is between [0,1] and gotta append an extra one to each row for the alpha value 
            data['a_fg_color'] = 0, 0, 0, 1
            data['a_size'] = 4

            prog['u_model'] = self.model
            prog.bind(gloo.VertexBuffer(data))

        self.update()

    def set_multiview_source(self, sensor_props):
        self.nframe_total = sys.maxsize
        self.sensor_props = sensor_props

        for s in self.sensor_props:
            gl_prog = gloo.Program(VERTEX_SHADER, FRAGMENT_SHADER)
            gl_prog['u_linewidth'] = 0
            gl_prog['u_antialias'] = 0
            gl_prog['u_model'] = self.model
            gl_prog['u_view'] = self.view
            gl_prog['u_size'] = 1
            self.GL_progs[s.name] = gl_prog

            self.nframe_total = min(self.nframe_total, len(s.imgs_color))
            self.nframe_total = min(self.nframe_total, len(s.imgs_depth))
        print(self.nframe_total)
        self.apply_zoom()
        self.set_frame(0)
    

if __name__ == '__main__':
    strTopFolder = "./data/"
    strFolderName = "test01"
    strMultiviewFolder = os.path.join(strTopFolder, strFolderName)
    strPosesFile = os.path.join(strMultiviewFolder, "unity3d_poses.txt")

    ## read the poses file at strTopFolder and use that to create the RGBDSensorProps
    sensor_props = ReadManualCalibPoses(strPosesFile)
    

    # for each cameras
    for s in sensor_props: 
        s.load_image_files(strMultiviewFolder)
    
    viewer = RGBDMultiView(sensor_type="OrbbecAstra", sensor_scale=0.5)
    viewer.set_multiview_source( sensor_props )
    xyz , rgb ,  data = viewer.load_frame_to_stl(0)
    

    viewer.show()
    app.run()

