import os
import numpy as np

from utils import listfiles_nohidden, sort_numerically

import pyquaternion


class RGBDCameraIntrinsics:
    def __init__(self, fx, fy, cx, cy, img_w, img_h, scale=1.0):
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.img_w = img_w
        self.img_h = img_h
        self.scale = scale

    def rescale(self, scale):
        return RGBDCameraIntrinsics(self.fx*scale, self.fy*scale, self.cx*scale, self.cy*scale, self.img_w*scale, self.img_h*scale, scale)

def GetCameraParameters(camera_name, scale):
    if camera_name == "OrbbecAstra": params = RGBDCameraIntrinsics(570.342, 570.342, 320, 240, 640, 480)
    elif camera_name == "OrbbecAstraV2": params = RGBDCameraIntrinsics(581.1102688880432, 576.1734668524375, 305.7483731000336, 244.8215753417885, 640, 480)
    elif camera_name == "OrbbecAstraPro": params = RGBDCameraIntrinsics(553.797, 553.722, 320, 240, 640, 480)
    elif camera_name == "OrbbecPersee": params = RGBDCameraIntrinsics(553.797, 553.722, 320, 240, 640, 480)
    return params.rescale(scale)


class CameraPose:
    ''' Regular poses from RTAB-Map'''
    def __init__(self, pose_vals):
        self.m11 = pose_vals[0]
        self.m12 = pose_vals[1]
        self.m13 = pose_vals[2]
        self.m14 = pose_vals[3]

        self.m21 = pose_vals[4]
        self.m22 = pose_vals[5]
        self.m23 = pose_vals[6]
        self.m24 = pose_vals[7]

        self.m31 = pose_vals[8]
        self.m32 = pose_vals[9]
        self.m33 = pose_vals[10]
        self.m34 = pose_vals[11]

    def rotationMatrix(self):
        row1 = [self.m11, self.m12, self.m13]
        row2 = [self.m21, self.m22, self.m23]
        row3 = [self.m31, self.m32, self.m33]
        return np.array([row1, row2, row3])

    def translationMatrix(self):
        return np.array([self.m14, self.m24, self.m34])

def ReadCameraPoses(str_filename):
    pose_file = open(str_filename, 'r')
    lines = pose_file.readlines()
    lines = [x.strip() for x in lines]
    pose_file.close()

    poses = []
    for l in lines:
        vals = l.split()
        vals = [float(v) for v in vals]
        poses.append(CameraPose(vals))

    return poses


class RGBDSensorProps:
    ''' Custom manually calibrated poses from Unity3D '''
    def __init__(self, cam_name, cam_position, cam_rotation):
        self.name = cam_name
        self.pos = cam_position # vec3 (x, y, z)
        self.rot = cam_rotation # euler angles (x, y, z) ##pyquaternion.Quaternion(cam_rotation) # quaterion (w, x, y, z)

    def load_image_files(self, top_folder):
        img_folder = os.path.join(top_folder, self.name)
        self.imgs_color = listfiles_nohidden(os.path.join(img_folder, "color"), includeInputPath=True)
        self.imgs_depth = listfiles_nohidden(os.path.join(img_folder, "depth"), includeInputPath=True)
        sort_numerically(self.imgs_color)
        sort_numerically(self.imgs_depth)


    def rotationMatrix(self):
        ''' Convert from quaterion to rotation matrix '''
        # return self.rot.rotation_matrix

        ex,ey,ez = self.rot
        tx = np.deg2rad(-ex) ## put everything from Unity3D space to our vispy space
        ty = np.deg2rad(-ey)
        tz = np.deg2rad(-ez)

        Rx = np.array([[1,0,0,0], [0, np.cos(tx), -np.sin(tx), 0], [0, np.sin(tx), np.cos(tx), 0],[0,0,0,1]])
        Ry = np.array([[np.cos(ty), 0, -np.sin(ty), 0], [0, 1, 0, 0], [np.sin(ty), 0, np.cos(ty),0], [0,0,0,1]])
        Rz = np.array([[np.cos(tz), -np.sin(tz), 0, 0], [np.sin(tz), np.cos(tz), 0, 0], [0,0,1,0], [0,0,0,1]])
        return np.dot(Rx, np.dot(Ry, Rz))

    def translationMatrix(self):
        return np.array(self.pos)
    
    def __str__(self):
        rot , trans = self.rotationMatrix() , self.translationMatrix()
        return "rot {}\n\ntrans{}\n".format(rot , trans)

def ReadManualCalibPoses(str_filename):
    pose_file = open(str_filename, 'r')
    lines = pose_file.readlines()
    lines = [x.strip() for x in lines]
    
    pose_file.close()

    poses = []
    for l in lines:
        vals = l.split(",")
        name = vals[0]
        vals = [float(v) for v in vals[1:]]
        poses.append(RGBDSensorProps(name, vals[0:3], vals[3:]))

    return poses
