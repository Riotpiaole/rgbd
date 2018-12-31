import sys
import os
from os import listdir
from os.path import isfile, join
import time

import re 
import random as rnd

import cv2
import numpy as np
from numpy.linalg import inv

from utils import *
from utils_camera import *
from utils_rgbd_images import *



def click_region_call_bk(event,x,y,flags,param):
    '''
    click_region_call_bk
        call back for find the region of interests
    '''
    if event == cv2.EVENT_LBUTTONDOWN:
        print(x,y,param[0][x,y]) 
        if (len(param) != 0 ):
            assert callable(param[0])
        


def convert_depth_2_rgb(img_depth, max_depth=3000):
    img_depth_rgb = img_depth*(255/max_depth) # colors will repeat if the depth can be measured beyond max_depth (default = 10 meters)
    img_depth_rgb = np.uint8(img_depth_rgb)
    img_depth_rgb = cv2.applyColorMap(img_depth_rgb, cv2.COLORMAP_JET)
    return img_depth_rgb

def transform_pointcloud_vectorized(ptcloud, rotMat, transMat, scaleFactor=1000):
    xyz, rgb = np.hsplit(ptcloud, 2)
    # apply rotation
    rotatedXYZ = np.matmul(xyz, rotMat[0:3, 0:3])

    # apply translation
    rotatedXYZ[..., 0] -= (transMat[0] * scaleFactor)
    rotatedXYZ[..., 1] += (transMat[1] * scaleFactor)
    rotatedXYZ[..., 2] += (transMat[2] * scaleFactor)

    transformedPtcloud = np.hstack([rotatedXYZ, rgb])

    return transformedPtcloud



def unproject_pointcloud(ptcloud, camera_params, scaleFactor=1000):
    img_pts = [] # im_x, im_y, im_z, r, g, b

    for p in ptcloud:
        x, y, z, r, g, b = p

        if z > 0: # reverse projection equations
            im_x = ((x/z)*camera_params.fx) + camera_params.cx 
            im_y = ((y/z)*camera_params.fy) + camera_params.cy

            img_pts.append([im_x, im_y, z, r, g, b])

    return np.array(img_pts)

def project_unproject_example():
    strVideoFolder = "./data/"
    strFolderName = "test01"
    strVideoFullPath = os.path.join(strVideoFolder, strFolderName)
    strPoseLocation = os.path.join(strVideoFullPath, "unity3d_poses.txt")

    print("\nPress Q to quit\n")

    ## load manually calibrated poses
    sensor_props = ReadManualCalibPoses(strPoseLocation)
    for s in sensor_props: s.load_image_files(strVideoFullPath)

    cam_params = GetCameraParameters("OrbbecAstra", 0.5)

    ## load images and ensure their are in sorted order
    imgFiles_color = sensor_props[0].imgs_color #listfiles_nohidden(os.path.join(strVideoFullPath, "color"), includeInputPath=True) 
    imgFiles_depth = sensor_props[0].imgs_depth #listfiles_nohidden(os.path.join(strVideoFullPath, "depth"), includeInputPath=True) 
    sort_numerically(imgFiles_color)
    sort_numerically(imgFiles_depth)

    ## load the images for two randomly selected poses
    img1_idx = random.choice(list(range(0, len(imgFiles_color))))
    img1_color = cv2.imread(imgFiles_color[img1_idx])
    gradient = cv2.morphologyEx(img1_color,cv.MORPH_GRADIENT,(3,3))
    
    img1_depth = cv2.imread(imgFiles_depth[img1_idx], -1)

    ptcloud1, nVertices1 = image_fusion(cam_params, img1_depth, img1_color)

    ## transform according to camera pose
    ptcloud1_transformed = transform_pointcloud_vectorized(ptcloud1[:].copy(), sensor_props[0].rotationMatrix(), sensor_props[0].translationMatrix())

    ## undo camera pose transformation
    ptcloud1_undo = transform_pointcloud_vectorized(ptcloud1[:].copy(), np.linalg.inv(sensor_props[0].rotationMatrix()), sensor_props[0].translationMatrix()*-1)

    ## reproject onto the image plane
    img_pts = unproject_pointcloud(ptcloud1_undo, cam_params)
    img1_reproj = np.zeros(img1_color.shape, img1_color.dtype)

    ## render the reporjected points onto an actual image
    img_h, img_w, img_chans = img1_color.shape
    for ip in img_pts:
        x, y, z, r, g, b = ip
        in_xrange = (x > 0) and (x < img_w)
        in_yrange = (y > 0) and (y < img_h)
        if in_xrange and in_yrange:  img1_reproj[int(y), int(x)] = np.array([b, g, r]) ## put it into opencv bgr ordering

    cv2.imshow("image 1", img1_color)
    cv2.imshow("image 1 - reproj", img1_reproj)
    cv2.waitKey(0)

def project_cameraA_onto_cameraB_example():
    strVideoFolder = "./data/"
    strFolderName = "test01"
    strVideoFullPath = os.path.join(strVideoFolder, strFolderName)
    strPoseLocation = os.path.join(strVideoFullPath, "unity3d_poses.txt")

    print("\nPress Q to quit\n")
    
    ## load manually calibrated poses and the image filepaths
    sensor_props = ReadManualCalibPoses(strPoseLocation)
    for s in sensor_props: s.load_image_files(strVideoFullPath)

    cam_params = GetCameraParameters("OrbbecAstra", 0.5)

    ## load the images for two randomly selected poses
    cam_pose1 = sensor_props[0]
    cam_pose2 = sensor_props[1]
    cam_pose3 = sensor_props[2]

    print(cam_pose1,cam_pose2 ,cam_pose3)
    # img_idx = random.choice(list(range(0, len(cam_pose1.imgs_color))))
    img_idx = 0

    ## load the pose and the images for camera 1
    img1_color = cv2.imread(cam_pose1.imgs_color[img_idx])
    img1_depth = cv2.imread(cam_pose1.imgs_depth[img_idx], -1)

# # Filtering the person
#     connectivity = 8
#     test =  img1_depth.copy() 
#     test_cleaned = cv2.morphologyEx(test , cv2.MORPH_OPEN, (3,3))
#     test_smoothed = cv2.bilateralFilter(test_cleaned.astype(np.float32), 19, 75, 75)
    
    
    
#     grad = cv2.morphologyEx(test_smoothed.copy(), cv2.MORPH_GRADIENT, (19,19))
#     test_smoothed[grad > 100] = 0

#     _, bin_mask = cv2.threshold(test_smoothed.astype(np.uint8), 0 , 255 ,cv2.THRESH_BINARY)
#     output = cv2.connectedComponentsWithStats(bin_mask, connectivity)
    
#     num_labels = output[0] # obtain the number of labels 
#     labels = output[1]     # matrix the size and its same shape as input image
#     stats = output[2]      # It has a length equal to the number of labels and a width equal to the number of stats
#     centroids = output[3]  # locations in x and y locations of each centroids


#     print(np.max(labels), np.min(labels), np.median(labels), np.mean(labels))
#     print(stats.shape,num_labels)
#     print(cv2.CC_STAT_AREA)
#     print(stats[1:, cv2.CC_STAT_AREA],1+np.argmax( stats[1:, cv2.CC_STAT_AREA] ))

#     one_piece = np.zeros(labels.shape, dtype=np.uint8)
#     cv2.namedWindow("click")
#     cv2.setMouseCallback("click",click_region_call_bk,[labels])
#     showImage(convert_depth_2_rgb(labels, max_depth=num_labels//2),name="click")
    
#     # selecting the largest region of labels
#     largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
#     one_piece[labels == largest_label] = 255  

#     # generate display images
#     test_rgb = convert_depth_2_rgb(test_smoothed, max_depth=6000)
#     grad_rgb = convert_depth_2_rgb(grad, max_depth=6000)
#     labels_rgb = convert_depth_2_rgb(labels, max_depth=num_labels//2)

#     imgs_to_show = [test_rgb, grad_rgb, labels_rgb, one_piece]
#     imgs_to_show_names = ["depth rgb", "grad rgb", "CCA", "one piece"]

    # showImageSet(imgs_to_show, imgs_to_show_names)

    img1_depth_rgb = convert_depth_2_rgb(img1_depth)
    img1_rgbd = cv2.addWeighted(img1_depth_rgb, 0.25, img1_color, 0.75, 0)

    ## load the pose and the images for camera 2
    img2_color = cv2.imread(cam_pose2.imgs_color[img_idx])
    img2_depth = cv2.imread(cam_pose2.imgs_depth[img_idx], -1)
    img2_depth_rgb = convert_depth_2_rgb(img2_depth)
    img2_rgbd = cv2.addWeighted(img2_depth_rgb, 0.25, img2_color, 0.75, 0)

    ## compute the point clouds using the images of camera 1 and camera 2
    ptcloud1, nVertices1 = image_fusion(cam_params, img1_depth, img1_color)
    ptcloud2, nVertices2 = image_fusion(cam_params, img2_depth, img2_color)
    
    reverse = transform_pointcloud_vectorized( 
                ptcloud2,
                cam_pose1.rotationMatrix(),
                cam_pose1.translationMatrix()*-1)

    reverse = transform_pointcloud_vectorized( 
        reverse,
        inv(cam_pose2.rotationMatrix()),
        cam_pose2.translationMatrix()*-1)
    
    check(reverse)
    check(ptcloud2)
    ## transform ptcloud1 into common world coordinate frame
    ptcloud1_transformed = transform_pointcloud_vectorized(ptcloud1[:].copy(), cam_pose1.rotationMatrix(), cam_pose1.translationMatrix())

    ## transform ptcloud2 into common world coordinate frame
    ptcloud2_transformed = transform_pointcloud_vectorized(ptcloud2[:].copy(), cam_pose2.rotationMatrix(), cam_pose2.translationMatrix())

    ## apply inverse of camera pose 1 to put ptcloud2 in the frame of reference of camera pose 1
    #ptcloud_cam2_on_cam1 = apply_translation( ptcloud2_transformed[:].copy(), cam_pose1.translationMatrix()*-1 )
    #ptcloud_cam2_on_cam1 = apply_rotation( ptcloud_cam2_on_cam1, np.linalg.inv(cam_pose1.rotationMatrix()) )
    ptcloud_cam2_on_cam1 = transform_pointcloud_vectorized(
        ptcloud2_transformed[:].copy(),
        np.linalg.inv(cam_pose1.rotationMatrix()), 
        cam_pose1.translationMatrix()*-1)
    ## reproject onto the image plane so now camera 2 pts will be rendered onto camera 1 image plane
    img_pts = unproject_pointcloud(ptcloud_cam2_on_cam1, cam_params)
    
    img_reproj = np.zeros(img1_color.shape, img1_color.dtype)

    ## render the reporjected points onto an actual image
    img_h, img_w, img_chans = img1_color.shape
    for ip in img_pts:
        x, y, z, r, g, b = ip
        in_xrange = (x > 0) and (x < img_w)
        in_yrange = (y > 0) and (y < img_h)
        if in_xrange and in_yrange:  img_reproj[int(y), int(x)] = np.array([b, g, r]) ## put it into opencv bgr ordering
    
    cv2.imshow("Sensor: " + cam_pose1.name + ", frame " + str(img_idx), img1_rgbd)
    cv2.imshow("Sensor: " + cam_pose2.name + ", frame " + str(img_idx), img2_rgbd)
    cv2.imshow("reproj "  + cam_pose2.name +  " onto " + cam_pose1.name, img_reproj)
    cv2.waitKey(0)


def showImage(img,name="Output_Image"):
    cv2.imshow(name,img)
    key = cv2.waitKey( 0 )& 0xFF 
    if key == ord( 'q' ):
        cv2.destroyAllWindows() 

def showImageSet(imgs,names):
    for img , n in zip(imgs,names):
        cv2.imshow(n , img)

    key = cv2.waitKey( 0 )& 0xFF 
    if key == ord( 'q' ):
        cv2.destroyAllWindows() 


if __name__ == '__main__':
    # project_unproject_example()
    project_cameraA_onto_cameraB_example()
    
    
   