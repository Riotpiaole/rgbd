import sys
import os
from os import listdir
from os.path import isfile, join
import time 


import config
import random as rnd


import cv2 , sys 
import numpy as np

from numpy.linalg import inv

from utils import *
from utils_camera import *
from utils_rgbd_images import *
from threading import Thread

def compute_img_diff(label, prev_label ,labels, shape=(240 ,320)):
    tmp = np.zeros(shape , dtype=np.uint8)
    tmp[ labels == label ]= 255
    # #====================================
    # # find the standard deviation error==
    # #====================================
    # # doesnt work when similarity is too huge  
    # _, std = cv2.meanStdDev(prev_label)
    # _, std2 = cv2.meanStdDev(tmp)
    # diff = np.abs(std2-std)
    
    # # ====================================
    # # mean_square_error # i doesnt work all of the value is simlar
    # # ====================================
    # err = np.sum((prev_label.astype(np.float64) - labels.astype(np.float64)) **2 )
    # err /=float(prev_label.shape[0] * prev_label.shape[1] )
    
    #=====================================
    # histograms==========================
    #=====================================
    prev_hist, next_hist = cv2.calcHist([prev_label],[0],None , [256],[0,256] ), cv2.calcHist([tmp],[0],None , [256],[0,256] ) 
    d = cv2.compareHist(prev_hist , next_hist , cv2.HISTCMP_CORREL )
    tmp[ labels == 1 ]= 255
    return d
    

def click_region_call_bk(event,x,y,flags,param):
    '''
    click_region_call_bk
        call back for find the region of interests
        param accept a call back function 
    '''
    
    if event == cv2.EVENT_LBUTTONDOWN:
        if (len(param) != 0 and param[1].frame_num == 0 and parma[1].current_mask[param[2]] == None):
            mask_cam1 = np.zeros(param[0].shape , dtype=np.uint8)
            mask_cam1[ param[0] == param[0][y,x]] = 255
            param[1].callback(mask_cam1,param[2])
        

def unproject_pointcloud(ptcloud, camera_params, scaleFactor=1000):
    img_pts = [] # im_x, im_y, im_z, r, g, b

    for p in ptcloud:
        x, y, z, r, g, b = p

        if z > 0: # reverse projection equations
            im_x = ((x/z)*camera_params.fx) + camera_params.cx 
            im_y = ((y/z)*camera_params.fy) + camera_params.cy

            img_pts.append([im_x, im_y, z, r, g, b])

    return np.array(img_pts)


def reproject_ptcloud(index,src,dest):
    img_h , img_w , img_chans = dest.shape
    start = time()
    for ip in src:
        x, y, z, r, g, b = ip
        in_xrange = (x > 0) and (x < img_w)
        in_yrange = (y > 0) and (y < img_h)
        if in_xrange and in_yrange:  dest[int(y), int(x)] = np.array([b, g, r]) ## put it into opencv bgr ordering
    end = time() 
    time_taken =  round (end - start ,2)
    print("successifully extracted image "+str(index)+" time taken :"+str(time_taken) , end="\r")

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

class DataPreprocessor():
    def __init__(self,config, img_shape=(240,320), num_cam=3 ):
        self.frame_num = 0
        self.config = config
        self.data = [ [] , [] , [] ]
        self.current_mask = [ None , None , None ]
        self.num_cam = num_cam
        self.load_data()
    
    # actual filtering
    @timeit(log_info="Obtain all the traning data ")
    def get_backward_frame(self,debug=False,save=False):
        cam_pose1 , cam_pose2  , cam_pose3 = self.sensor_props[0] ,self.sensor_props[1] , self.sensor_props[2]  
        # for each pair of camera images 
        for i , ((image_cam1_color , image_cam1_depth),(image_cam2_color , image_cam2_depth), (image_cam3_color, image_cam3_depth)) in enumerate(zip(self.data[0] , self.data[1] , self.data[2])): 
            start = time()
            ptcloud2 , nVertice2 = image_fusion(self.cam_params,image_cam2_depth,image_cam2_color)
            ptcloud3 , nVertice3 = image_fusion(self.cam_params,image_cam3_depth,image_cam3_color)

            ptcloud2_transformed = transform_pointcloud_vectorized(ptcloud2[:].copy(), cam_pose2.rotationMatrix(), cam_pose2.translationMatrix())
            ptcloud3_transformed = transform_pointcloud_vectorized(ptcloud3[:].copy(), cam_pose3.rotationMatrix(), cam_pose3.translationMatrix())

            # project cam2 and cam3 backward toward cam1
            ptcloud_cam2_on_cam1 = transform_pointcloud_vectorized(
                ptcloud2_transformed[:].copy(), 
                inv(cam_pose1.rotationMatrix()),
                cam_pose1.translationMatrix()*-1)
            
            ptcloud_cam3_on_cam1 = transform_pointcloud_vectorized(
                ptcloud3_transformed[:].copy(), 
                inv(cam_pose1.rotationMatrix()),
                cam_pose1.translationMatrix()*-1)
            
            # reproject into new cam frame
            img_pts2_1 = unproject_pointcloud(ptcloud_cam2_on_cam1,self.cam_params)            
            img_pts3_1 = unproject_pointcloud(ptcloud_cam3_on_cam1,self.cam_params)
            
            img_pts = np.vstack( ( img_pts2_1 , img_pts3_1 ) )
            img_reproj = np.zeros(image_cam1_color.shape,image_cam1_color.dtype)
            
            reproject_ptcloud(i , img_pts , img_reproj)
            
            img_reproj = cv2.morphologyEx( img_reproj , cv2. MORPH_CLOSE,  (2,2))
            img_reproj = cv2.morphologyEx( img_reproj , cv2. MORPH_OPEN,  (2,2))
            img_reproj = cv2.bilateralFilter(img_reproj, 10 , 30, 50)
            
            # ===================================================================
            # Mask remove black color
            # ===================================================================
            img_reproj = cv2.morphologyEx( img_reproj , cv2. MORPH_CLOSE,  (2,2))
            img_reproj = cv2.morphologyEx( img_reproj , cv2. MORPH_OPEN,  (2,2))
            
            mask_path1 = "./mask/{}/cam{}_mask{}.png".format(self.config.strFolderName,0,i)
            mask_path2 = "./mask/{}/cam{}_mask{}.png".format(self.config.strFolderName,1,i)
            mask_path3= "./mask/{}/cam{}_mask{}.png".format(self.config.strFolderName,2,i)
            
            mask1 = cv2.imread(mask_path1,-1)
            mask1 = np.dstack((mask1,mask1,mask1)).astype(np.uint8)

            mask2 = cv2.imread(mask_path2,-1)
            mask2 = np.dstack((mask2,mask2,mask2)).astype(np.uint8)
            
            mask3 = cv2.imread(mask_path3,-1)
            mask3 = np.dstack((mask3,mask3,mask3)).astype(np.uint8)

            ptcloudM2 , nVertice2 = image_fusion(self.cam_params,image_cam2_depth,mask2)
            ptcloudM3 , nVertice3 = image_fusion(self.cam_params,image_cam3_depth,mask3)

            ptcloud2M_transformed = transform_pointcloud_vectorized(ptcloudM2[:].copy(), cam_pose2.rotationMatrix(), cam_pose2.translationMatrix())
            ptcloud3M_transformed = transform_pointcloud_vectorized(ptcloudM3[:].copy(), cam_pose3.rotationMatrix(), cam_pose3.translationMatrix())

            # project cam2 and cam3 backward toward cam1
            ptcloud_cam2M_on_cam1 = transform_pointcloud_vectorized(
                ptcloud2M_transformed[:].copy(), 
                inv(cam_pose1.rotationMatrix()),
                cam_pose1.translationMatrix()*-1)
            
            ptcloud_cam3M_on_cam1 = transform_pointcloud_vectorized(
                ptcloud3M_transformed[:].copy(), 
                inv(cam_pose1.rotationMatrix()),
                cam_pose1.translationMatrix()*-1)
            
            # reproject into new cam frame
            img_pts2M_1 = unproject_pointcloud(ptcloud_cam2M_on_cam1,self.cam_params)            
            img_pts3M_1 = unproject_pointcloud(ptcloud_cam3M_on_cam1,self.cam_params)

            imgM_pts = np.vstack( ( img_pts2M_1 , img_pts3M_1 ) )
            
            mask_reproj = np.zeros(image_cam1_color.shape,image_cam1_color.dtype)
            reproject_ptcloud(i , imgM_pts  , mask_reproj)
            mask_reproj = cv2.morphologyEx( mask_reproj , cv2. MORPH_CLOSE,  (2,2))
            mask_reproj = cv2.morphologyEx( mask_reproj , cv2. MORPH_OPEN,  (2,2))
            mask_reproj = cv2.bilateralFilter(mask_reproj, 10 , 30, 50)
            
            mask_reproj[ mask_reproj == 255] =1 
            mask_reproj = 1 - mask_reproj
            mask_reproj[ mask_reproj == 1 ] = 255
            img_reproj[ mask_reproj == 255 ] = 255

            mask1[ mask1 == 255 ] =1 
            mask1 = 1- mask1
            mask1[ mask1 == 1 ] = 255

            image_cam1_color[ mask1 == 255 ] = 255

            if debug: showImageSet([image_cam1_color,img_reproj , mask_reproj],["front","back" , "mask_reproj"])

            #  folders to be saved on 
            save_path = self.config.strFilterFullPath 
            save_train_folder_path , save_target_folder_path = os.path.join(save_path,"train"), os.path.join(save_path,"target")
            
            # name of the file
            save_target_file_name , save_train_file_name = os.path.join(save_target_folder_path,"label{}.png".format(i)) , os.path.join(save_train_folder_path,"train{}.png".format(i))
            
            if save: 
                cv2.imwrite(save_target_file_name,img_reproj)
                cv2.imwrite(save_train_file_name,image_cam1_color)
            
            end =time()
            time_taken =  round (end - start ,2)
            # print("Saving train and target at frame {} and Taken:{} ms".format(i,time_taken),end="\r")
            

    # helper func for gettin rgb an d
    def get_rgbd(self,cam,frame_num):
        img_depth = cv2.imread(self.sensor_props[cam].imgs_depth[frame_num],-1)
        img_clr = cv2.imread(self.sensor_props[cam].imgs_color[frame_num])
        return  img_clr ,img_depth


    def load_data(self):
        self.sensor_props = ReadManualCalibPoses(self.config.strPoseLocation)
        for s in self.sensor_props: s.load_image_files(self.config.strVideoFullPath)
        
        self.cam_params = GetCameraParameters("OrbbecAstra", 0.5)        
        self.total_frame_num = len(self.sensor_props[0].imgs_color)
        
        # obtain the each camera images
        for cam in range( self.num_cam ):
            path = "./mask/{}/cam{}_mask0.png".format(self.config.strFolderName , cam+1)
            if os.path.isfile(path):self.current_mask[cam] = cv2.imread(path , -1)
            else:self.foreground_extraction(cam,0)            


    @timeit(log_info="Extracted all the images from each frame")    
    def rgbd_filtering(self,debug=False):
        if debug:
            for camera in range(self.num_cam):
                self.images_extraction(camera , debug=debug)
        else:
            print("debug verbose 0 .....")
            threads = []
            for camera in range(self.num_cam):
                process = Thread(target=self.images_extraction,args=[camera])
                process.start()
                threads.append(process)
            for process in threads: process.join() 
    
    #  wrapper function for multi-thread computing the image background
    def images_extraction(self,cam,debug=False):
        start = time()
        for frame_num in range(self.total_frame_num):
            self.foreground_extraction(cam,frame_num,debug=debug,save=False) # remove background
        end = time() 
        time_taken =  round (end - start ,2)
        print("Finished extracting camera {}, Time: {} ms".format(cam, time_taken))
    
    def foreground_extraction(self ,cam, frame_num,debug = False ,save=False): 
        mask_path ="./mask/{}/cam{}_mask0.png".format(self.config.strFolderName , cam + 1)
        img_color , img_depth = self.get_rgbd(cam,frame_num)        
        next_mask, num_mask = self.filter_img_to_labels(cam,img_depth,debug)
        
        if not os.path.isfile(mask_path) and frame_num == 0 : 
            # the first frame was running require interactive filtering
            cv2.namedWindow("labels")
            cv2.setMouseCallback("labels", click_region_call_bk,[next_mask,self,cam]) # obtain the masked image 
            cv2.imshow("labels",convert_depth_2_rgb(next_mask,max_depth=num_mask//2))
            cv2.waitKey(0)
        # start filtering from 1 onward
        prev_mask = self.current_mask[cam].astype(np.uint8)
        
        # compute the result of the function
        comparison = np.array(
            [ compute_img_diff( label , prev_mask ,  next_mask ) 
                    for label in np.unique(next_mask) ])
        result = np.zeros(next_mask.shape)
        result[next_mask == np.argmax(comparison)] = 1  # 255
        
        # for some of the filtering is inverting the expected result 
        non_zero_size = np.count_nonzero(result)
        zero_size = result.shape[0]*result.shape[1]- non_zero_size
        if zero_size < non_zero_size : result = 1 - result
    
        result[result==1] = 255  # creating the mask for given frame

        mask = np.dstack((result,result,result)).astype(np.uint8)
        
        clr = np.bitwise_and(img_color,mask)  
        
        self.data[cam].insert(frame_num , (clr,img_depth))
        if debug: 
            os.system("clear")
            print("frame_num {} comparison result is {} and max label is {}".format(frame_num ,comparison[np.argmax(comparison)] , np.argmax(comparison)),end="\n")
            showImageSet([self.current_mask[cam],prev_mask,clr,convert_depth_2_rgb(img_depth),mask],
                        ["current_mask ","prev_mask ","clr" , "depth","mask"])

        self.current_mask[cam] = result.astype(np.uint8)
        if save: cv2.imwrite("./mask/{}/cam{}_mask{}.png".format(self.config.strFolderName,cam,frame_num),result.astype(np.uint8))

    def filter_img_to_labels(self,cam,img_depth, debug=False):
        # Extracting the important region relative to previous filters
        param = self.config.filter_param[self.config.strFolderName][cam]
        dsize, ksize , connectivity = param['dsize'] , param['ksize'], param['connectivity']
        bilater = param['bilater']
        grad = param['grad']
        m_open , m_close = param['open'] , param['close']
        max_depth = param['maxDep']
        right_xmin = param['rx']
        left_xmin = param['lx']
        lower_y = param['yd']
        if max_depth <= 2000: raise ValueError("invalid max depth size, at least greater 2000") 
        if img_depth.any():img_depth[ img_depth >= max_depth ] = 0 

        
        # bilateralfiltering to increase the thickness of an image
        dp_clean = cv2.morphologyEx( img_depth , cv2. MORPH_CLOSE,   m_close )
        dp_clean = cv2.morphologyEx( dp_clean , cv2.MORPH_OPEN ,  m_open )
        
        dp_smooth = cv2.bilateralFilter(dp_clean.astype(np.float32), dsize , bilater[0], bilater[1])
        # Forground extraction to intensified the features
        dp_grad = cv2.morphologyEx( dp_smooth , cv2.MORPH_GRADIENT , ksize)
        dp_smooth[ dp_grad > grad ] = 0  # remove some scatter points

        _ , bin_mask = cv2.threshold(dp_smooth.astype(np.uint8), 0 ,255 , cv2.THRESH_BINARY)
        
        if right_xmin: bin_mask[:,right_xmin:] = 0
        if left_xmin: bin_mask[:,0:left_xmin] = 0
        if lower_y: 
            bin_mask[lower_y:,0:left_xmin] = 0 
            bin_mask[lower_y:,right_xmin:] = 0 

        # labels the each region of labels
        num_labels , labels , stats , centroids = cv2.connectedComponentsWithStats(bin_mask, connectivity)
        # if debug: 
        #     img_to_show = [ convert_depth_2_rgb(img,max_depth=num_labels//2) for img in 
        #         [dp_smooth.copy() , dp_grad.copy() , labels.copy()] ]
        #     img_to_show_names = [ "smooth" , "grad" , "label"]
        #     showImageSet( img_to_show , img_to_show_names )

        return labels, num_labels

    def callback(self,mask,cam):
        print("Store mask in {} cam {} at frame {}".format(self.config.strVideoFolder,cam+1, self.frame_num) )
        self.current_mask[cam] = mask.astype(np.uint8) # this one 
        path = "./mask/{}/cam{}_mask0.png".format(self.config.strFolderName,cam+1)
        
        # store the images in the mask/SAMPLE_NAME/mask.png
        cv2.imwrite(os.path.join(
            "./mask/{}/cam{}_mask0.png".format(self.config.strFolderName,cam+1))
            ,self.current_mask[cam])
        
        # destroy the windows 
        cv2.destroyAllWindows()
        

    def filter_demo(self,cam):
        self.images_extraction(cam , debug=True)

            
    def demo(self):
        self.load_data()
        self.rgbd_filtering()
        self.get_backward_frame(save=True , debug=False)
        

if __name__ == "__main__":
    demo = DataPreprocessor(config)
    demo.demo()
    
    
    
    