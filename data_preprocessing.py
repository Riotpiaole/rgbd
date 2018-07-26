import sys
import os
from time import time
from os.path import isfile, join

# import config
from config import config


import cv2
import sys
import numpy as np

from numpy.linalg import inv

# Must have function for image filtering
from utils import (
    transform_pointcloud_vectorized,
    scatter_point_filtering,
    unproject_pointcloud,
    click_region_call_bk,
    convert_depth_2_rgb,
    reproject_ptcloud,
    compute_img_diff,
    black_bg,
    white_bg,
)

# Helper function
from utils import (
    multi_threads_wrapper,
    check_folders,
    showImageSet,
    timeit,
)

from utils_camera import (
    ReadManualCalibPoses,
    GetCameraParameters
)
from utils_rgbd_images import (
    image_fusion
)


config = config("ImgSeq_Po_02_Bag", radius=0)


def two_camera_reprojection(
        iteration,
        cam_params,
        sensor_props,
        rgb1,
        rgb2,
        dep1,
        dep2,
        debug=False,
        radius=2):
    cam_pose1, cam_pose2, cam_pose3 = sensor_props

    ptcloud2, _ = image_fusion(cam_params, dep1, rgb1)
    ptcloud3, _ = image_fusion(cam_params, dep2, rgb2)

    ptcloud2_transformed = transform_pointcloud_vectorized(
        ptcloud2[:].copy(), cam_pose2.rotationMatrix(), cam_pose2.translationMatrix())
    ptcloud3_transformed = transform_pointcloud_vectorized(
        ptcloud3[:].copy(), cam_pose3.rotationMatrix(), cam_pose3.translationMatrix())

    # project cam2 and cam3 backward toward cam1
    ptcloud_cam2_on_cam1 = transform_pointcloud_vectorized(
        ptcloud2_transformed[:].copy(),
        inv(cam_pose1.rotationMatrix()),
        cam_pose1.translationMatrix() * -1)

    ptcloud_cam3_on_cam1 = transform_pointcloud_vectorized(
        ptcloud3_transformed[:].copy(),
        inv(cam_pose1.rotationMatrix()),
        cam_pose1.translationMatrix() * -1)

    # reproject into new cam frame
    img_pts2_1 = unproject_pointcloud(ptcloud_cam2_on_cam1, cam_params)
    img_pts3_1 = unproject_pointcloud(ptcloud_cam3_on_cam1, cam_params)

    img_pts = np.vstack((img_pts2_1, img_pts3_1))
    img_reproj = np.zeros(rgb1.shape, rgb1.dtype)

    debug_diff = reproject_ptcloud(
        iteration, img_pts, img_reproj, radius=radius)

    img_reproj = scatter_point_filtering(img_reproj)
    if debug:
        showImageSet([debug_diff, img_reproj], [
                     "without_inpainted", "with_inpainted"])

    return img_reproj


class DataPreprocessor():
    def __init__(self, config, img_shape=(240, 320), num_cam=3):
        self.frame_num = 0
        self.config = config
        self.data = [[], [], []]
        self.current_mask = [None, None, None]
        self.num_cam = num_cam
        self.load_data()

    # actual filtering
    @timeit(log_info="Obtain all the traning data ")
    def get_backward_frame(self, debug=True, save=False):
        # for each pair of camera images
        for i, ((image_cam1_color, image_cam1_depth), (image_cam2_color, image_cam2_depth),
                (image_cam3_color, image_cam3_depth)) in enumerate(zip(self.data[0], self.data[1], self.data[2])):

            img_reproj = two_camera_reprojection(
                i,
                self.cam_params,
                self.sensor_props,
                image_cam2_color,
                image_cam3_color,
                image_cam2_depth,
                image_cam3_depth,
                debug=False,
                radius=self.config.radius)

            # ===================================================================
            # Mask remove black color
            # ===================================================================
            mask_path1 = "./mask/{}/cam{}_mask{}.png".format(
                self.config.strFolderName, 0, i)
            mask_path2 = "./mask/{}/cam{}_mask{}.png".format(
                self.config.strFolderName, 1, i)
            mask_path3 = "./mask/{}/cam{}_mask{}.png".format(
                self.config.strFolderName, 2, i)

            mask1 = cv2.imread(mask_path1, -1)
            mask1 = np.dstack((mask1, mask1, mask1)).astype(np.uint8)

            mask2 = cv2.imread(mask_path2, -1)
            mask2 = np.dstack((mask2, mask2, mask2)).astype(np.uint8)

            mask3 = cv2.imread(mask_path3, -1)
            mask3 = np.dstack((mask3, mask3, mask3)).astype(np.uint8)

            mask_reproj = two_camera_reprojection(
                i,
                self.cam_params,
                self.sensor_props,
                mask2,
                mask3,
                image_cam2_depth,
                image_cam3_depth,
                debug=debug)

            img_reproj_bk, img_cam1_color_bk = black_bg(
                img_reproj[:].copy(), image_cam1_color[:].copy(),
                mask1, mask_reproj)

            img_reproj_wh, img_cam1_color_wh = white_bg(
                img_reproj[:].copy(), image_cam1_color[:].copy(),
                mask1, mask_reproj)

            if debug:
                showImageSet([img_reproj_bk, img_cam1_color_bk, img_reproj_wh, img_cam1_color_wh],
                             ["front_bk", "back_bk", "front_wh", "back_wh"])

            #  folders to be saved on
            save_path = self.config.strFilterFullPath
            save_path_bk = self.config.strFilterFullPathBlack
            check_folders(save_path_bk)
            if save:
                self.save(save_path, i, img_cam1_color_wh, img_reproj_wh)
                self.save(save_path_bk, i, img_cam1_color_bk, img_reproj_bk)

    def save(self, save_path, index, train, label):
        save_train_folder_path, save_target_folder_path = os.path.join(
            save_path, "train"), os.path.join(save_path, "target")

        check_folders(save_train_folder_path)
        check_folders(save_target_folder_path)

        save_target_file_name = os.path.join(
            save_target_folder_path,
            "label{}.png".format(index))
        save_train_file_name = os.path.join(
            save_train_folder_path,
            "train{}.png".format(index))

        cv2.imwrite(save_target_file_name, label)
        cv2.imwrite(save_train_file_name, train)

    # helper func for gettin rgb an d

    def get_rgbd(self, cam, frame_num):
        img_depth = cv2.imread(
            self.sensor_props[cam].imgs_depth[frame_num], -1)
        img_clr = cv2.imread(self.sensor_props[cam].imgs_color[frame_num])
        return img_clr, img_depth

    def load_data(self):
        self.sensor_props = ReadManualCalibPoses(self.config.strPoseLocation)
        for s in self.sensor_props:
            s.load_image_files(self.config.strVideoFullPath)

        self.cam_params = GetCameraParameters("OrbbecAstra", 0.5)
        self.total_frame_num = len(self.sensor_props[0].imgs_color)

        # Obtain all initial mask
        for cam in range(self.num_cam):
            path = "./mask/{}/cam{}_mask0.png".format(
                self.config.strFolderName, cam)
            if os.path.isfile(path):
                self.current_mask[cam] = cv2.imread(path, -1)
            else:
                self.foreground_extraction(cam, 0)
        self.rgbd_filtering(debug=False)

    @timeit(log_info="Extracted all the masks from each frame")
    def rgbd_filtering(self, debug=False):
        if debug:
            for camera in range(self.num_cam):
                self.images_extraction(camera, debug)
                print("Running camera extraction ", camera)
        else:
            print("debug verbose 0 .....")
            self.threaded_images_extraction(self)

    #  wrapper function for multi-thread computing the image background
    @staticmethod
    @multi_threads_wrapper([0, 1, 2])
    def threaded_images_extraction(arg, *args):
        arg, cam = arg[0], args[0]
        for frame_num in range(arg.total_frame_num):
            arg.foreground_extraction(
                cam, frame_num, save=True)  # remove background

    def images_extraction(self, cam, debug=False):
        start = time()
        for frame_num in range(self.total_frame_num):
            self.foreground_extraction(
                cam, frame_num, debug=debug, save=False)  # remove background
        end = time()
        time_taken = round(end - start, 2)
        print(
            "Finished extracting camera {}, Time: {} ms".format(
                cam, time_taken))

    def foreground_extraction(self, cam, frame_num, debug=False, save=False):
        check_folders("./mask/{}".format(self.config.strFolderName))
        mask_path = "./mask/{}/cam{}_mask{}.png".format(
            self.config.strFolderName, cam, frame_num)
        img_color, img_depth = self.get_rgbd(cam, frame_num)
        next_mask, num_mask = self.connected_comp_labeling(
            cam, img_depth, debug=False)
        result = None

        if not os.path.isfile(mask_path) and frame_num == 0:
            # the first frame was running require interactive filtering
            cv2.namedWindow("select first labels")
            cv2.setMouseCallback(
                "select first labels", click_region_call_bk, [
                    next_mask, self, cam])  # obtain the masked image
            cv2.imshow(
                "select first labels",
                convert_depth_2_rgb(
                    next_mask,
                    max_depth=num_mask // 2))
            k = cv2.waitKey(0) & 0xff
            if k == ord('q'):
                cv2.destroyAllWindows()
            result = self.current_mask[cam]
        elif os.path.isfile(mask_path) and frame_num != 0:
            result = cv2.imread(mask_path, -1)
        else:
            # compute the result of the function
            prev_mask = self.current_mask[cam].astype(np.uint8)
            comparison = np.array(
                [compute_img_diff(label, prev_mask, next_mask)
                 for label in np.unique(next_mask)])
            result = np.zeros(next_mask.shape)
            result[next_mask == np.argmax(comparison)] = 1  # 255

            # for some of the filtering is inverting the expected result
            non_zero_size = np.count_nonzero(result)
            zero_size = result.shape[0] * result.shape[1] - non_zero_size
            if zero_size < non_zero_size:
                result = 1 - result

            result[result == 1] = 255  # creating the mask for given frame
            if save:
                cv2.imwrite(
                    "./mask/{}/cam{}_mask{}.png".format(
                        self.config.strFolderName, cam, frame_num), result.astype(
                        np.uint8))

        mask = np.dstack((result, result, result)).astype(np.uint8)
        clr = np.bitwise_and(img_color, mask)

        self.data[cam].insert(frame_num, (clr, img_depth))

        if debug:
            showImageSet([self.current_mask[cam], clr, convert_depth_2_rgb(
                img_depth), mask], ["current_mask ", "clr", "depth", "mask"])
        self.current_mask[cam] = result.astype(np.uint8)

    def connected_comp_labeling(self, cam, img_depth, debug=True):
        # Extracting the important region relative to previous filters
        param = self.config.filter_param[self.config.strFolderName][cam]
        dsize, ksize, connectivity = param['dsize'], param['ksize'], param['connectivity']
        bilater = param['bilater']
        grad = param['grad']
        m_open, m_close = param['open'], param['close']
        max_depth = param['maxDep']
        right_xmin = param['rx']
        left_xmin = param['lx']
        lower_y = param['yd']
        if max_depth <= 2000:
            raise ValueError("invalid max depth size, at least greater 2000")
        if img_depth.any():
            img_depth[img_depth >= max_depth] = 0
        # # bilateralfiltering to increase the thickness of an image
        dp_smooth = scatter_point_filtering(
            img_depth, m_close, m_open, grad, dsize, bilater, ksize)
        _, bin_mask = cv2.threshold(
            dp_smooth.astype(
                np.uint8), 0, 255, cv2.THRESH_BINARY)

        if right_xmin:
            bin_mask[:, right_xmin:] = 0
        if left_xmin:
            bin_mask[:, 0:left_xmin] = 0
        if lower_y:
            bin_mask[lower_y:, 0:left_xmin] = 0
            bin_mask[lower_y:, right_xmin:] = 0

        # labels the each region of labels
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            bin_mask, connectivity)
        if debug:
            img_to_show = [
                convert_depth_2_rgb(
                    img,
                    max_depth=num_labels //
                    2) for img in [
                    convert_depth_2_rgb(
                        img_depth,
                        max_depth=7000),
                    dp_smooth.copy(),
                    labels.copy()]]
            img_to_show_names = ["smooth", "label"]
            showImageSet(img_to_show, img_to_show_names)

        return labels, num_labels

    def callback(self, mask, cam):
        print("Store mask in {} cam {} at frame {}".format(
            self.config.strVideoFolder, cam, self.frame_num))
        self.current_mask[cam] = mask.astype(np.uint8)  # this one
        path = "./mask/{}/cam{}_mask0.png".format(
            self.config.strFolderName, cam)
        # store the images in the mask/SAMPLE_NAME/mask.png
        cv2.imwrite(path, self.current_mask[cam])

    def filter_demo(self, cam):
        self.images_extraction(cam, debug=True)

    def demo(self):
        self.get_backward_frame(save=False, debug=True)


if __name__ == "__main__":
    demo = DataPreprocessor(config)
    demo.demo()
