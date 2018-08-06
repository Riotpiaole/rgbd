import sys
import os
from time import time
from os.path import isfile, join

# import config
from config import config
from threading import Thread

import cv2
import sys
import numpy as np

from numpy.linalg import inv
import h5py
# Must have function for image filtering
from utils import (
    showImage,
    transform_pointcloud_vectorized,
    multi_process_wrapper,
    scatter_point_filtering,
    unproject_pointcloud,
    click_region_call_bk,
    convert_depth_2_rgb,
    reproject_ptcloud,
    compute_img_diff,
    black_bg,
    white_bg,
    chunks,
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


def two_camera_reprojection(
        iteration,
        cam_params,
        sensor_props,
        result,
        rgb1,
        rgb2,
        dep1,
        dep2,
        debug=False,
        radius=2,
        suffix=""
):
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
        iteration, img_pts, img_reproj, radius=radius, suffix=suffix)

    img_reproj = scatter_point_filtering(img_reproj)
    if debug:
        showImageSet([debug_diff, img_reproj], [
                     "without_inpainted", "with_inpainted"])
    result[:] = img_reproj


class DataPreprocessor():
    def __init__(
        self,
        config,
        img_shape=(
            240,
            320),
        num_cam=3,
        debug_mode=False,
        redo=False
    ):

        self.frame_num = 0
        self.config = config
        self.debug_mode = debug_mode
        self.data = [[], [], []]
        self.current_mask = [None, None, None]

        self.masks_dir = "./mask/%s/masks.npy" % self.config.strFolderName

        # calibraring
        if redo:
            os.system("rm -rf %s" % self.masks_dir)

        self.masks = [[], [], []]

        self.datasets = []
        self.datasets_bk = []
        
        self.front_back_depth_map = []

        self.num_cam = num_cam

    # actual filtering
    @timeit(log_info="Obtain all the traning data ")
    def get_backward_frame(self, debug=False, save=False):
        # for each pair of camera images

        for frame_num, (
                (image_cam1_color, image_cam1_depth),
                (image_cam2_color, image_cam2_depth),
                (image_cam3_color, image_cam3_depth),
                (mask1, mask2, mask3)
            ) in enumerate(
            zip(*self.data,
                zip(*self.masks)
                )
        ):

            # ===============================================================
            # Mask remove black color
            # ===============================================================

            #  re-do unprojection for predicting the images
            mask1 = np.dstack((mask1, mask1, mask1)).astype(np.uint8)
            mask2 = np.dstack((mask2, mask2, mask2)).astype(np.uint8)
            mask3 = np.dstack((mask3, mask3, mask3)).astype(np.uint8)

            img_reproj = np.zeros(
                image_cam1_color.shape,
                image_cam1_color.dtype)
            
            mask_reproj = np.zeros(
                image_cam1_color.shape,
                image_cam1_color.dtype)
            
            depth_reproj = np.zeros( 
                image_cam1_color.shape,
                image_cam1_depth.dtype
            )


            repoj_args = [
                frame_num,
                self.cam_params,
                self.sensor_props,
                img_reproj,
                image_cam2_color,
                image_cam3_color,
                image_cam2_depth,
                image_cam3_depth,
                False,
                self.config.radius,
                "%s img_reproj_reproj" % self.config.strFolderName]
            
            #  seting up params for reprojecting the mask
            mask_args = repoj_args.copy()
            mask_args[3] = mask_reproj
            mask_args[4] = mask2
            mask_args[5] = mask3
            mask_args[10] = "%s mask_reproj" % self.config.strFolderName
            
            depth_args = repoj_args.copy()
            depth_args[3] = depth_reproj
            depth_args[4] = np.dstack((
                image_cam2_depth,
                image_cam2_depth,
                image_cam2_depth
            ))

            depth_args[5] = np.dstack((
                image_cam3_depth,
                image_cam3_depth,
                image_cam3_depth,
            ))
            depth_args[10] = "%s depth_reproj" % self.config.strFolderName

            if debug:
                two_camera_reprojection(*repoj_args)
                two_camera_reprojection(*mask_args)
                two_camera_reprojection(*depth_args)
            else:
                threads = []
                for arg in [repoj_args, mask_args , depth_args]:
                    process = Thread(
                        target=two_camera_reprojection, args=[
                            *arg])
                    process.start()
                    threads.append(process)

                for thread in threads:
                    thread.join()

            img_reproj_bk, img_cam1_color_bk = black_bg(
                img_reproj[:].copy(), image_cam1_color[:].copy(),
                mask1, mask_reproj)

            img_reproj_wh, img_cam1_color_wh = white_bg(
                img_reproj[:].copy(), image_cam1_color[:].copy(),
                mask1, mask_reproj)
            
            depth_reproj[ mask_reproj == 0 ] = 0
            depth_reproj = np.dsplit(depth_reproj,3)[0]
            depth_reproj = depth_reproj.reshape ( (240,320))
            self.front_back_depth_map.append(
                (image_cam1_depth , depth_reproj)
            )

            if debug:
                showImageSet([img_reproj_bk, img_cam1_color_bk, img_reproj_wh, img_cam1_color_wh],
                             ["front_bk", "back_bk", "front_wh", "back_wh"])

            #  folders to be saved on
            save_path = self.config.strFilterFullPath
            save_path_bk = self.config.strFilterFullPathBlack
            # check_folders(save_path_bk)

            if save:
                self.save(
                    save_path,
                    frame_num,
                    img_cam1_color_wh,
                    img_reproj_wh)
                self.save(
                    save_path_bk,
                    frame_num,
                    img_cam1_color_bk,
                    img_reproj_bk,
                    bk=True)
            print("")

    def save(self, save_path, index, train, label, bk=False):
        save_train_folder_path, save_target_folder_path = os.path.join(
            save_path, "train"), os.path.join(save_path, "target")

        check_folders(save_train_folder_path)
        check_folders(save_target_folder_path)

        if bk:
            self.datasets_bk.append((train, label))
        else:
            self.datasets.append((train, label))

    def get_rgbd(self, cam, frame_num):
        img_depth = cv2.imread(
            self.sensor_props[cam].imgs_depth[frame_num], -1)
        img_clr = cv2.imread(self.sensor_props[cam].imgs_color[frame_num])
        return img_clr, img_depth

    @timeit(log_info="Loading all rgbd images")
    def load_rgbd_imgs(self):
        threads = []

        def get_rgbds(
            obj, cam): return [
            obj.data[cam].insert(
                num, obj.get_rgbd(
                    cam, num)) for num in range(
                obj.total_frame_num)]

        for cam in range(self.num_cam):
            process = Thread(target=get_rgbds, args=[self, cam])
            process.start()
            threads.append(process)
        for thread in threads:
            thread.join()

    def load_data(self):
        os.system("clear")
        print("========================================================================================================")
        print("Preprocessing dataset from \n" + str(self.config))
        self.sensor_props = ReadManualCalibPoses(self.config.strPoseLocation)
        for s in self.sensor_props:
            s.load_image_files(self.config.strVideoFullPath)

        self.cam_params = GetCameraParameters("OrbbecAstra", 0.5)
        self.total_frame_num = len(self.sensor_props[0].imgs_color)

        self.load_rgbd_imgs()

        # Obtain all initial mask
        for cam in range(self.num_cam):
            if os.path.isfile(self.masks_dir):
                self.masks = np.load(self.masks_dir)
            else:
                self.extract_initial_mask(cam)

            self.current_mask[cam] = self.masks[cam][0]

        # if the mask is interrupted in the middle of storage
        if not len(self.masks[0]) == self.config.num_images:
            self.rgbd_filtering(debug=False)
        print("========================================================================================================")

    def rgbd_filtering(self, debug=False):
        if debug:
            print("%s debug verbose 1 ....." % self.config.strFolderName)
            for camera in range(self.num_cam):
                self.images_extraction(camera, debug)
                print("Running camera extraction ", camera)
        else:
            print("%s debug verbose 0 ....." % self.config.strFolderName)
            print("Filtering images")
            self.threaded_images_extraction(self)
            print("Saving total %s images to %s " %
                  (len(self.masks[0]), self.masks_dir))
            np.save(
                "./mask/%s/masks.npy" %
                self.config.strFolderName,
                self.masks)

    #  wrapper function for multi-thread computing the image background
    @staticmethod
    @multi_threads_wrapper([0, 1, 2])
    def threaded_images_extraction(*args):
        arg, cam, iteration = args
        for frame_num in range(1, arg.total_frame_num):
            arg.background_extraction(
                cam, frame_num, save=True)  # remove background

    def extract_initial_mask(self, cam):
        clr, depth = self.get_rgbd(cam, 0)
        next_mask, num_mask = self.connected_comp_labeling(
            cam, depth, debug=False)
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
        self.masks[cam].append(result.astype(np.uint8))

    def images_extraction(self, cam, debug=False):
        start = time()
        for frame_num in range(1, self.total_frame_num):
            self.background_extraction(
                cam, frame_num, debug=debug, save=False)  # remove background
        end = time()
        time_taken = round(end - start, 2)
        print(
            "Finished extracting camera {}, Time: {} ms".format(
                cam, time_taken))

    def background_extraction(self, cam, frame_num, debug=False, save=False):
        check_folders("./mask/{}".format(self.config.strFolderName))

        img_color, img_depth = self.get_rgbd(cam, frame_num)
        next_mask, num_mask = self.connected_comp_labeling(
            cam, img_depth, debug=False)
        result = None

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
            self.masks[cam].append(result.astype(np.uint8))
        mask = np.dstack((result, result, result)).astype(np.uint8)
        clr = np.bitwise_and(img_color, mask)

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

    def demo(self):
        '''extract the images background and front with opencv'''
        self.get_backward_frame(save=False, debug=True)

    def unzip(self, folder):
        ''' unpack numpy.ndarray to collections of png images train and label'''
        data_dir = "./data/%s" % folder
        if os.path.isfile(data_dir + "/images.npy"):
            check_folders(data_dir + "/train")
            check_folders(data_dir + "/target")
            data = np.load(data_dir + "/images.npy")

            @multi_threads_wrapper(list(chunks(data, 100)))
            def save_unzip_imgs(*args):
                data, iteration = args
                for frame_num, (X, y) in enumerate(data):
                    train_img = data_dir + \
                        "/train/train%s.png" % str(iteration + frame_num)
                    label = data_dir + \
                        "/target/target%s.png" % str(iteration + frame_num)
                    cv2.imwrite(train_img, X)
                    cv2.imwrite(label, y)
            save_unzip_imgs()
        else:
            print("Pre-ziped image was not found ina%s" % data_dir)

    def make_dataset(self, npy=False , depth=False):
        if os.path.isfile("./data/%s" % self.config.strFolderName):
            self.get_backward_frame(save=True, debug=False)
        self.get_backward_frame(save=False, debug=False)
        if npy:
            np.save(
                "./data/%s/images.npy" %
                self.config.strFolderName,
                self.datasets)
            np.save(
                "./data/%s/images_bk.npy" %
                self.config.strFolderNameBlack,
                self.datasets_bk)
        
        if depth:
            print("Saving %s depth map with reprojected and masked" % self.config.strVideoFolder)
            np.save( 
                "./data/%s/front_back_depth.npy" %
                self.config.strFolderName, 
                self.front_back_depth_map
            )   

        print("========================================================================================================")
        print("Saving all training images in %s | with totoal %d images |" % (
            self.config.strFolderName,
            len(self.datasets)
        )
        )
        print("========================================================================================================")

    def unzip_npy_to_imgs(self):
        print("========================================================================================================")
        print("unpack all images to directory %s" % self.config.strFolderName)
        self.unzip(self.config.strFolderName)
        self.unzip(self.config.strFolderNameBlack)
        print("========================================================================================================")


if __name__ == "__main__":
    config = config("ImgSeq_Po_01")
    demo = DataPreprocessor(config, debug_mode=False)
    demo.load_rgbd_imgs()
    demo.load_data()
    demo.make_dataset(depth=True)
    # demo.make_dataset()
