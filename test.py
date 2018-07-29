import os
import numpy as np
import cv2
from utils import (
    chunks,
    check_folders,
    multi_threads_wrapper
)
data = np.load(
    "/Users/rockliang/Documents/Research/VISION/RGBD/rgbd/data/ImgSeq_Po_00_first_test/images.npy")
result = chunks(data, 100)


def unzip():
    ''' unpack numpy.ndarray to collections of png images train and label'''
    data_dir = "./data/ImgSeq_Po_00_first_test"
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
