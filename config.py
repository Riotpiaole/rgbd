import os
import numpy as np
from utils import multi_threads_wrapper
# Folder of all the images Non changing data
strVideoFolder = "./unity-multiview/data"
strFilterFolder = "./data"

# calibration parameters for image filtering
calibs = {
    "ImgSeq_Po_00_first_test": {
        "radius": 3,
        "cam1_calib": {'maxDep': 3000, 'lx': 40, 'rx': 250, 'yd': None, 'open': (2, 2), 'close': (2, 2), 'ksize': (5, 5), 'dsize': 10, 'bilater': (30, 50), 'grad': 500, 'connectivity': 520},
        "cam2_calib": {'maxDep': 3500, 'lx': 140, 'rx': None, 'yd': 215, 'open': (2, 2), 'close': (2, 2), 'ksize': (5, 5), 'dsize': 8, 'bilater': (10, 65), 'grad': 60, 'connectivity': 8},
        "cam3_calib": {'maxDep': 3500, 'lx': 50, 'rx': 220, 'yd': 160, 'open': (3, 3), 'close': (5, 5), 'ksize': (3, 3), 'dsize': 7, 'bilater': (10, 30), 'grad': 2000, 'connectivity': 20},
    },
    "ImgSeq_Liang_02_CapsShield": {
        "radius": 0,
        "cam1_calib": {'maxDep': 4500, 'lx': 40, 'rx': 250, 'yd': None, 'open': (2, 2), 'close': (2, 2), 'ksize': (5, 5), 'dsize': 10, 'bilater': (30, 50), 'grad': 10, 'connectivity': 3},
        "cam2_calib": {'maxDep': 3500, 'lx': 0, 'rx': None, 'yd': None, 'open': (2, 2), 'close': (2, 2), 'ksize': (5, 5), 'dsize': 8, 'bilater': (10, 65), 'grad': 60, 'connectivity': 8},
        "cam3_calib": {'maxDep': 3500, 'lx': 50, 'rx': 220, 'yd': 160, 'open': (3, 3), 'close': (5, 5), 'ksize': (3, 3), 'dsize': 7, 'bilater': (10, 30), 'grad': 2000, 'connectivity': 20},
    },
    "ImgSeq_Po_02_Bag": {
        "radius": 2,
        "cam1_calib": {'maxDep': 5500, 'lx': 40, 'rx': 250, 'yd': None, 'open': (2, 2), 'close': (2, 2), 'ksize': (3, 3), 'dsize': 3, 'bilater': (30, 50), 'grad': 100, 'connectivity': 520},
        "cam2_calib": {'maxDep': 3000, 'lx': 60, 'rx': 200, 'yd': 0, 'open': (2, 2), 'close': (2, 2), 'ksize': (5, 5), 'dsize': 8, 'bilater': (10, 65), 'grad': 60, 'connectivity': 8},
        "cam3_calib": {'maxDep': 3000, 'lx': 50, 'rx': 220, 'yd': 160, 'open': (3, 3), 'close': (5, 5), 'ksize': (3, 3), 'dsize': 7, 'bilater': (10, 30), 'grad': 2000, 'connectivity': 20},
    },
    "ImgSeq_Liang_01": {
        "radius": 2,
        "cam1_calib": {'maxDep': 4200, 'lx': None, 'rx': None, 'yd': None, 'open': (2, 2), 'close': (2, 2), 'ksize': (2, 2), 'dsize': 10, 'bilater': (30, 50), 'grad': 500, 'connectivity': 4},
        "cam2_calib": {'maxDep': 4100, 'lx': None, 'rx': None, 'yd': None, 'open': (2, 2), 'close': (2, 2), 'ksize': (5, 5), 'dsize': 8, 'bilater': (10, 65), 'grad': 60, 'connectivity': 8},
        "cam3_calib": {'maxDep': 4000, 'lx': None, 'rx': 220, 'yd': 160, 'open': (3, 3), 'close': (5, 5), 'ksize': (3, 3), 'dsize': 3, 'bilater': (10, 30), 'grad': 2000, 'connectivity': 10},
    },

    "ImgSeq_Po_01": {
        "radius": 2,
        "cam1_calib": {'maxDep': 3500, 'lx': 40, 'rx': 250, 'yd': None, 'open': (2, 2), 'close': (2, 2), 'ksize': (5, 5), 'dsize': 10, 'bilater': (10, 50), 'grad': 500, 'connectivity': 520},
        "cam2_calib": {'maxDep': 4000, 'lx': None, 'rx': None, 'yd': 215, 'open': (2, 2), 'close': (2, 2), 'ksize': (5, 5), 'dsize': 8, 'bilater': (10, 65), 'grad': 100, 'connectivity': 8},
        "cam3_calib": {'maxDep': 3500, 'lx': 40, 'rx': 220, 'yd': 160, 'open': (3, 3), 'close': (5, 5), 'ksize': (3, 3), 'dsize': 7, 'bilater': (10, 30), 'grad': 2000, 'connectivity': 20},
    },
    "ImgSeq_Po_03_RedShirt": {
        "radius": 2,
        "cam1_calib": {'maxDep': 3000, 'lx': 40, 'rx': 250, 'yd': None, 'open': (2, 2), 'close': (2, 2), 'ksize': (5, 5), 'dsize': 10, 'bilater': (30, 50), 'grad': 500, 'connectivity': 520},
        "cam2_calib": {'maxDep': 4000, 'lx': None, 'rx': None, 'yd': None, 'open': (2, 2), 'close': (2, 2), 'ksize': (5, 5), 'dsize': 8, 'bilater': (10, 65), 'grad': 60, 'connectivity': 8},
        "cam3_calib": {'maxDep': 3500, 'lx': 50, 'rx': 220, 'yd': 160, 'open': (3, 3), 'close': (5, 5), 'ksize': (3, 3), 'dsize': 7, 'bilater': (10, 30), 'grad': 2000, 'connectivity': 20},
    },
}

# Each Stream of data that represent a set of datas


class config(object):
    '''configuration object for '''

    def __init__(self, dataset_name):
        self.strFolderName = dataset_name
        self.strVideoFolder = strVideoFolder
        self.strFilterFolder = strFilterFolder
        self.strFolderNameBlack = dataset_name + "black"

        self.strVideoFullPath = os.path.join(
            strVideoFolder, self.strFolderName)
        self.strPoseLocation = os.path.join(
            self.strVideoFullPath, "unity3d_poses.txt")

        self.strFilterFullPath = os.path.join(
            strFilterFolder, self.strFolderName)
        self.strFilterFullPathBlack = os.path.join(
            strFilterFolder, self.strFolderNameBlack)
        self.radius = calibs[self.strFolderName]["radius"]

        self.cam1_calib = calibs[self.strFolderName]["cam1_calib"]
        self.cam2_calib = calibs[self.strFolderName]["cam2_calib"]
        self.cam3_calib = calibs[self.strFolderName]["cam3_calib"]

        if os.path.exists( self.strVideoFullPath ):
            self.num_images = len(
                os.listdir(
                    self.strVideoFullPath +
                    "/cam01/color"))
        elif os.path.isfile( "../data/%s/images.npy" % self.strFolderName ):
            self.num_images = np.load(
                    "../data/%s/images.npy" % (
                        self.strFolderName )
                ).shape[0]

    @property
    def filter_param(self):
        return{self.strFolderName: [self.cam1_calib, self.cam2_calib, self.cam3_calib]}

    def __len__(self):
        return self.num_images

    @property
    def input_shape(self):
        return (240, 320, 3)

    def __str__(self):
        result = "---------------------------------------------------------------------------"
        result += "\n|Configuration: %s | radius filtering %d | num_of_imgs %d|\n" % (
            self.strFolderName, self.radius, self.num_images)
        result += "---------------------------------------------------------------------------"
        return result


class streams_config(object):
    def __init__(self):
        self.all_datasets = {}
        self.total_imgs = 0
        for folder in calibs.keys():
            temp = config(folder)
            self.total_imgs += temp.num_images
            self.all_datasets[folder] = temp

    def process(self, name):
        return self.all_datasets[name]

    @property
    def to_list(self):
        return sorted(list(self.all_datasets.values()), key=len)
        # return list(self.all_datasets.values())


if __name__ == "__main__":
    streams_config = streams_config()
