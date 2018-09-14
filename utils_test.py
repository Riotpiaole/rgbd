from utils import ( 
    check_folders,
    showImageSet,
    convert_depth_2_rgb
)

import os 
import numpy as np 
from os import listdir, makedirs
from os.path import isfile, join, exists

def test_check_folder():
    path="../log/gan/"
    check_folders(path,verbose=True,save=True)
    assert os.path.exists(path) 
    assert os.path.exists("../log")
    os.remove("../log/gan")
    os.remove("../log/")
    

if __name__ == "__main__":
    data = np.load ("./data/ImgSeq_Liang_01black/images.npy")
    print(len(data[0]))
    for (front , back) in data:
        # front , back = convert_depth_2_rgb(front),\
        #     convert_depth_2_rgb(back)
        showImageSet([front , back] , ["front" , "back"])