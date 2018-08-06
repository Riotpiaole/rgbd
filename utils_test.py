from utils import ( 
    check_folders
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
    data = np.load ("./data/ImgSeq_Po_01/front_back_depth.npy")
    print(len(data))
    for (front , back) in data:
        print(front.shape , back.shape)