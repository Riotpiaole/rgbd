from utils import ( 
    check_folders
)
import os 
from os import listdir, makedirs
from os.path import isfile, join, exists

def test_check_folder():
    path="../log/gan/"
    check_folders(path,verbose=True,save=True)
    assert os.path.exists(path) 
    assert os.path.exists("../log")
    os.remove("../log/gan")
    os.remove("../log/")
    