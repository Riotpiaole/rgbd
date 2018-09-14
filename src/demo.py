import cv2 , os 
from tqdm import tqdm
from auto_encoder import DeepConvAutoEncoder 
from utils import (
    showImage,
    showImageSet , 
    inverse_normalization,
    check,
    listfiles_nohidden,
    vectorized_read_img
)
import matplotlib.pyplot as plt 



if __name__  == "__main__":
    model = DeepConvAutoEncoder(
        epoch=0,
        img_shape=[256,256,3],
        learning_rate=1e-5,
        loss="l1",
        white_bk=True )
    model.load()
    model.demo("ImgSeq_Po_01",imgs=True)
    # video_name = os.path.join("/Users/rockliang/Downloads/input_color","input_colors.avi")
    # fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    # video = cv2.VideoWriter(video_name , fourcc , 50 ,(256 ,256), True)   
    