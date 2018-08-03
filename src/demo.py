from auto_encoder import DeepConvAutoEncoder 
from utils import showImageSet , check
import matplotlib.pyplot as plt 

if __name__  == "__main__":
    model = DeepConvAutoEncoder(
        epoch=0,
        img_shape=[256,256,3],
        learning_rate=1e-5,
        loss="l1",
        white_bk=False )
    model.load()
    model.demo("ImgSeq_Po_00_first_test",save=True)
    