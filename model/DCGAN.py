import os 
import sys

from model import data_model
from auto_encoder import l1_loss

sys.path.insert("../")

from utils import ( 
    check_folders
)

class DCGAN( data_model):
    def __init__(
        self , 
        flag="", 
        epochs=100 , 
        img_shape = [ 256 ,256 ,3 ],
        learning_rate=1.e-3,
        white_bk=True,
        batch_size = 20,
        name="deep_conv_generative_model",
        loss = ["l1",l1_loss]):
        
        bk = "bk"
        if white_bk:
            bk = "wh"
        
        super.__init__(
            self, 
            "_%s_lr_%s_imgdim%s_loss_%s" %
                (bk,
                learning_rate,
                img_shape[0],
                loss),
            "DCGAN",
            epochs=epochs,
            batch_size=batch_size,
            white_bk=True)
        
        self.learning_rate = learning_rate 
        self.generator_loss , self.dc_loss , self.discriminator_loss = \
            tuple(loss)
        self.patch_size = [64, 64]
        self.n_batch_per_epoch = self.batch_size * 100

        # init all need dir and model
        self.build(self.img_shape)
        self.disc_weights_path = os.path.join(
            self.weight_path, "disc_weight_epoch.h5")
        self.gen_weights_path = os.path.join(
            self.weight_path, "gen_weight_epoch.h5")
        self.DCGAN_weights_path = os.path.join(
            self.weight_path, "DCGAN_weight_epoch.h5")
        check_folders(self.weight_path)
        
    
    def build(self, im_shape):
        pass 