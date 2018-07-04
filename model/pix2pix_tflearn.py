from tf_utils_layers import * 
from model import Model 
class Conv_GAN(Model):
    def __init__(self):
        Model.__init__("DCGAN","DCGAN")
        img_dim =(256,256,3)
        self.dcgan = DCGAN(img_dim)
        self.gen = generator_unet_deconv(None,img_dim,0,10)
        # create a DNN training generator model 
        self.gen = tflearn.regression(self.gen,loss="mae",optimizer="adam")
        

        # create a DNN training discriminator 