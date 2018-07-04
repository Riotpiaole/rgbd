from keras_utils_layers import generator_unet_deconv , generator_unet_upsampling # Generator 
from keras_utils_layers import DCGAN , DCGAN_discriminator # Discriminator 
from keras_utils_layers import get_nb_patch

import os
import sys
import time
import numpy as np
from keras.utils import generic_utils 
from keras.optimizers import Adam, SGD
import keras.backend as K
from model import Model

sys.path.append("../")

def l1_loss(y_true , y_pred):
    return K.sum( K.abs( y_pred  - y_true), axis=-1)

class DeepConv_Image_Translation(Model):
    def __init__( self  ,flag = "upsample" , epoch=10):
        Model.__init__(self,"Deep_Conv_Image_Translation","DCGAN")
        self.image_dim = [256,256,3]
        self.patch_size = [64,64]
        self.batch_size = 4
        self.nb_epoch = epoch
        self.build(self.image_dim)
        self.disc_weights_path = os.path.join(self.model_path , "disc_weight_epoch.h5") 
        self.gen_weights_path = os.path.join(self.model_path , "gen_weight_epoch.h5")
        self.DCGAN_weights_path = os.path.join(self.model_path, "DCGAN_weight_epoch.h5")
    
    def build(self, img_dim):
        self.generator = generator_unet_upsampling(img_dim , 2 , 
            model_name="generator_unet_upsampling")
        nb_patch , img_dim_disc = get_nb_patch(img_dim ,self.patch_size)

        # TODO test mbd
        self.discriminator = DCGAN_discriminator(img_dim_disc ,nb_patch,2,
                model_name="DCGAN_discriminator")
        
        opt_dcgan, opt_discriminator = Adam(epsilon=1e-08) ,Adam(epsilon=1e-08)
         
        self.generator.compile(loss="mae" , optimizer=opt_discriminator)
        self.discriminator.trainable =False
        
        self.DCGAN_model = DCGAN(self.generator , self.discriminator , img_dim , self.patch_size , "channels_last" )

        loss = [ l1_loss , 'binary_crossentropy']
        loss_weight = [ 1E1 , 1 ]

        self.DCGAN_model.compile(loss = loss , loss_weights= loss_weight , optimizer=opt_dcgan)

        self.discriminator.trainable = True
        self.discriminator.compile(loss="binary_crossentropy",optimizer=opt_discriminator)        
    
    def save(self):
        self.generator.save_weights( self.gen_weights_path, overwrite=True)
        self.discriminator.save_weights( self.disc_weights_path , overwrite=True)
        self.DCGAN_model.save_weights(self.DCGAN_weights_path,overwrite=True)

    def load(self):
        self.generator.load_weights(self.gen_weights_path)
        self.discriminator.load_weights(self.disc_weights_path)
        self.DCGAN_model.load_weights(self.DCGAN_weights_path)


    def train(self , label_smoothing=False):
        gen_loss, disc_loss  = 100 , 100
        total_epoch = self.nb_epoch * self.batch_size
        e_ptr = 0
        n_batch_per_epoch = 100
        if os.path.exists(self.gen_weights_path):
            print("Found prev_trained models ...")
            self.load()
            print("Retrain the model ")

        try:
            for e in range( self.nb_epoch ):
                progbar = generic_utils.Progbar(total_epoch)
                batch_counter = 1 
                
                start =time.time()
                for X , y in self.gen_batch(self.batch_size):
                    X_disc , y_disc =  self.get_disc_batch(X,y,self.generator , batch_counter ,self.patch_size,label_smoothing=label_smoothing,label_flipping=0)
                    
                    disc_loss = self.discriminator.train_on_batch(X_disc , y_disc)
                    

                    X_gen_target, X_gen = next(self.gen_batch(self.batch_size))
                    y_gen = np.zeros((X_gen.shape[0],2),dtype=np.uint8)
                    y_gen[:,1] =1
                    
                    self.discriminator.trainable = False
                    gen_loss = self.DCGAN_model.train_on_batch(X_gen , [X_gen_target, y_gen ])

                    self.DCGAN_model .trainable = True

                    progbar.add(self.batch_size, values=[("Discrminaitor logloss", disc_loss),
                                                    ("Generator tot", gen_loss[0]),
                                                    ("Generator L1 loss", gen_loss[1]),
                                                    ("Generator logloss", gen_loss[2])])

                    if batch_counter >= n_batch_per_epoch:
                        break
                print("")
                print('Epoch %s/%s, Time: %s' % (e + 1, self.nb_epoch, ms_to_hr_mins(time.time() - start)), end="\r")
                if e % 5 == 0: self.save(e)
                e_ptr = e 
        except KeyboardInterrupt:
            print("Interruption occured.... Saving the model Epochs:{}".format(e))
            self.save()

if __name__ == "__main__":
    model = DeepConv_Image_Translation()
    model.train()

        