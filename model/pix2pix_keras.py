import os
import sys
from time import time 
import random as rnd

sys.path.append("../")

import h5py ,math
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt 

# keras module 
import keras.backend as K
from keras.optimizers import Adam
from keras.utils import generic_utils 
from keras.preprocessing import image

# internal module 
from model import data_model ,inverse_normalization 
from models import generator_unet_deconv, DCGAN_discriminator, DCGAN
from utils import timeit , check_folders  ,plot_generated_batch ,get_nb_patch

def l1_loss(y_true , y_pred):
    return K.sum( K.abs( y_pred  - y_true), axis=-1)

class  K_DCGAN(data_model):
    def __init__( self  ,flag = "deconv" , epoch=100000,img_shape = [256,256,3]):
        data_model.__init__(self,"K_DCGAN_dim_256_relu","DCGAN",img_shape=img_shape,epochs=epoch)
        # training params 
        self.patch_size = [64,64]
        self.n_batch_per_epoch = self.batch_size * 100
        
        # init all need dir and model
        self.build(self.img_shape)
        self.disc_weights_path = os.path.join(self.weight_path , "disc_weight_epoch.h5") 
        self.gen_weights_path = os.path.join(self.weight_path , "gen_weight_epoch.h5")
        self.DCGAN_weights_path = os.path.join(self.weight_path, "DCGAN_weight_epoch.h5")
        check_folders(self.weight_path)
    
    def build(self, img_shape):
        self.generator = generator_unet_deconv(img_shape , 2 ,  self.batch_size,
            model_name="generator_unet_deconv",activation="relu")
            
        nb_patch , img_shape_disc = get_nb_patch(img_shape ,self.patch_size)

        self.discriminator = DCGAN_discriminator(img_shape_disc ,nb_patch,2,
                model_name="DCGAN_discriminator")
        
        opt_dcgan, opt_discriminator = Adam(lr=0.0001,epsilon=1e-08) ,Adam(lr=0.0001,epsilon=1e-08)
         
        self.generator.compile(loss="categorical_crossentropy" , optimizer=opt_discriminator)
        self.discriminator.trainable =False
        
        self.DCGAN_model = DCGAN(self.generator , self.discriminator , img_shape , self.patch_size , "channels_last" )

        loss = [ l1_loss , 'binary_crossentropy']
        loss_weight = [ 1E1 , 1 ]

        self.DCGAN_model.compile(loss = loss , loss_weights= loss_weight , optimizer=opt_dcgan)

        self.discriminator.trainable = True
        self.discriminator.compile(loss="binary_crossentropy",optimizer=opt_discriminator)        

    def log_checkpoint(self,epoch , batch, loss):
        log_path =os.path.join(self.weight_path , "checkpoint")
        
        prev_epochs , prev_batch_size =  0 , 0
        if os.path.isfile(log_path):
            with open( log_path, "w+") as f:
                line = f.readline()
                if "Epoch" in line: 
                    line = f.readline().split(" ")
                    prev_epochs  = int(line[4])
        
        with open(log_path , "w+") as f:
            f.write( "Model_Name {} ".format(self.title))
            f.write( "Epoch {} in batch {}".format( 
                epoch + prev_epochs ,
                batch ))
            f.write( "\n")
            f.write( "Losses: {}".format( loss ))
   
    def save(self):
        if not os.path.exists(self.gen_weights_path):
            h5py.File(self.gen_weights_path)
            h5py.File(self.disc_weights_path)
            h5py.File(self.DCGAN_weights_path)
        self.generator.save_weights( self.gen_weights_path, overwrite=True)
        self.discriminator.save_weights( self.disc_weights_path , overwrite=True)
        self.DCGAN_model.save_weights(self.DCGAN_weights_path,overwrite=True)

    def test_img(self):
        idx = rnd.choice([ i for i in range(0 , len(self.data['X']) )]) # pick a random index
        
        X , y = self.get_data( idx ) # normalized images
        self.load()
                
        X_pred = self.generator.predict(np.array([X]))
        X = image.array_to_img(inverse_normalization(X,self.max , self.min))
        y = image.array_to_img(inverse_normalization(y,self.max , self.min))
        X_pred = image.array_to_img(inverse_normalization(X_pred[0],self.max , self.min))
        
        suffix = "End_test"

        result = np.hstack((X ,y , X_pred))
        
        check_folders("../figures/%s" % (self.title) )
        plt.savefig("../figures/%s/current_batch_%s.png" % (self.title,suffix))
        plt.imshow(result)
        plt.axis("off")
        plt.show()

    def load(self):
        '''Load models weight from log/${model_name}'''
        if os.path.exists(self.gen_weights_path):
            self.generator.load_weights(self.gen_weights_path)
            self.discriminator.load_weights(self.disc_weights_path)
            self.DCGAN_model.load_weights(self.DCGAN_weights_path)    
        else:
            raise FileNotFoundError("No Previous Model Found") 
        print("Loading model  from {}".format([self.disc_weights_path,
                                                self.gen_weights_path,
                                                self.DCGAN_weights_path]))

    def summary(self ,name="DCGAN"):
        if name == "Generator":
            self.generator.summary()
        elif name == "Discriminator":
            self.discriminator.summary()
        else:
            self.DCGAN_model.summary()

    @timeit(log_info="Training pix2pix")
    def train(self , label_smoothing=False,retrain=False):
        gen_loss, disc_loss  = 100 , 100
        n_batch_per_epoch = self.n_batch_per_epoch
        total_epoch = n_batch_per_epoch * self.batch_size
        try:
            if retrain:
                print("Found prev_trained models ...")
                self.load()
                print("Retrain the model ")
        except FileNotFoundError:
            print("No previous model found start train a new model")
        
        try:
            os.system("clear")
            for e in range( self.nb_epochs ):
                batch_counter = 1 
                start = time()
                progbar = generic_utils.Progbar(total_epoch)

                for X , y in self.gen_batch(self.batch_size):
                    
                    X_disc , y_disc =  self.get_disc_batch(X,y,self.generator, 
                            batch_counter ,
                            self.patch_size,
                            label_smoothing=label_smoothing,
                            label_flipping=0)
                    
                    disc_loss = self.discriminator.train_on_batch(X_disc , y_disc)
                    
                
                    X_gen_target, Y_gen= next(self.gen_batch(self.batch_size))
                    
                    self.generator.train_on_batch(X_gen_target , Y_gen)
                    
                    y_gen = np.zeros((Y_gen.shape[0],2),dtype=np.uint8)
                    y_gen[:,1] =1
                    
                    self.discriminator.trainable = False
                    gen_loss = self.DCGAN_model.train_on_batch(X_gen_target , [Y_gen, y_gen ])
                    
                    
                    self.DCGAN_model.trainable = True
                    
                    batch_counter += 1
                    progbar.add(self.batch_size, values=[("D logloss", disc_loss),
                                                    ("G tot", gen_loss[0]),
                                                    ("G L1", gen_loss[1]),
                                                    ("G logloss", gen_loss[2])])
                    if batch_counter % (n_batch_per_epoch / 2) == 0:
                        # Get new images from validation
                        plot_generated_batch(X, y, self.generator,self.batch_size, "channels_last", "training",self.title,self)
                        X_test, y_test = next(self.gen_batch(self.batch_size , validation=True)) # get next validation batches
                        plot_generated_batch(X_test, y_test, self.generator,self.batch_size, "channels_last", "validation",self.title,self)

                    if batch_counter >= n_batch_per_epoch:
                        break 

                print("")
                t_time =time() - start
                print('Epoch %s/%s, Time: %s ms' % (e + 1, self.nb_epochs, round(t_time,2) ),end="\r")
                if e % 5 == 0: 
                    self.save()
                    self.log_checkpoint(e , batch_counter , 
                                                    [("D logloss", disc_loss),
                                                    ("G tot", gen_loss[0]),
                                                    ("G L1", gen_loss[1]),
                                                    ("G logloss", gen_loss[2])])
                
                
        except KeyboardInterrupt:
            print("\nInterruption occured.... Saving the model Epochs:{}".format(e))
            self.save()
            self.log_checkpoint(e , batch_counter , [("D logloss", disc_loss),
                                                    ("G tot", gen_loss[0]),
                                                    ("G L1", gen_loss[1]),
                                                    ("G logloss", gen_loss[2])])


if __name__ == "__main__":
    model = K_DCGAN()
    model.train(retrain=True)
    model.test_img()
        
