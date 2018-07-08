import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import tensorflow as tf 
import keras
from keras.layers import Input, Dense, Lambda, Flatten, Reshape, Layer
from keras.layers import Conv2D, Conv2DTranspose
from keras.models import Model
from keras import backend as K
from keras import metrics
from keras.callbacks  import TensorBoard 
import sys , os 
sys.path.append("..") 
from utils import * 
from model import data_model 



class VAEAutoEncoder(data_model):
    def __init__(self , title="VAE_auto" , model_name ="vae_autoencoder" , img_shape=( 128 , 128 ,3 ) , batch_size=50):
        data_model.__init__(self , title ,model_name,
                                img_shape=img_shape,
                                epochs=100,
                                batch_size=batch_size)
        self.build()
        self.trianed_weight_path = os.path.join(self.weight_path , "vae_en_deencoder.h5")
        self.tensorboard_path = os.path.join(self.weight_path , "board")
        check_folders(self.tensorboard_path)


    def build(self):
        # All of the params
        img_rows, img_cols, img_chns = self.img_shape
        latent_dim = 64
        intermediate_dim = 256
        epsilon_std = 1.0
        epochs = 300
        filters = 64
        num_conv = 3
        batch_size = 256
        # sampling layer
        def sampling(args):
            z_mean, z_log_var = args
            epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim),
                                    mean=0., stddev=epsilon_std)
            return z_mean + K.exp(z_log_var) * epsilon
        
        x = Input(shape=self.img_shape)
        with tf.name_scope("Encoder"):
            conv_1 = Conv2D(img_chns,kernel_size=(3, 3),strides=(1, 1),padding='same', activation='relu')(x)
            conv_2 = Conv2D(filters,kernel_size=(3, 3),padding='same', activation='relu',strides=(2, 2))(conv_1)
            conv_3 = Conv2D(filters,kernel_size=num_conv,padding='same', activation='relu',strides=(2, 2))(conv_2)
            conv_4 = Conv2D(filters,kernel_size=num_conv,padding='same', activation='relu',strides=(2, 2))(conv_3)
            flat = Flatten()(conv_4)
            hidden = Dense(intermediate_dim, activation='relu')(flat)

            # mean and variance for latent variables
            z_mean = Dense(latent_dim)(hidden)
            z_log_var = Dense(latent_dim)(hidden)
            z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])
        
        # Custom loss layer
        class CustomVariationalLayer(Layer):
            def __init__(self, **kwargs):
                self.is_placeholder = True
                super(CustomVariationalLayer, self).__init__(**kwargs)

            def vae_loss(self, x, x_decoded_mean_squash):
                x = K.flatten(x)
                x_decoded_mean_squash = K.flatten(x_decoded_mean_squash)
                xent_loss = img_rows * img_cols * metrics.binary_crossentropy(x, x_decoded_mean_squash)
                kl_loss = - 0.5 * K.mean(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
                return K.mean(xent_loss + kl_loss)

            def call(self, inputs):
                x = inputs[0]
                x_decoded_mean_squash = inputs[1]
                loss = self.vae_loss(x, x_decoded_mean_squash)
                self.add_loss(loss, inputs=inputs)
                # We don't use this output.
                return x
        
        with tf.name_scope("Decoder"):
            decoder_hid = Dense(intermediate_dim, activation='relu')
            decoder_upsample = Dense(16 * 16 * filters, activation='relu')

            if K.image_data_format() == 'channels_first':
                output_shape = (batch_size, filters, 16, 16)
            else:
                output_shape = (batch_size, 16, 16, filters)

            decoder_reshape = Reshape(output_shape[1:])
            decoder_deconv_1 = Conv2DTranspose(filters,
                                            kernel_size=num_conv,
                                            padding='same',
                                            strides=(2, 2),
                                            activation='relu')
            decoder_deconv_2 = Conv2DTranspose(filters,
                                            kernel_size=num_conv,
                                            padding='same',
                                            strides=(2, 2),
                                            activation='relu')
            decoder_deconv_3_upsamp = Conv2DTranspose(filters,
                                                    kernel_size=(3, 3),
                                                    strides=(2, 2),
                                                    padding='same',
                                                    activation='relu')
            decoder_mean_squash = Conv2D(img_chns,
                                        kernel_size=(3, 3),
                                        strides=(1, 1),
                                        padding='same',
                                        activation='sigmoid')

            hid_decoded = decoder_hid(z)
            up_decoded = decoder_upsample(hid_decoded)
            reshape_decoded = decoder_reshape(up_decoded)
            deconv_1_decoded = decoder_deconv_1(reshape_decoded)
            deconv_2_decoded = decoder_deconv_2(deconv_1_decoded)
            x_decoded_relu = decoder_deconv_3_upsamp(deconv_2_decoded)
            x_decoded_mean_squash = decoder_mean_squash(x_decoded_relu)
            y = CustomVariationalLayer()([x, x_decoded_mean_squash])
        
        with tf.name_scope("TrainOps"):
            vae = Model(x, y)
            vae.compile(optimizer='rmsprop', loss=None)
        
        self.model = vae 
        vae.summary()
    @training_wrapper
    def train(self,retrain=False):
        if retrain:
            print("Loading Model")
            self.load()
        self.model.fit( x = self.data['X'] , y =self.data['y'],
                        epochs=self.nb_epochs , batch_size=self.batch_size,
                        validation_data=(self.validation['X'],self.validation['y']),
                        callbacks=[TensorBoard(log_dir=self.tensorboard_path)])    
    
if __name__ == "__main__":
    model = VAEAutoEncoder()
    