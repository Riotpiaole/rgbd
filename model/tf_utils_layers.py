import sys
import tflearn 


sys.path.insert(0,"..")
from config import * 

import tensorflow as tf

from keras.preprocessing import image  
from tflearn.data_utils import image_preloader
from tflearn.data_preprocessing import ImagePreprocessing 
from tflearn.data_augmentation import ImageAugmentation

from tflearn.layers.core import input_data , dropout , fully_connected  , flatten
from tflearn.layers.conv import conv_2d , max_pool_2d ,upsample_2d , deconv_2d

from tflearn.activations import leaky_relu , relu
from tflearn.layers.normalization import batch_normalization

import numpy as np

def minb_disc(x,reuse=True):
    with tf.variable_scope("MiniBatchDiscrminator",reuse=reuse):
        diffs = tf.expand_dims(x,3) - tf.expand_dims(tf.transpose(x,perm=[1,2,0]) , 0)
        abs_diffs = tf.sum(tf.abs(diffs),2)
        x = tf.sum( tf.exp(-abs_diffs),2)
        return x

def lambda_output(inputshape):
    return inputshape[:2]

# =====================================
# tfLearn 
# =====================================

# helper function 
def conv_block_unet(x, filter_size, name, batch_norm_mode, batch_norm_axis, batch_norm=True , strides =[2,2],reuse=False):
    with tf.variable_scope(name,reuse=reuse):
        x = leaky_relu(x, 0.2 )
        x = conv_2d(x, filter_size, (3,3) , strides=strides, name=name)
        if batch_norm: x = batch_normalization(x)
        return x 

def up_conv_block_unet(x,x2, filter_size , name , batch_norm_mode , batch_norm_axis , batch_norm=True , dropout_=False  ,reuse=False):
    with tf.variable_scope(name,reuse=reuse):
        x = relu(x)
        x = upsample_2d(x , (2,2))
        x = conv_2d(x , filter_size , (3,3) , strides=(1,1))
        if batch_norm: x = batch_normalization(x) 
        if dropout_: x = dropout(x,.5)
        x = tf.concat([x,x2], batch_norm_axis)
        return x 

def deconv_block_unet(x, x2 , filter_size , height, width, batch_size, name , batch_norm_mode , batch_norm_axis , batch_norm = True , dropout_=False,reuse=False):
    with tf.variable_scope("deconv_conv_layer",reuse=reuse):
        o_shape = ( batch_size, height * 2, width* 2 , filter_size)
        x = relu(x)
        x = deconv_2d(x,(3,3))        
        if batch_norm: x = batch_normalization(x)
        if dropout_: x = dropout(x,.5) 
        result=tf.concat([x,x2],batch_norm_axis)
        return result

def generator_unet_upsampling(x,img_dim, bn_mode , model_name= "generator_unet_upsampling",reuse=False):
    nb_filters = 64 
    nb_channels = img_dim[-1]
    min_s = min(img_dim[:-1])
    bn_axis = -1 
    
    with tf.variable_scope("Generator_upsample",reuse=reuse):
        nb_conv = int(np.floor(np.log(min_s) / np.log(2)))
        list_nb_filters = [nb_filters * min(8, (2 ** i)) for i in range(nb_conv)]
        # Encoder  creating a deep Network
        if x == None : x = input_data(shape=img_dim)
        with tf.name_scope("Encoder"):
            list_encoder = [ conv_2d(x,list_nb_filters[0],(3,3),
                                strides=[2,2] , name="unet_conv2D_1" ,
                                padding="same")]
            
            for i, filter_size in enumerate(list_nb_filters[1:]):
                name = "unet_conv2D_%s" % (i + 2)
                conv = conv_block_unet(list_encoder[-1], filter_size, name, bn_mode, bn_axis,reuse=False)
                list_encoder.append(conv)
        # Prepare decoder filters
        list_nb_filters = list_nb_filters[:-2][::-1]
        if len(list_nb_filters) < nb_conv - 1:
            list_nb_filters.append(nb_filters)
        
        # Decoder 
        with tf.name_scope("Decoder"):
            list_decoder = [up_conv_block_unet(list_encoder[-1], list_encoder[-2],
                                       list_nb_filters[0], "unet_upconv2D_1", bn_mode, bn_axis, dropout_=True)]
            for i, filter_size in enumerate(list_nb_filters[1:]):
                name = "unet_upconv2D_%s" % (i + 2)
                # Dropout only on first few layers
                if i < 2:d = True
                else:d = False
                conv = up_conv_block_unet(list_decoder[-1], list_encoder[-(i + 3)], filter_size, name, bn_mode, bn_axis, dropout_=d)
                list_decoder.append(conv)
                x = relu(list_decoder[-1])

                x = upsample_2d(x,(2,2))
                x = conv_2d(x , nb_channels, (3, 3), name="last_conv", padding="same")
                x = tf.tanh(x)
            return x

def generator_unet_deconv(x,img_dim , bn_mode ,  batch_size, model_name="generator_unet_deconv",reuse=False):
    nb_filters = 64 
    bn_axis = -1
    h , w , nb_channels = img_dim 
    min_s = min(img_dim[:-1])
    with tf.variable_scope("Generator_deconv",reuse=reuse):
        if x == None : x = input_data(shape=img_dim)
        nb_conv = int(np.floor(np.log(min_s) / np.log(2)))
        list_nb_filters = [nb_filters * min(8, (2 ** i)) for i in range(nb_conv)]

        # Encoder
        with tf.name_scope("Encoder"):
            list_encoder = [conv_2d(x,list_nb_filters[0], (3, 3),
                                strides=(2, 2), name="unet_conv2D_1", padding="same")]
            # update current "image" h and w
            h, w = h / 2, w / 2
            for i, f in enumerate(list_nb_filters[1:]):
                name = "unet_conv2D_%s" % (i + 2)
                conv = conv_block_unet(list_encoder[-1], f, name, bn_mode, bn_axis)
                list_encoder.append(conv)
                h, w = h / 2, w / 2

        # Prepare decoder filters
        list_nb_filters = list_nb_filters[:-1][::-1]
        if len(list_nb_filters) < nb_conv - 1:
            list_nb_filters.append(nb_filters)

        # Decoder
        with tf.name_scope("Decoder"):
            list_decoder = [deconv_block_unet(list_encoder[-1], list_encoder[-2],
                                            list_nb_filters[0], h, w, batch_size,
                                            "unet_upconv2D_1", bn_mode, bn_axis, dropout_=True)]
            h, w = h * 2, w * 2
            for i, f in enumerate(list_nb_filters[1:]):
                name = "unet_upconv2D_%s" % (i + 2)
                # Dropout only on first few layers
                if i < 2:
                    d = True
                else:
                    d = False
                conv = deconv_block_unet(list_decoder[-1], list_encoder[-(i + 3)], f, h,
                                        w, batch_size, name, bn_mode, bn_axis, dropout_=d)
                list_decoder.append(conv)
                h, w = h * 2, w * 2
                x = relu(x)
                x = deconv_2d( x , (3,3), name="deconv_2d")
                x = tanh(x)
        
            
        return x

def PatchGAN(x,img_dim,patch_num,reuse=False,mbd=False):
    bn_axis = -1 
    nb_filters = 64 
    nb_conv = int(np.floor(np.log(img_dim[1]) / np.log(2)))
    list_filters = [nb_filters * min(8, (2 ** i)) for i in range(nb_conv)]
    patch_num = str(patch_num)
    if not mbd:scope_name,first_conv_name  = "PatchGAM_"+patch_num , "disc_conv2d_1_%s" % (patch_num)
    else: scope_name,first_conv_name = "MBD_PatchGAM_"+patch_num ,"MBD_disc_conv2d_1_%s" % (patch_num)

    with tf.variable_scope(scope_name,reuse=reuse):
        x = conv_2d(x,list_filters[0],(3,3), strides=(2,2), name=first_conv_name, padding="same")
        x = batch_normalization(x)
        x = leaky_relu(x , .2)

        for i , filter_size in enumerate(list_filters[1:]):
            if not mbd:name = "disc_conv2d_%s_%s" % (i + 2,patch_num)
            else: name = "MBD_disc_conv2d_%s_%s" % (i + 2,patch_num)
            x = conv_2d( x , filter_size , ( 3,3), strides=(2,2), name = name, padding ="same")
            x = batch_normalization(x)
            x = leaky_relu( x )
        sys.exit(0)
        x_flat = flatten(x)
        x = tf.layers.dense(x_flat,2,activation=tf.nn.softmax, name="disc_dense_"+patch_num)
        return tf.concat([x,x_flat],-1)
              
def DCGAN_discriminator(list_input,img_dim, bn_mode , model_name="DCGAN_Discrminiator",reuse=False):
    bn_axis = -1 
    nb_filters = 64 
    nb_conv = int(np.floor(np.log(img_dim[1]) / np.log(2)))
    list_filters = [nb_filters * min(8, (2 ** i)) for i in range(nb_conv)]
    if list_input == None: list_input = [input_data(shape=img_dim, name="disc_input_%s" % i) for i in range(nb_patch)]
    # First Conv 
    with tf.variable_scope("Discrminator",reuse=reuse):
        x = [ PatchGAN(patch,img_dim,patch_num,reuse=False)[0] for patch_num,patch in enumerate(list_input) ]
        x_mbd = [ PatchGAN(patch,img_dim,patch_num,reuse=False,mbd=True)[1] for patch_num,patch in enumerate(list_input) ]
        
        if len(x) > 1: x = tf.concat( x , -1 )
        else: x = x [0]
        
        # Dont Understand #TODO Ask Po
        if use_mbd:
            if len(x_mbd) > 1: x_mbd = tf.concat(x_mbd,-1)
            else: x_mbd = x_mbd[0]
            num_kernels = 100
            dim_per_kernel = 5 
        
            M = tf.layers.dense(x_mbd, num_kernels*dim_per_kernel,use_bias=True ,activation=None )
            MBD = minb_disc(minb_disc)
        x_out = tf.layers.dense(x,2 , activation="softmax" , name="disc_ouput")
        return x_out 

def DCGAN(img_dim , patch_size=(64,64)):
    gen_input = input_data(shape=img_dim , name="DCGAN_input")
    
    generated_img = generator_unet_upsampling(gen_input, img_dim , 0)
    h , w = img_dim[:-1]
    (ph , pw) = patch_size 
    

    list_row_idx = [(i * ph, (i + 1) * ph) for i in range(h // ph)]
    list_col_idx = [(i * pw, (i + 1) * pw) for i in range(w // pw)]
    
    list_gen_patch = []
    for row_idx in list_row_idx:
        for col_idx in list_col_idx:
            x_patch = generated_img[:,row_idx[0]:row_idx[1],col_idx[0]:col_idx[1], :]
            list_gen_patch.append(x_patch)
    
    # stacked GEN and discriminator 
    discriminator = DCGAN_discriminator(list_gen_patch,img_dim,4,0)
    # generator

if __name__ == "__main__":
    DCGAN((256,256,3))