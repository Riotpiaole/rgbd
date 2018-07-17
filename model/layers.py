import keras.backend as K
from keras.layers import Input, Concatenate
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.layers.core import  Dropout, Activation, Lambda, Reshape
from keras.layers.convolutional import Conv2D, Conv2DTranspose, UpSampling2D 

def minb_disc(x):
    diffs = K.expand_dims(x, 3) - K.expand_dims(K.permute_dimensions(x, [1, 2, 0]), 0)
    abs_diffs = K.sum(K.abs(diffs), 2)
    x = K.sum(K.exp(-abs_diffs), 2)
    return x


def lambda_output(input_shape):
    return input_shape[:2]


def conv_block_unet(x, f, name, bn_mode, bn_axis, bn=True, strides=(2,2),activation=None):

    x = LeakyReLU(0.2)(x)
    x = Conv2D(f, (3, 3), strides=strides, name=name, padding="same",activation=activation)(x)
    if bn: x = BatchNormalization(axis=bn_axis)(x)

    return x


def up_conv_block_unet(x, x2, f, name, bn_mode, bn_axis, bn=True, dropout=False,activation=None):
    x = Activation("relu")(x)
    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(f, (3, 3), name=name, padding="same",activation=activation)(x)
    if bn:
        x = BatchNormalization(axis=bn_axis)(x)
    if dropout:
        x = Dropout(0.5)(x)
    x = Concatenate(axis=bn_axis)([x, x2])

    return x


def deconv_block_unet(x, x2, f, h, w, batch_size, name, bn_mode, bn_axis, bn=True, dropout=False, activation=None):
    o_shape = (batch_size, h * 2, w * 2, f)
    x = Conv2DTranspose(f, (3, 3), output_shape=o_shape, strides=(2, 2), padding="same", activation=activation)(x)
    if bn:
        x = BatchNormalization(axis=bn_axis)(x)
    if dropout:
        x = Dropout(0.5)(x)
    x = Concatenate(axis=bn_axis)([x, x2])
    return x
