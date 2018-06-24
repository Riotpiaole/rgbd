import tflearn
import tensorflow as tf 
from tflearn.layers.core import input_data , dropout , fully_connected 
from tflearn.layers.conv import conv_2d , max_pool_2d



def autoencoder(input_shape):
    # MNIST autoencoder 
    encoder = input_data(shape=[None , input_shape])
    
    # encoder 
    with tf.name_scope("Encoder"):
        encoder = fully_connected(encoder , 256)
        encoder = fully_connected(encoder , 64)

    # decoder 
    with tf.name_scope("Decoder"):
        decoder = fully_connected(encoder, 256)
        decoder = fully_connected(decoder, input_shape , activation='sigmoid')

    network = tflearn.regression(decoder , optimizer='adam',
                            learning_rate = .001 ,
                            loss="mean_square",
                            metric=None)
    return network

def conv_autoencoder(input_shape):
    img_prep = ImagePreprocessing()
    img_prep.add_featurewise_zero_center()
    img_prep.add_featurewise_stdnorm()

    with tf.name_scope("Encoder"):
        encoder = input_data(shape=(None, input_shape[0] , input_shape[1] , input_shape[2]),
                            data_preprocessing=img_prep)
        
        encoder = conv_2d(encoder ,16,7,activation='relu')
        encoder = dropout(encoder , .25 )# replacible for noisy input 
        encoder = max_pool_2d(encoder , 2 )
        
        encoder = conv_2d(encoder,16,7,activation='relu')
        encoder = max_pool_2d(encoder , 2)
        
        encoder = conv_2d(encoder,8,7,activation='relu')
        encoder = max_pool_2d(encoder , 2)

    with tf.name_scope('Decoder'):
        decoder = conv_2d(encoder, 8, 7, activation='relu')
        decoder = upsample_2d(decoder, 2)
        
        decoder = conv_2d(decoder, 16, 7, activation='relu')
        decoder = upsample_2d(decoder, 2)
        
        decoder = conv_2d(decoder, 16, 7, activation='relu')
        decoder = upsample_2d(decoder, 2)
        decoder = conv_2d(decoder, 3, 7)

    model = regression( decoder , optimizer='adam' , 
                        loss='binary_crossentropy',
                        learning_rate=.005)
    model = tflearn.DNN(model,tensorboard_verbose=2 )
    return model 

def GAN(input_shape):
    with tf.variable_scope('Generator')