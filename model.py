import tflearn
import tensorflow as tf 
import os , cv2

from utils import *
from config import strFolderName
from tflearn.data_utils import image_preloader
from tflearn.data_preprocessing import ImagePreprocessing 
from tflearn.data_augmentation import ImageAugmentation
from tflearn.layers.core import input_data , dropout , fully_connected 
from tflearn.layers.conv import conv_2d , max_pool_2d ,upsample_2d
from keras.preprocessing import image

def load_data():
    train_dir , target_dir = "./data/{}/target".format(strFolderName) , "./data/{}/train".format(strFolderName)
    train , target = os.listdir(train_dir), os.listdir(target_dir)
    train , target = [ cv2.imread(os.path.join(train_dir ,img),-1).astype(np.float64) for img in train ] ,\
                 [ cv2.imread(os.path.join(target_dir,img),-1).astype(np.float64) for img in target]
    return train , target
    

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
    img_prep.add_featurewise_stdnorm
    with tf.name_scope("Encoder"):
        encoder = input_data(shape=(None, input_shape[0] , input_shape[1] , input_shape[2]) , data_preprocessing=img_prep)
        
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

    model = tflearn.regression( decoder , optimizer='adadelta' , 
                        loss='binary_crossentropy',
                        learning_rate=.005)
    model = tflearn.DNN(model,tensorboard_verbose=0,tensorboard_dir="./log")
    return model 

def GAN(input_shape):
    pass 

# if __name__ == "__main__":
#     model = conv_autoencoder((240,320,3))
#     X, y = load_data()
#     model.fit(X,y,n_epoch=10, shuffle=True , show_metric=True,
#                 batch_size=20, validation_set=0.1,
#                 run_id='selfie_conv_autoencoder')
#     model.save("./model/selfie_conv_autoencoder/convauto.ckpt")


#TODO add test image code 
# if __name__ == "__main__":
#     X, y = load_data()
#     rand_img_id = 0 
#     X_test , y_test = X[ rand_img_id ] , y[ rand_img_id ]
#     model = conv_autoencoder((240,320,3))
#     model.load("./model/selfie_conv_autoencoder/convauto.ckpt")
#     X_pred = model.predict(np.array([X_test]))[0]
#     X_pred = image.array_to_img(X_pred )
#     X_pred.save("./output.jpg")
    
#     toShow = [ X_test , X_pred , y_test ]
#     toShowName = [ "source" , "pred" , "target"]
#     check(X_pred)
    # showImageSet(toShow , toShowName)

if __name__ == "__main__":
    model = conv_autoencoder((240,320,3))
    X, y = load_data()
    model.load("./model/selfie_conv_autoencoder/convauto.ckpt")
    model.fit(X,y,n_epoch=2000, shuffle=True , show_metric=True,
                batch_size=10, validation_set=0.1,
                run_id='selfie_conv_autoencoder')
    model.save("./model/selfie_conv_autoencoder/convauto.ckpt")
