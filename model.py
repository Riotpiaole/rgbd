import tflearn
import tensorflow as tf 
import os , cv2
import numpy as np
import random as rnd

from utils import *
from config import strFolderName

from tflearn.data_utils import image_preloader
from tflearn.data_preprocessing import ImagePreprocessing 
from tflearn.data_augmentation import ImageAugmentation
from tflearn.layers.core import input_data , dropout , fully_connected 
from tflearn.layers.conv import conv_2d , max_pool_2d ,upsample_2d
from keras.preprocessing import image

def bgr_to_rgb(img):
    b , g , r =  np.dsplit((img))
    return np.dstack((r,g,b))

def rgb_to_bgr(img):
    r , g , b = np.dsplit((img))
    return np.dstack((b,g,r))

def read_img(filedir,name):
    return cv2.imread(os.path.join(filedir,name),-1).astype(np.float64)

class Model(object):
    def __init__(self,name,save_name):
        self.name = name
        self.save_name = save_name
        self.model_path = "./model/"+self.name+"/"+self.save_name
        self.model_dir = "./data/"+strFolderName
        self.target_dir  , self.train_dir = self.model_dir+"/target" , self.model_dir+"/train" 
        self.data = {}
        self.load_data()

    def load_data(self):
        train, target= os.listdir(self.train_dir), os.listdir(self.target_dir)
        # load all of the images
        self.data['train'] = [ read_img(self.train_dir,img ) for img in train ]
        self.data['target'] = [ read_img(self.train_dir,img ) for img in target ]

    def get_data(self,index):
        return self.data['train'][index] , self.data['target'][index]

    def build(self,input_shape):
        pass

    def test_img(self, name="output.jpg"):
        idx = rnd.choice([ i for i in range(0 , len(self.data['train']) )]) # pick a random index
        X , y = self.get_data( idx )
        self.model.load("./model/"+self.name)
        X_pred = self.model.predict(X)
        
        # save the array
        X_pred = image.array_to_img(X_pred )
        X_pred.save("./"+name)
        
        X_pred = cv2.imread("./"+name)
        showImageSet([X ,y,X_pred], ['input' , 'output', name])

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

def GAN(input_shape):
    pass 


if __name__ == "__main__":
    test_model = Model("Files")
    