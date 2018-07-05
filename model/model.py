import tflearn
import tensorflow as tf 
import os , cv2 , sys  ,math
import numpy as np
import random as rnd

sys.path.insert(0,"..")

from utils import *
from config import strFolderName
import keras_utils_layers
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
    img = cv2.imread(os.path.join(filedir,name),-1).astype(np.float64)
    resize_img = cv2.resize(img,(256,256) )
    return keras_utils_layers.normalization(resize_img)

def extract_patches(X , patch_size):
    list_X = []
    list_row_idx = [(i * patch_size[0], (i + 1) * patch_size[0]) for i in range(X.shape[1] // patch_size[0])]
    list_col_idx = [(i * patch_size[1], (i + 1) * patch_size[1]) for i in range(X.shape[2] // patch_size[1])]

    for row_idx in list_row_idx:
        for col_idx in list_col_idx:
            list_X.append(X[:, row_idx[0]:row_idx[1], col_idx[0]:col_idx[1], :])
    
    return list_X


class data_model(object):
    def __init__(self,name,save_name):
        self.name = name
        self.save_name = save_name
        self.model_path = "../log/"+self.name+"/"+self.save_name
        self.model_dir = "../data/"+strFolderName
        self.target_dir  , self.train_dir = self.model_dir+"/target" , self.model_dir+"/train" 
        self.data = {}
        self.load_data()

    def load_data(self):
        train, target= os.listdir(self.train_dir), os.listdir(self.target_dir)
        # load all of the images
        self.data['X'] = [ read_img(self.train_dir,img ) for img in train ]
        self.data['y'] = [ read_img(self.target_dir,img ) for img in target ]
        num_of_sample = math.floor(.2 * len(self.data['X'])) # 20 % of validation sample 
        self.validation= { 'X': np.array(self.data['X'][:num_of_sample]) , 'y':np.array(self.data['y'][:num_of_sample])}
        self.data['X'] = np.array(self.data['X'][num_of_sample:])
        self.data['y'] = np.array(self.data['y'][num_of_sample:])
        
        self.num_train , self.num_val = len(self.data['X']) , len(self.validation['X'])

    def get_data(self,index):
        return self.data['X'][index] , self.data['y'][index]
    


    def get_disc_batch(self ,X,y, generator , batch_counter , patch_size, label_smoothing=False, label_flipping=0):
        '''
        Get the discriminator batch data 
            Generator predict the sample
        '''
        
        y_disc = np.zeros((X.shape[0],2), dtype=np.uint8)
        if batch_counter % 2 == 0:
            X_disc = generator.predict(X)
            y_disc[:,0 ]=1
        
        else:
            X_disc = X
            if label_smoothing: y_disc[:, 1] = np.random.uniform(low=0.9, high=1, size=y_disc.shape[0])
            else: y_disc[:, 1] = 1

        if label_flipping > 0:
            p = np.random.binomial(1, label_flipping)
            if p > 0: y_disc[:, [0, 1]] = y_disc[:, [1, 0]]
        
        X_disc = extract_patches(X_disc , patch_size)
        return X_disc , y_disc 

   

    def gen_batch(self, batch_size , validation=False): 
        while True:
            idx = np.random.choice(len(self.data['X']),batch_size ,replace=False)
            if not validation:
                yield  self.data['X'][idx] , self.data['y'][idx]
            else: 
                yield  self.validation['X'][idx] , self.validation['y'][idx]

    def build(self,input_shape):
        pass

    def test_img(self, name="output.jpg"):
        idx = rnd.choice([ i for i in range(0 , len(self.data['train']) )]) # pick a random index
        X , y = self.get_data( idx )
        self.model.load("../log/model/"+self.name+"/"+self.save_name+".ckpt")
        X_pred = self.model.predict(X)
        
        # save the array
        X_pred = image.array_to_img(X_pred )
        X_pred.save("./"+name)
        
        X_pred = cv2.imread("./"+name)
        showImageSet([X ,y,X_pred], ['input' , 'output', name])


if __name__ == "__main__":
    test_model = Model("Files")
    