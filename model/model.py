import tflearn
import tensorflow as tf 
import os , cv2 , sys  ,math
import numpy as np
import random as rnd

sys.path.insert(0,"..")

from utils import *
from config import strFolderName
from keras_utils_layers import extract_patches ,normalization , inverse_normalization
import matplotlib.pyplot as plt
from keras.preprocessing import image
from functools import wraps
import h5py

def bgr_to_rgb(img):
    b , g , r =  np.dsplit((img),3)
    return np.dstack((r,g,b))

def rgb_to_bgr(img):
    r , g , b = np.dsplit((img),3)
    return np.dstack((b,g,r))

def read_img(filedir,name,img_shape):
    img_shape = ( img_shape[0] ,img_shape[1])
    img = cv2.imread(os.path.join(filedir,name),-1).astype(np.float64)
    resize_img = cv2.resize(img, img_shape)
    resize_img = bgr_to_rgb(resize_img)
    return normalization(resize_img)

class data_model(object):
    def __init__(self,title,model_name,img_shape =( 256 ,256 ,3 ),epochs=100,batch_size = 2 ):
        '''data_model
        Loading all of the image from `../data` to self.data
            self.data['X']: front image 
            self.data['y']: back image 
    
        Required  
            Implements -> self.trained_weight_path 
                       -> self.tensorboard_path

        Arguments:
            

        '''
        
        self.title = title
        self.model_name = model_name
        self.batch_size = batch_size 
        self.nb_epochs = epochs
        self.img_shape = img_shape 
        self.weight_path = "../log/"+self.title+"/"+self.model_name # dir for weights of dir
        self.trianed_weight_path = os.path.join(self.weight_path , "%s.h5" %  self.model_name)
        self.data_dir = "../data/"+strFolderName  # dir for all of the data
        self.target_dir  , self.train_dir = self.data_dir+"/target" , self.data_dir+"/train" 
        self.data = {}
        self.load_data()

    def load_data(self):
        train, target= os.listdir(self.train_dir), os.listdir(self.target_dir)
        # load all of the images
        self.data['X'] = [ read_img(self.train_dir,img , self.img_shape) for img in train ]
        self.data['y'] = [ read_img(self.target_dir,img, self.img_shape ) for img in target ]
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
        generator.train_on_batch(X,y)
        y_disc = np.zeros((X.shape[0],2), dtype=np.uint8)
        if batch_counter % 2 == 0:
            # X_disc = generator.predict(X)
            X_disc = generator.predict(X)
            y_disc[:,0 ]=1
        
        else:
            X_disc = y
            if label_smoothing: y_disc[:, 1] = np.random.uniform(low=0.9, high=1, size=y_disc.shape[0])
            else: y_disc[:, 1] = 1

        if label_flipping > 0:
            p = np.random.binomial(1, label_flipping)
            if p > 0: y_disc[:, [0, 1]] = y_disc[:, [1, 0]]
        
        X_disc = extract_patches(X_disc , patch_size)
        return X_disc , y_disc 

    def gen_batch(self, batch_size , validation=False):
        '''get validation and train batches''' 
        while True:
            if not validation:
                idx = np.random.choice(len(self.data['X']),batch_size ,replace=False)
                yield  self.data['X'][idx] , self.data['y'][idx]
            else: 
                idx = np.random.choice(len(self.validation['X']),batch_size ,replace=False)
                yield  self.validation['X'][idx] , self.validation['y'][idx]

    def build(self):
        '''Build the model for the network '''
        pass

    def load(self):
        '''Load models weight from log/model_name'''
        if os.path.exists(self.trained_weight_path):
            self.model.load_weights(self.trained_weight_path)
        else:
            raise FileNotFoundError("No Previous Model Found") 
        print("Loading model  from {}".format(self.trained_weight_path))

    def save(self):
        if not os.path.exists(self.trained_weight_path):
            h5py.File(self.trained_weight_path)
        self.model.save_weights(self.trained_weight_path)        

    def test_img(self):
        idx = rnd.choice([ i for i in range(0 , len(self.data['X']) )]) # pick a random index
        
        X , y = self.get_data( idx ) # normalized images
        self.load()
                
        X_pred = self.model.predict(np.array([X]))
        X = image.array_to_img(inverse_normalization(X))
        y = image.array_to_img(inverse_normalization(y))
        X_pred = image.array_to_img(inverse_normalization(X_pred[0]))
        
        suffix = "End_test"

        result = np.hstack((X ,y , X_pred))
        
        check_folders("../figures/%s" % (self.title) )
        plt.savefig("../figures/%s/current_batch_%s.png" % (self.title,suffix))
        plt.imshow(result)
        plt.axis("off")
        plt.show()


if __name__ == "__main__":
    test_model = Model("Files")
    