import tflearn
import tensorflow as tf 
import os , cv2 ,sys 

sys.path.insert(0,"..")
# dir code 
from model import Model
from utils import * 

# machine learning module 
from keras.preprocessing import image  
from tflearn.data_utils import image_preloader
from tflearn.data_preprocessing import ImagePreprocessing 
from tflearn.data_augmentation import ImageAugmentation
from tflearn.layers.core import input_data , dropout , fully_connected 
from tflearn.layers.conv import conv_2d , max_pool_2d ,upsample_2d

class conv_autoencoder(Model):
    def __init__(self,name="selfie_conv_autoencoder",save_name="convauto",input_shape=(240,320,3)):
        Model.__init__(self,name,save_name)
        self.build()
    
    def build(self,input_shape=(240,320,3)):
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
        self.model = tflearn.DNN(model,tensorboard_verbose=0,tensorboard_dir="./log")
    
    @timeit(log_info="Training finished ",flag=True)
    def train(self,batch_size=10,n_epochs=2000):
        
        if os.path.exists(self.model_path+".ckpt.meta"):
            print("Found previous trained model ...")
            self.model.load(self.model_path+".ckpt")
        try:
            self.model.fit(self.data['train'],self.data['target'],
                        n_epoch=n_epochs,batch_size=batch_size,
                        show_metric=True,validation_set=.1, 
                        run_id =self.name)
        except KeyboardInterrupt:
            print("Emergency stop..... saving the model... ")
            self.model.save(self.model_path+".ckpt")
            print("Finish saving ....")
                    

if __name__ =="__main__":
    model = conv_autoencoder()
    
    