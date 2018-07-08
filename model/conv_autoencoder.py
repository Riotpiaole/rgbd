import tflearn
import tensorflow as tf 
import os , cv2 ,sys 


sys.path.insert(0,"..")
# dir code 
from model import data_model
from utils import * 
from keras.utils import Progbar

# machine learning module 
from keras.preprocessing import image  
from keras.layers import Input , Dense , Conv2D , MaxPool2D , UpSampling2D
from keras.models import Model 
from keras.layers.normalization import BatchNormalization
from keras import backend as K 
from keras.callbacks import TensorBoard
from keras.optimizers import RMSprop
import h5py

def Conv2D_Max2D(x,filter_size,bnorm=False):
    x = Conv2D(filter_size,(3,3),activation="relu" ,padding="same")(x)
    if bnorm:x = BatchNormalization()(x)
    x = MaxPool2D(pool_size=(2,2),padding="same")(x)
    return x 

def Conv2D_UnSample(x,filter_size):
    x = Conv2D(filter_size,(3,3),activation="relu",padding="same")(x)
    x = UpSampling2D((2,2))(x)
    return x 


class conv_autoencoder(data_model):
    def __init__(self,title="selfie_conv_autoencoder",model_name="convauto",input_shape=(128,128,3)):
        data_model.__init__(self,title,model_name,input_shape,epochs=100,batch_size=50)
        self.trained_weight_path = os.path.join(self.weight_path , "{}.h5".format(self.title)) 
     
    def build(self,bnorm=False):
        encoder , decoder = [ 32  , 64 , 128 ] , [ 128 , 64  , 128 ]
        input_img = Input(shape=self.img_shape)
        
        auto_encoder = self.auto_encoder(input_img , encoder , decoder )
        autoencoder = Model(input_img,  auto_encoder )
        
        self.model = autoencoder
        self.model.summary()
        self.model.compile(loss='mae', optimizer ="adam")
    

    def auto_encoder(self , input_img, listEncoderFSize , listDecoderFSize):
        with tf.name_scope("Encoder"):
            x = Conv2D_Max2D(input_img,listEncoderFSize[0],True)
            for f_size in listEncoderFSize[1:]:
                x = Conv2D_Max2D(x,f_size,True)
        
        with tf.name_scope("Decoder"):
            x = Conv2D_UnSample(x,listDecoderFSize[0])
            for f_size in listDecoderFSize[1:]:
                x = Conv2D_UnSample(x,f_size)
            decoder = Conv2D(3,(3,3),activation="tanh",padding="same")(x)
        return decoder        

    @timeit(log_info="Training finished ")
    def train(self,batch_size=100,n_epochs=10000):
        try:
            self.model.fit( self.data['X'], self.data['y'],
                epochs=self.nb_epochs,
                batch_size=self.batch_size,validation_data=(
                    self.validation['X'],self.validation['y']),
                callbacks=[TensorBoard(log_dir="/tmp/autoencoder")])
            print ("Training complete saving the model.")
            self.save()
        except KeyboardInterrupt:
            print("Saving Model.....")
            self.save()
                    

if __name__ =="__main__":
    model = conv_autoencoder()
    model.build()
    model.test_img()
    
    