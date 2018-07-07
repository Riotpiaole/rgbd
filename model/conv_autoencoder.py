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
    def __init__(self,name="selfie_conv_autoencoder",save_name="convauto",input_shape=(128,128,3)):
        data_model.__init__(self,name,save_name,input_shape)
        self.input_shape = input_shape
        self.nb_epochs = 100
        self.weight_path = os.path.join(self.model_path , "{}.h5".format(self.name)) 

     
    def build(self,bnorm=False):
        encoder , decoder = [ 32  , 64 , 128 ] , [ 128 , 64  , 128 ]
        input_img = Input(shape=self.input_shape)
        
        auto_encoder = self.auto_encoder(input_img , encoder , decoder )
        autoencoder = Model(input_img,  auto_encoder )
        
        self.model = autoencoder
        self.model.summary()
        self.model.compile(loss='mean_squared_error', optimizer = RMSprop())
    
    def save(self):
        if not os.path.exists(self.model_path):
            path = self.model_path.split("/")
            
            os.mkdir(path[0]+"/"+path[1]+"/"+path[2])
            os.mkdir(self.model_path)
            h5py.File(self.weight_path)
        self.model.save_weights(self.weight_path)

    
    def load(self):
        self.model.load_weights(self.weight_path)


    def auto_encoder(self , input_img, listEncoderFSize , listDecoderFSize):
        with tf.name_scope("Encoder"):
            x = Conv2D_Max2D(input_img,listEncoderFSize[0],True)
            for f_size in listEncoderFSize[1:]:
                x = Conv2D_Max2D(x,f_size,True)
        
        with tf.name_scope("Decoder"):
            x = Conv2D_UnSample(x,listDecoderFSize[0])
            for f_size in listDecoderFSize[1:]:
                x = Conv2D_UnSample(x,f_size)
            decoder = Conv2D(3,(3,3),activation="sigmoid",padding="same")(x)
        return decoder


    def debug_picker(self,bnorm=False):
        encoder , decoder = [ 32 , 64 , 128 ] , [ 128 , 64 , 128]

        input_img = Input(shape=self.input_shape)
        
        auto_encoder = self.auto_encoder(input_img , encoder , decoder )
        autoencoder = Model(input_img,  auto_encoder )
        
        self.model = autoencoder
        self.model.compile(loss='mean_squared_error', optimizer = RMSprop())
        

    @timeit(log_info="Training finished ",flag=True)
    def train(self,batch_size=100,n_epochs=100):
        try:
            self.model.fit( self.data['X'], self.data['y'],
                epochs=self.nb_epochs,
                batch_size=batch_size,validation_data=(
                    self.validation['X'],self.validation['y']),
                callbacks=[TensorBoard(log_dir="/tmp/autoencoder")])
        except KeyboardInterrupt:
            print("Saving Model.....")
            self.save()
                    

if __name__ =="__main__":
    model = conv_autoencoder()
    model.build()
    model.train()
    
    