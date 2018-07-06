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

def Conv2D_Max2D(filter_size,bnorm):
    x = Conv2D(filter_size,(3,3),activation="relu" ,padding="same")(input_img)
    if bnorm:x = BatchNormalization()(x)
    x = MaxPool2D((2,2),padding="same")(x)

def Conv2D_UnSample(filter_size,bnorm):
    x = Conv2D(filter_size,(3,3),activation="relu",padding="same")(x)
    x = UpSampling2D((2,2))(x)


class conv_autoencoder(data_model):
    def __init__(self,name="selfie_conv_autoencoder",save_name="convauto",input_shape=(256,256,3)):
        data_model.__init__(self,name,save_name)
        self.input_shape = input_shape
        self.weight_path = os.path.join(self.model_path , "{}.h5".format(self.name)) 

     
    def build(self,bnorm=False):
        input_img = Input(shape=self.input_shape)
        with tf.name_scope("Encoder"):
            x = Conv2D(128,(3,3),activation="relu" ,padding="same")(input_img)
            if bnorm:x = BatchNormalization()(x)
            x = MaxPool2D((2,2),padding="same")(x)
            
            x = Conv2D(64,(3,3),activation="relu", padding="same")(x)
            if bnorm:x = BatchNormalization()(x)
            x = MaxPool2D((2,2),padding="same")(x)

            x = Conv2D(32,(3,3),activation="relu",padding="same")(x)
            if bnorm:x = BatchNormalization()(x)
            encoded = MaxPool2D((2,2),padding="same")(x)

        x = Conv2D(16,(3,3),activation="relu",padding="same")(encoded)
        with tf.name_scope("Decoder"):
            x = Conv2D(32,(3,3),activation="relu",padding="same")(x)
            x = UpSampling2D((2,2))(x)
            
            x = Conv2D(64,(3,3),activation="relu",padding="same")(x)
            x = UpSampling2D((2,2))(x)

            x = Conv2D(128,(3,3),activation="relu",padding="same")(x)
            x = UpSampling2D((2,2))(x)

        decoded = Conv2D(3,(3,3),activation="tanh",padding="same")(x)

        model = Model(input_img , decoded)
        model.compile( optimizer ="adam" , loss="mae")
        self.model = model 

    
    def save(self):
        if not os.path.exists(self.model_path):
            path = self.model_path.split("/")
            
            os.mkdir(path[0]+"/"+path[1]+"/"+path[2])
            os.mkdir(self.model_path)
            h5py.File(self.weight_path)
        self.model.save_weights(self.weight_path)

    
    def load(self):
        self.model.load_weights(self.weight_path)

    def debug_picker(self,bnorm=False):
        input_img = Input(shape=self.input_shape)
        
        with tf.name_scope("Encoder"):
            x = Conv2D(64,(3,3),activation="relu", padding="same")(input_img)
            if bnorm:x = BatchNormalization()(x)
            x = MaxPool2D((2,2),padding="same")(x)

            x = Conv2D(32,(3,3),activation="relu",padding="same")(x)
            if bnorm:x = BatchNormalization()(x)
            encoded = MaxPool2D((2,2),padding="same")(x)

        x = Conv2D(16,(3,3),activation="relu",padding="same")(encoded)

        with tf.name_scope("Decoder"):
            x = Conv2D(32,(3,3),activation="relu",padding="same")(x)
            x = UpSampling2D((2,2))(x)
            
            x = Conv2D(64,(3,3),activation="relu",padding="same")(x)
            x = UpSampling2D((2,2))(x)

        decoded = Conv2D(3,(3,3),activation="tanh",padding="same")(x)

        model = Model(input_img , decoded)
        model.compile( optimizer ="adam" , loss="mae")

    @timeit(log_info="Training finished ",flag=True)
    def train(self,batch_size=100,n_epochs=100):
        try:
            self.model.fit( self.data['X'], self.data['y'],
                epochs=self.nb_epochs,
                batch_size=batch_size,validation_data=(
                    self.validation['X'],self.validation['y']),
                callbacks=[TensorBoar(log_dir="/tmp/autoencoder")])
        except KeyboardInterrupt:
            print("Saving Model.....")
            self.save()
                    

if __name__ =="__main__":
    model = conv_autoencoder()
    model.debug_picker()
    model.train()
    
    