import tflearn
import sys

sys.path.insert(0,"..")

from utils import *
from model import data_model
import tensorflow as tf 


def generator(x, reuse=False):
    with tf.variable_scope("Generator",reuse=reuse):
        x = tflearn.fully_connected(x, n_units=7 * 7 * 128)
        x = tflearn.batch_normalization(x)
        x = tf.nn.tanh(x)
        x = tf.reshape(x, shape=[-1, 7, 7, 128])
        x = tflearn.upsample_2d(x, 2)
        x = tflearn.conv_2d(x, 64, 5, activation='tanh')
        x = tflearn.upsample_2d(x, 2)
        x = tflearn.conv_2d(x, 3, 5,activation='sigmoid')
        return x

def discriminator(x,reuse=False):
    with tf.variable_scope("Discriminator",reuse=reuse):
        # num_filter , filter_size ,  strides 
        # x = tflearn.conv_2d(x, 64, 5, activation='tanh')
        x = tflearn.conv_2d(x, 64, 5 ,activation='tanh')
        x = tflearn.avg_pool_2d(x, 2)# ksize
        x = tflearn.conv_2d(x, 128, 5 ,activation='tanh')
        x = tflearn.avg_pool_2d(x, 2)
        x = tflearn.fully_connected(x, 1024, activation='tanh')
        x = tflearn.fully_connected(x, 2)
        x = tf.nn.softmax(x)
        return x

class GAN(data_model):
    def __init__(self,name="generator adverstial net",save_name="GAN",input_shape=(320,320,3),z_dim = 200): 
        data_model.__init__(self,name,save_name)
        self.total_samples = len(self.data['X'])
        self.z_dim = z_dim # norise value 
        self.img_dim = input_shape
        self.build(input_shape)
    

    def build(self, input_shape ):
        height , width , channel = self.img_dim
        
        with tf.name_scope("Inputs"):
            gen_input = tflearn.input_data(shape = [ None, self.z_dim, channel],name="input_gen_noise")
            input_disc_noise = tflearn.input_data(shape=[None ,self.z_dim, channel ], name="input_disc_noise")
            input_disc_real = tflearn.input_data(shape=[None,  height , width , channel ], name='input_disc_real')            

        with tf.name_scope("GAN"):
            # Building Discriminator 
            disc_fake = discriminator( generator(input_disc_noise)  )
            disc_real = discriminator( input_disc_real , reuse=True)
            
            disc_net = tf.concat([disc_fake, disc_real], axis=0)

            # Build Stacked Generator/Discriminator
            gen_net = generator(gen_input, reuse=True)
            stacked_gan_net = discriminator(gen_net, reuse=True)
        # Build Training Ops for both Generator and Discriminator.
        # Each network optimization should only update its own variable, thus we need
        # to retrieve each network variables (with get_layer_variables_by_scope).
        
        disc_vars = tflearn.get_layer_variables_by_scope('Discriminator')
        # We need 2 target placeholders, for both the real and fake image target.
        disc_target = tflearn.multi_target_data(['target_disc_fake', 'target_disc_real'],
                                                shape=[None, 2])
        
        disc_model = tflearn.regression(disc_net, optimizer='adam',
                                        placeholder=disc_target,
                                        loss='binary_crossentropy',
                                        trainable_vars=disc_vars,
                                        batch_size=32, name='target_disc',
                                        op_name='DISC')

        gen_vars = tflearn.get_layer_variables_by_scope('Generator')
        gan_model = tflearn.regression(stacked_gan_net, optimizer='adam',
                                    loss='categorical_crossentropy',
                                    trainable_vars=gen_vars,
                                    batch_size=32, name='target_gen',
                                    op_name='GEN')

        # Define GAN model, that output the generated images.
        self.model = tflearn.DNN(gan_model)
            
    
    def train(self):
        disc_noise = np.random.uniform(-1., 1., size=[self.total_samples, self.z_dim])
        # Prepare target data to feed to the discriminator (0: fake image, 1: real image)
        y_disc_fake = np.zeros(shape=[self.total_samples])
        y_disc_real = np.ones(shape=[self.total_samples])
        
        y_disc_fake = tflearn.data_utils.to_categorical(y_disc_fake, 2)
        y_disc_real = tflearn.data_utils.to_categorical(y_disc_real, 2)

        # Prepare input data to feed to the stacked generator/discriminator
        gen_noise = np.random.uniform(-1., 1., size=[self.total_samples, self.z_dim])
        # Prepare target data to feed to the discriminator
        # Generator tries to fool the discriminator, thus target is 1 (e.g. real images)
        y_gen = np.ones(shape=[self.total_samples])
        y_gen = tflearn.data_utils.to_categorical(y_gen, 2)

        # Start training, feed both noise and real images.
        try:
            self.model.fit(X_inputs={'input_gen_noise': gen_noise,
                            'input_disc_noise': disc_noise,
                            'input_disc_real': self.data['X']},
                    Y_targets={'target_gen': y_gen,
                            'target_disc_fake': y_disc_fake,
                            'target_disc_real': y_disc_real},
                    n_epoch=300)
        except KeyboardInterrupt:
            self.model.save(self.save_name+".ckpt")
if __name__ == "__main__":
    model = GAN()
    model.train()
    
    
