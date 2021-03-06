import sys
import os
from time import time
import keras
from keras.optimizers import Adam
from keras.utils import generic_utils
from keras.preprocessing import image
import keras.backend as K

from model import data_model, inverse_normalization
from models import generator_unet_deconv

from utils import (
    timeit,
    check_folders,
    plot_generated_batch,
    training_wrapper
)


def l1_loss(y_true, y_pred):
    return K.sum(K.abs(y_pred - y_true), axis=-1)


class DeepConvAutoEncoder(data_model):
    def __init__(
            self,
            epoch=100000,
            img_shape=[256,256,3],
            learning_rate=1.e-5,
            white_bk=True,
            batch_size=20,
            name="deep_conv_autoencoder",
            loss="l1"):

        bk = "bk"
        if white_bk:
            bk = "wh"

        super().__init__(
            name +
            "_%s_lr_%s_imgdim%s_loss_%s" %
                (bk,
                learning_rate,
                img_shape[0],
                loss),
            "generator",
            epochs=epoch,
            batch_size=batch_size,
            white_bk=white_bk)

        self.learning_rate = learning_rate
        self.loss = self.get_loss(loss)
        self.build(self.img_shape)
        self.n_batch_per_epoch = 10

    @staticmethod
    def get_loss(loss):
        if loss == "l1":
            return l1_loss
        elif loss == "squared_hinge":
            return keras.losses.squared_hinge
        else:
            return keras.losses.categorical_crossentropy

    def build(self, img_shape):
        self.model = generator_unet_deconv(
            img_shape,
            2,
            self.batch_size,
            model_name="generator_unet_deconv",
            activation=None)
        opt_discriminator = Adam(
            lr=self.learning_rate,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=10e-8)
        self.model.compile(
            loss=self.loss,
            optimizer=opt_discriminator)

    def log_checkpoint(self, epoch, batch, loss):
        log_path = os.path.join(self.weight_path, "checkpoint")

        prev_epochs, prev_batch_size = 0, 0
        if os.path.isfile(log_path):
            with open(log_path, "w+") as f:
                line = f.readline()
                if "Epoch" in line:
                    line = f.readline().split(" ")
                    prev_epochs = int(line[4])

        with open(log_path, "w+") as f:
            f.write("Model_Name {} ".format(self.title))
            f.write("Epoch {} in batch {}".format(
                epoch + prev_epochs,
                batch))
            f.write("\n")
            f.write("Losses: {}".format(loss))

    # @training_wrapper
    @timeit(log_info="Training deconv_autoEncoder")
    def train(self, retrain=False):
        n_batch_per_epoch = self.n_batch_per_epoch
        total_epoch = n_batch_per_epoch * self.batch_size
        gen_loss = 100
        if retrain:
            try:
                print("Looking for previous model ...")
                self.load()
                print("Found Model start retraining...")
            except FileNotFoundError:
                print("No previous model found retraining a new one")
        for e in range(self.nb_epochs):
            batch_counter = 1
            start = time()
            progbar = generic_utils.Progbar(total_epoch)

            for X, y in self.gen_batch(self.batch_size):
                gen_loss = self.model.train_on_batch(X, y)
                batch_counter += 1
                progbar.add(self.batch_size, values=[
                    ("G loss ", gen_loss)])

                if batch_counter % (n_batch_per_epoch / 2) == 0:
                    plot_generated_batch(
                        X,
                        y,
                        self.model,
                        self.batch_size,
                        "training",
                        self.title,
                        self)
                    # get next validation batches
                    X_test, y_test = next(self.gen_batch(
                        self.batch_size, validation=True))
                    plot_generated_batch(
                        X_test,
                        y_test,
                        self.model,
                        self.batch_size,
                        "validation",
                        self.title,
                        self)

                if batch_counter > n_batch_per_epoch:
                    break

            print("")
            t_time = time() - start
            print('Epoch %s/%s, Time: %s ms' %
                  (e + 1, self.nb_epochs, round(t_time, 2)), end="\r")
            
            if e % 5 == 0:
                self.save()
                self.log_checkpoint(e, batch_counter, [
                                    ("G loss ", gen_loss)])


if __name__ == "__main__":
    model = DeepConvAutoEncoder(epoch=100000)
    model.model.summary()
    # model.train(retrain=True)
    # model.test_img()
