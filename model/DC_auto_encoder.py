import sys
import os
from time import time
import numpy as np
from keras.optimizers import Adam
from keras.utils import generic_utils
from keras.preprocessing import image

sys.path.append("../")

from model import data_model, inverse_normalization
from models import generator_unet_deconv
from utils import (
    timeit,
    check_folders,
    plot_generated_batch,
    get_nb_patch,
    training_wrapper
)

def l1_loss(y_true, y_pred):
    return K.sum(K.abs(y_pred - y_true), axis=-1)


class DeepConvAutoEncoder(data_model):
    def __init__(self, epoch=100000, img_shape=[256, 256, 3]):
        super().__init__(
            "deep_conv_autoencoder_bk_lr_1e-4",
            "generator",
            epochs=epoch,
            batch_size=20,
            white_bk=False)
        self.build(self.img_shape)
        self.n_batch_per_epoch = 10
        check_folders(self.weight_path)

    def build(self, img_shape):
        self.model = generator_unet_deconv(
            img_shape,
            2,
            self.batch_size,
            model_name="generator_unet_deconv",
            activation=None)
        opt_discriminator = Adam(lr=1e-4, epsilon=10e-8)
        self.model.compile(
            loss="categorical_crossentropy",
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

    @training_wrapper
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

                if batch_counter >= n_batch_per_epoch:
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
    model = DeepConvAutoEncoder()
    model.model.summary()
    model.train(retrain=False)
    model.test_img()
