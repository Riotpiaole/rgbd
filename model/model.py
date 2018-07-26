import tensorflow as tf
import os
import cv2
import sys
import math
import numpy as np
import random as rnd

sys.path.insert(0, "..")

import h5py
from utils import (
    timeit,
    read_img,
    bgr_to_rgb,
    rgb_to_bgr,
    check_folders,
    alphanum_key,
    extract_patches,
    normalization,
    inverse_normalization,
    tanh_normalization,
    tanh_inverse_normalization
)

from functools import wraps
import matplotlib.pyplot as plt
from keras.preprocessing import image
from config import strFolderName, strFolderNameBlack


class data_model(object):
    def __init__(
            self,
            title,
            model_name,
            img_shape=(
                256,
                256,
                3),
            epochs=100,
            batch_size=2,
            white_bk=False,
            reverse_norm=False):
        '''data_model
        Loading all of the image from `../data` to self.data
            self.data['X']: front image
            self.data['y']: back image
        '''

        self.title = title
        self.model_name = model_name
        self.batch_size = batch_size
        self.reverse_norm = reverse_norm
        self.nb_epochs = epochs
        self.img_shape = img_shape
        self.weight_path = "../log/" + self.title + "/" + \
            self.model_name  # dir for weights of dir
        self.trained_weight_path = os.path.join(
            self.weight_path, "%s.h5" %
            self.model_name)
        if white_bk:
            self.data_dir = "../data/" + strFolderName  # dir for all of the data
        else:
            self.data_dir = "../data/" + strFolderNameBlack
        self.target_dir, self.train_dir = self.data_dir + \
            "/target", self.data_dir + "/train"
        self.data = {}
        self.load_data()

    @timeit(log_info="Loading Data from dir")
    def load_data(self):
        train, target = os.listdir(self.train_dir), os.listdir(self.target_dir)
        # load all of the images
        train = sorted(train, key=alphanum_key)
        target = sorted(target, key=alphanum_key)

        self.data['X'] = [
            read_img(
                self.train_dir,
                img,
                self.img_shape) for img in train]
        self.data['y'] = [
            read_img(
                self.target_dir,
                img,
                self.img_shape) for img in target]

        entire_samples = np.vstack((self.data['X'], self.data['y']))
        self.max = np.max(entire_samples)
        self.min = np.min(entire_samples)

        if self.reverse_norm:
            self.data['X'] = tanh_normalization(
                self.data['X'], self.max, self.min)
            self.data['y'] = tanh_normalization(
                self.data['y'], self.max, self.min)
        else:
            self.data['X'] = normalization(self.data['X'], self.max, self.min)
            self.data['y'] = normalization(self.data['y'], self.max, self.min)

        # 20 % of validation sample
        num_of_sample = math.floor(.2 * len(self.data['X']))
        self.validation = {'X': np.array(self.data['X'][:num_of_sample]), 'y': np.array(
            self.data['y'][:num_of_sample])}
        self.data['X'] = np.array(self.data['X'][num_of_sample:])
        self.data['y'] = np.array(self.data['y'][num_of_sample:])
        self.num_train, self.num_val = len(
            self.data['X']), len(
            self.validation['X'])

    def get_data(self, index):
        return self.data['X'][index], self.data['y'][index]

    def get_disc_batch(
            self,
            X,
            y,
            generator,
            batch_counter,
            patch_size,
            label_smoothing=False,
            label_flipping=0):
        '''
        Get the discriminator batch data
            Generator predict the sample
        '''
        generator.train_on_batch(X, y)
        y_disc = np.zeros((X.shape[0], 2), dtype=np.uint8)
        if batch_counter % 2 == 0:
            X_disc = generator.predict(X)
            y_disc[:, 0] = 1

        else:
            X_disc = y
            if label_smoothing:
                y_disc[:, 1] = np.random.uniform(
                    low=0.9, high=1, size=y_disc.shape[0])
            else:
                y_disc[:, 1] = 1

        if label_flipping > 0:
            p = np.random.binomial(1, label_flipping)
            if p > 0:
                y_disc[:, [0, 1]] = y_disc[:, [1, 0]]

        X_disc = extract_patches(X_disc, patch_size)
        return X_disc, y_disc

    def gen_batch(self, batch_size, validation=False):
        '''get validation and train batches'''
        while True:
            if not validation:
                idx = np.random.choice(
                    len(self.data['X']), batch_size, replace=False)
                yield self.data['X'][idx], self.data['y'][idx]
            else:
                idx = np.random.choice(
                    len(self.validation['X']), batch_size, replace=False)
                yield self.validation['X'][idx], self.validation['y'][idx]

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
        # pick a random index
        idx = rnd.choice([i for i in range(0, len(self.data['X']))])

        X, y = self.get_data(idx)  # normalized images
        self.load()

        X_pred = self.model.predict(np.array([X]))
        if self.reverse_norm:
            X = image.array_to_img(
                neg_inverse_normalization(
                    X, self.max, self.min))
            y = image.array_to_img(
                neg_inverse_normalization(
                    y, self.max, self.min))
            X_pred = image.array_to_img(
                neg_inverse_normalization(
                    X_pred[0], self.max, self.min))
        else:
            X = image.array_to_img(
                inverse_normalization(
                    X, self.max, self.min))
            y = image.array_to_img(
                inverse_normalization(
                    y, self.max, self.min))
            X_pred = image.array_to_img(
                inverse_normalization(
                    X_pred[0], self.max, self.min))

        suffix = "End_test"

        result = np.hstack((X, y, X_pred))

        check_folders("../figures/%s" % (self.title))
        plt.imshow(result)
        plt.axis("off")
<<<<<<< HEAD
        plt.show()
        plt.savefig(
            "../figures/%s/current_batch_%s.png" %
            (self.title, suffix))
=======
        plt.savefig("../figures/%s/current_batch_%s.png" % (self.title,suffix))
        plt.show()
        plt.clf()
        plt.close()
       

>>>>>>> master


if __name__ == "__main__":
    test_model = data_model("Files", "Name")
