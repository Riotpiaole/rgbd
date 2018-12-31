import os
import cv2
import sys
import math
from tqdm import tqdm
import numpy as np
import random as rnd

sys.path.insert(0, "..")

import h5py
import matplotlib.pyplot as plt
from utils import (
    pil_to_cv2Img,
    rgb_to_bgr,
    showImageSet,
    showImage,
    timeit,
    read_img,
    check_folders,
    listfiles_nohidden,
    check,
    vectorized_read_img,
    alphanum_key,
    extract_patches,
    normalization,
    inverse_normalization,
    tanh_normalization,
    tanh_inverse_normalization
)

# import matplotlib.pyplot as plt
from keras.preprocessing import image
from config import streams_config


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
            validation_size=.2,
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
        self.configs = streams_config()
        self.nb_epochs = epochs
        self.img_shape = img_shape

        self.read_list_of_imgs = lambda x: np.array(list(map(vectorized_read_img, x)))
        self.weight_path = "../log/" + self.title + "/"
        check_folders(self.weight_path)

        self.trained_weight_path = os.path.join(
            self.weight_path, "%s.h5" %
            self.model_name)

        self.white_bk = white_bk

        # image with max and min with rgb as 0 to 255
        self.max = 255.0
        self.min = 0.0

        # total number of images through out all of the directories
        self.total_imgs = self.configs.total_imgs

        self.data = {
            "X": np.array([]),
            "y": np.array([])
        }

        self.load_datas(validation_size=validation_size)

    def load_datas(self, validation_size=.2):
        suffix = "black"
        if self.white_bk:
            suffix = ""

        print("--------------------------------------------------------------------------------")
        for config in  self.configs.to_list:
            self.load_data(config.strFolderName + suffix)
        print("--------------------------------------------------------------------------------")

        # 20 % of validation sample
        num_of_sample = math.floor(validation_size * self.total_imgs)
        self.validation = {
            'X': np.array(self.data['X'][:num_of_sample]),
            'y': np.array(self.data['y'][:num_of_sample])
        }

        self.data['X'] = np.array(self.data['X'][num_of_sample:])
        self.data['y'] = np.array(self.data['y'][num_of_sample:])

        # size references
        self.num_train, self.num_val = len(
            self.data['X']), len(
            self.validation['X'])

        print(
            "Loaded all image sequence total of : %d , %d training_imgs  and %d validation imgs" %
            (self.total_imgs, self.num_train, self.num_val))

    def load_data(self, data_set):
        data_dir = os.path.join("../data/", data_set)
        train_dir, target_dir = os.path.join(data_dir + "/train"), \
            os.path.join(data_dir + "/target")

        # load all of the images
        train_files , target_files =sorted(
                listfiles_nohidden(
                    train_dir,
                    includeInputPath=True,
                    ext='png'), key=alphanum_key) ,\
            sorted(
                listfiles_nohidden(
                    target_dir,
                    includeInputPath=True,
                    ext='png'), key=alphanum_key)


        self.data["X"] = np.append(self.data["X"], train_files)
        self.data["y"] = np.append(self.data["y"], target_files)
        # log the progress
        print("|Loading image sequence {}| : {}%/100%".format(
            data_set,
            round(
                self.data["X"].shape[0]
                / self.total_imgs * 100, 2)))

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
                    self.num_train, batch_size, replace=False)
                yield self.read_list_of_imgs(self.data['X'][idx]),\
                    self.read_list_of_imgs(self.data['y'][idx])
            else:
                idx = np.random.choice(
                    self.num_val, batch_size, replace=False)
                yield self.read_list_of_imgs(self.validation['X'][idx]),\
                    self.read_list_of_imgs(self.validation['y'][idx])



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
        '''Test the progress of the model during training '''
        # pick a random index
        idx = rnd.choice([i for i in range(0, len(self.data['X']))])

        X, y = self.get_data(idx)  # normalized images
        self.load()

        X_pred = self.model.predict(np.array([X]))
        if self.reverse_norm:
            X = image.array_to_img(
                tanh_inverse_normalization(
                    X, self.max, self.min))
            y = image.array_to_img(
                tanh_inverse_normalization(
                    y, self.max, self.min))
            X_pred = image.array_to_img(
                tanh_inverse_normalization(
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
        plt.show()
        plt.savefig(
            "../figures/%s/current_batch_%s.png" %
            (self.title, suffix))

    def predict(self, X , y , func=None, meshed=True):
        '''Predict the result and create a meshed image and return '''
        rever_norm = inverse_normalization
        if self.reverse_norm:
            rever_norm = tanh_inverse_normalization

        if func: result = func(self , X )

        result = self.predicts(X,rever_norm)

        X , y = rever_norm(X).astype(np.uint8)[0] , rever_norm(y).astype(np.uint8)[0]
        X , y = rgb_to_bgr(X) , rgb_to_bgr(y)

        if meshed:
            meshed = np.hstack((X,y, result ))
            return [meshed] , [ "Input::Label::Prediction"]

        return [X , y , result] , [ 'Input' , 'SupposeOutput' , 'predictedResult ']

    def predicts(self, X , rever_norm):
        result = self.model.predict(X)
        result = image.array_to_img(rever_norm(result)[0])
        result = pil_to_cv2Img(result)
        return result

    def demo(self,dataset_name , save=False, show=False , imgs = False):
        ''' Demoing the result of prediction in Input,GroundTruth,Prediction'''
        if not ( save or show  or imgs): raise ValueError("You are not doing anything")
        config = self.configs.process(dataset_name)
        suffix = "black"
        if self.white_bk: # check the background color images
            suffix = ""

        if save:
            video_name = os.path.join(self.weight_path,dataset_name+".avi")
            print("Saving the video to ",video_name)
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            video = cv2.VideoWriter(video_name , fourcc , 60 ,(768 ,256), True)

        data_dir = os.path.join("../data/", config.strFolderName + suffix)
        train_dir, target_dir = os.path.join(data_dir + "/train"), \
            os.path.join(data_dir + "/target")

        # load all of the images
        train_files , target_files =sorted(
                listfiles_nohidden(
                    train_dir,
                    includeInputPath=True,
                    ext='png'), key=alphanum_key) ,\
            sorted(
                listfiles_nohidden(
                    target_dir,
                    includeInputPath=True,
                    ext='png'), key=alphanum_key)
        for (idx ,( train , target)) in enumerate(
            tqdm(
                zip(train_files , target_files),
                        total=len(train_files),
                        unit="image",
                        leave=False)):
            X , y = vectorized_read_img(train,unit_vec=True) , vectorized_read_img(target, unit_vec=True)
            # the images must be in (1 , 256, 256 ,3)
            img , names = self.predict(X,y, meshed=not imgs)
            # output is (256 , 768, 3)
            if save: video.write(img[0])
            if show: showImageSet(img , names)
            if imgs: cv2.imwrite(data_dir+"/prediction/%s.png" % idx ,  img[len(img) -1 ] )

        if save: video.release()


if __name__ == "__main__":
    test_model = data_model("Files", "Name")
