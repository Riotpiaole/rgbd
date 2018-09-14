import os
import sys

from os import listdir, makedirs
from os.path import isfile, join, exists


from tqdm import tqdm

import shutil
import numpy as np

import math
import random

from threading import Thread
from multiprocessing import Process

import re
import cv2
import sys

import matplotlib.pyplot as plt


def break_point():
    sys.exit(0)


def chunks(l, n):
    """ Yield successive n-sized chunks from l. """
    for i in range(0, len(l), n):
        yield l[i:i + n]


def tryint(s):
    try:
        return int(s)
    except BaseException:
        return s


def image_crop(mask):
    '''Take a mask and crop into the most reasonable region'''
    pass

def pil_to_cv2Img(image):
    # convert to np array 
    img_array = np.array(image.convert('RGB')) 
    cv_img = img_array[: , : , ::-1].copy()
    return cv_img


def alphanum_key(s):
    '''
        Turn a string into a list of string and number chunks.
        "z23a" -> ["z", 23, "a"]
    '''
    return [tryint(c) for c in re.split('([0-9]+)', s)]


def sort_numerically(l):
    ''' Sort the given list in the way that humans expect. For example we expect [20,10,100] to be sorted as [10, 20, 100] rather than [10, 100, 20]. '''
    l.sort(key=alphanum_key)


def number_of_digits(n):
    ''' Takes a number n as input and returns the number of digits n has. '''
    if n > 0:
        return int(math.log10(n)) + 1
    elif n == 0:
        digits = 1
    else:
        return int(math.log10(-n)) + 2  # +1 if you don't count the '-'


def time_in_seconds_to_d_h_m_s(seconds):
    ''' Return the tuple of days, hours, minutes and seconds '''
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    days, hours = divmod(hours, 24)
    # print("{0[0]} days, {0[1]} hours, {0[2]} minutes, {0[3]} seconds".format(time_in_seconds_to_d_h_m_s(seconds)))
    return days, hours, minutes, seconds


def hyperparameter_tunning(func, lrs):
    '''Fine Tuning Hyper'''
    def innerwrapper(*args, **kwargs):
        learning_rate = [1e-3, 1e-4, 1e-5]  # often opts
        result = func(*args, **kwargs)

        return result
    return innerwrapper


def listfiles_nohidden(inputPath, includeInputPath=False, ext=''):
    '''
        Return a list of files in a given directory ignoring the hidden files.
        Optional agrument ext is to ensure that the files also end with a certain extension.
    '''
    # return [ f for f in listdir(inputPath) if isfile(join(inputPath,f)) and
    # not f.startswith('.') and f.endswith(ext)]
    return [
        join(
            inputPath,
            f) if includeInputPath else f for f in listdir(inputPath) if isfile(
            join(
                inputPath,
                f)) and not f.startswith('.') and f.endswith(ext)]


def extract_subset_of_files(inputPath, outputPath, expectedNumber):
    ''' Given an input path take a random sample of size expectedNumber and copy them to outputPath'''
    files = listfiles_nohidden(inputPath)
    if len(files) <= expectedNumber:
        raise Exception("Expected number of samples (" +
                        str(expectedNumber) +
                        ") greater than number of files (" +
                        str(len(files)) +
                        ").")

    indices = random.sample(range(0, len(files)), expectedNumber)
    subset_files = [files[i] for i in indices]

    for sf in subset_files:
        shutil.copyfile(inputPath + sf, outputPath + sf)


def decimate_fileset(inputPath, outputPath, nKeep=10):
    files = listfiles_nohidden(inputPath)
    for i in range(0, len(files), nKeep):
        shutil.copyfile(inputPath + files[i], outputPath + files[i])


def split_folder(inputPath, outputPath, nFolders):
    ''' Given an input folder path, take all the files in there and place them into N different folders preserving order. '''
    files = listfiles_nohidden(inputPath)
    sort_numerically(files)

    nFiles = len(files)
    nChunkSize = int(nFiles / nFolders)

    blocks = list(chunks(files, nChunkSize))

    if not exists(outputPath):
        makedirs(outputPath)

    for i in range(0, len(blocks)):
        strPath = outputPath + "%03d/" % (i)
        if not exists(strPath):
            makedirs(strPath)
        for f in blocks[i]:
            shutil.copyfile(inputPath + f, strPath + f)

        print_text_progress_bar((i + 1) / len(blocks))
    print()


def showImage(img, name="Output_Image"):
    cv2.imshow(name, img)
    key = cv2.waitKey(0) & 0xFF
    if key == ord('q'):
        cv2.destroyAllWindows()


def showImageSet(imgs, names, destroy=True):
    for img, n in zip(imgs, names):
        cv2.imshow(n, img)

    key = cv2.waitKey(0) & 0xFF
    if key == ord('q'):
        cv2.destroyAllWindows()


def check(arr):
    '''check the given array whether or not is existing value but 0'''
    print(
        "checking ",
        "mean: " ,np.mean(arr),
        "max: ",np.max(arr),
        "min: ",np.min(arr),
        "dtype: ",arr.dtype,
        "shape: ",arr.shape)


from time import time
from functools import wraps


def re_format_hr_min_sec(time_taken):
    result = ""
    tuple_times = time_in_seconds_to_d_h_m_s(time_taken)
    for i, key in enumerate(tuple_times):
        if i == 0 and key != 0:
            result += "{}hrs ".format(key)
        elif i == 1 and key != 0:
            result += "{}mins ".format(key)
        elif i == 2 and key != 0:
            result += "{}s ".format(key)
        elif i == 3 and key != 0:
            result += "{}ms".format(key)
    return result


def timeit(log_info=None, flag=False):
    def wrapper(func):
        @wraps(func)
        def inner_wrapper(*args, **kwargs):
            start = time()
            result = func(*args, **kwargs)  # recalling the function
            end = time()
            time_taken = round(end - start, 2)
            if log_info and not flag:
                print("{} elapsed time: {} ms".format(log_info, time_taken))
            elif not log_info and not flag:
                print("elapsed time: {} ms".format(time_taken))
            elif not log_info and flag:
                print(
                    "elapsed time: {}".format(
                        re_format_hr_min_sec(time_taken)))
            else:
                print(
                    "{} elapsed time: {}".format(
                        log_info,
                        re_format_hr_min_sec(time_taken)))
            return result
        return inner_wrapper
    return wrapper


def create_folder(indx, folder_path, verbose=False,save=False):
    '''Recursive function take number of / tag and created the folders if not found ...'''
    
    strRecurPath = "/".join(folder_path[:indx])
    if indx == 1:  # reach ../sth
        if not os.path.exists(strRecurPath):
            if verbose:
                print(
                    "Folder {} not found, Create the folder {}.".format(
                        strRecurPath, strRecurPath))
            if save: os.mkdir(strRecurPath)
        else:
            if verbose:
                print("Folder {} found Skipping.".format(strRecurPath)) 
        return
    create_folder(indx-1 ,  folder_path )
    if not os.path.exists(strRecurPath):
        if verbose:
            print(
                "Folder {} not found, Create the folder {}.".format(
                    strRecurPath, strRecurPath))
        if save: os.mkdir(strRecurPath)
    else:
        if verbose:
            print("Folder {} found Skipping.".format(strRecurPath)) 
    return

    
       

def check_folders(folder_paths,verbose=False,save=True):
    '''check_folders
    Checking the Given Path existence if not create the folder
    Arguments:
        folder_paths:(str) directory of folders that will be checked and created

    ```python3
    >>> #when ../log/ doesn't exists and ../log/model doesn't exists
    >>> check_folders("../log/model_name/")
    Folder log not found, Create the folder log.
    Folder model_name not found, Create the folder model_name.
    >>> import os
    >>> os.listdir("../log/")
    ['model_name']
    >>> os.listdir("../")
    ['log']
    >>> check_folders("../log/model_name/")
    Folder log found Skipping.
    Folder model found Skipping.
    ```
    '''
    
    path_split = folder_paths.split("/")
    path_split.remove("") # avoid last char end with /
    size = len(path_split)
    create_folder(size , path_split ,verbose=verbose,save=save)


def training_wrapper(func):
    '''
    training_wrappers
            a python decroator over training parameters over a `def train(self)` with inhernted function
            self.save for key board interruption can be tracked and save the model

    ```python

    def save(self):
        ...# some saving function

    ...

    @training_wrapper
    def train(self):
        ...# training iterations

    ```
    '''
    def innerwrapper(*args, **kwargs):
        result = -1
        try:
            os.system("clear")  # for linux and mac only
            print("")
            print("====================================================================================================")
            print("Start training.")
            print("====================================================================================================\n")
            print("\n")

            result = func(*args, **kwargs)
            print("")
            print("====================================================================================================")
            print("Training completed saving the model.")
            print("====================================================================================================")
            print("\nSaving Current trained model ....a")
            args[0].save()
        except KeyboardInterrupt:
            print("Interruption detected ... Saving the model ")
            args[0].save()
        return result
    return innerwrapper


def normalization(arr, arr_max=255, arr_min=0):  # normalized between 0 and 1
    result = (arr - arr_min) / (arr_max - arr_min)
    return result.astype(np.float64)


def inverse_normalization(arr, arr_max=255, arr_min=0):
    result = (arr_max - arr_min) * (arr) + arr_min
    return result.astype(np.uint8)


def tanh_normalization(arr, arr_max=255, arr_min=0):  # normalized between 0 and 1
    result = (2 * (arr - arr_min) / (arr_max - arr_min)) - 1
    return result.astype(np.float64)


def tanh_inverse_normalization(arr, arr_max=255, arr_min=0):
    result = (1 / 2 * (arr + 1)) * (arr_max - arr_min) + arr_min
    return result.astype(np.uint8)


# normalized between standard deviation
def std_normalization(arr, arr_mean, arr_std):
    result = (arr - arr_mean) / arr_std
    return result.astype(np.uint8)


def std_inverse_normalization(arr, arr_mean, arr_std):
    result = (arr + arr_mean) * arr_std
    return result.astype(np.uint8)


def bgr_to_rgb(img):
    '''Convert image from bgr to rgb'''
    b, g, r = np.dsplit((img), 3)
    return np.dstack((r, g, b))


def rgb_to_bgr(img):
    '''Convert image from rgb to bgr'''
    r, g, b = np.dsplit((img), 3)
    return np.dstack((b, g, r))


def read_img(strName, img_shape, blur=True):
    '''
    read_img
        Reading image with opencv `cv.imread` with converting bgr to rgb in given `img_shape`
        with float64 dtype

    ```python
    >>img = read_img(../data/cat.png", (256,256,3))
    >>print(img.shape)
    (256,256,3)
    >>print(img.dtype)
    np.float64
    ```
    Arguments
        strFileDir :(str) directory of the file
        strName : (str) name of the file
        img_shape: len[( dict[int] )]==3  image shape
        blur: bool Gaussian Blur the given image # or smoothening

    Return:
        img
    '''
    img_shape = (img_shape[0], img_shape[1])

    img = cv2.imread(
        strName, -1
    ).astype(np.float64)

    resize_img = cv2.resize(img, img_shape)

    if blur:
        resize_img = cv2.blur(resize_img, (3, 3))

    resize_img = bgr_to_rgb(resize_img)
    return resize_img


def multi_threads_wrapper(iterable):
    def wrapper(func):
        def inner_wrapper(*args, **kwargs):
            threads = []
            step = 0
            for index, arg in enumerate(iterable):
                if index != 0:
                    step = step +len(arg)
                process = Thread(target=func, args=[*args, arg, step])
                process.start()
                threads.append(process)
            for thread in tqdm(
                    threads,
                    total=len(threads),
                    unit="thread",
                    leave=False):
                thread.join()
            return
        return inner_wrapper
    return wrapper


def multi_process_wrapper(iterable):
    '''multi_process_wrapper
        Wrapping given function and a constant variable for CPUs wise
        multi-processingself.
    
    Arguments:
        iterable: iterable chunkable array
        func: function that will be execute each iteration

        inner_args  *args for function arguments
                    step log_size for each step
    >>> @multi_process_wrapper([1,2,3])
    >>> def dosomething(*args):
    ...     arg , iteration = args
    ...     print(arg , iteration)
    >>> dosomething(10)
    10 1
    10 2 
    10 3

    >>> # generate 10 process for executing the soemthing_over_array
    >>> @multi_process_wrapper( list(chunk([i for i in range 1000] , 100) ) )
    >>> def something_over_array(*args)
    ...     iteration = args
    ...     print( iteartion)
    >>> something_over_array()
    10
    
    '''
    def wrapper(func):
        def inner_wrapper(*args, **kwargs):
            processes = []
            step = 0
            for index, item in enumerate(iterable):
                if index != 0:
                    step = step +len(item)

                process = Process(target=func, args=[*args, item, step])
                process.start()
                print("Starting process.....")
            for process in processes:
                process.join()
            print("Process completed.....")
            return
        return inner_wrapper
    return wrapper

def data_whiten(data,fudge=1e-18):
    U, s, Vt = np.linalg.svd(X, full_matrices=False)

    # U and Vt are the singular matrices, and s contains the singular values.
    # Since the rows of both U and Vt are orthonormal vectors, then U * Vt
    # will be white
    X_white = np.dot(U, Vt)
    return X_white

    



# =======================================================
# for Pix2Pix_keras uses
# =======================================================


def get_nb_patch(img_dim, patch_size):
    assert img_dim[0] % patch_size[0] == 0, "patch_size does not divide height"
    assert img_dim[1] % patch_size[1] == 0, "patch_size does not divide width"
    nb_patch = (img_dim[0] // patch_size[0]) * (img_dim[1] // patch_size[1])
    img_dim_disc = (patch_size[0], patch_size[1], img_dim[-1])

    return nb_patch, img_dim_disc


def get_disc_batch(X, y, generator_model, batch_counter, patch_size,
                   image_data_format, label_smoothing=False, label_flipping=0):

    # Create X_disc: alternatively only generated or real images
    if batch_counter % 2 == 0:
        # Produce an output
        X_disc = generator_model.predict(X)
        y_disc = np.zeros((X_disc.shape[0], 2), dtype=np.uint8)
        y_disc[:, 0] = 1

        if label_flipping > 0:
            p = np.random.binomial(1, label_flipping)
            if p > 0:
                y_disc[:, [0, 1]] = y_disc[:, [1, 0]]

    else:
        X_disc = y
        y_disc = np.zeros((X_disc.shape[0], 2), dtype=np.uint8)
        if label_smoothing:
            y_disc[:, 1] = np.random.uniform(
                low=0.9, high=1, size=y_disc.shape[0])
        else:
            y_disc[:, 1] = 1

        if label_flipping > 0:
            p = np.random.binomial(1, label_flipping)
            if p > 0:
                y_disc[:, [0, 1]] = y_disc[:, [1, 0]]

    # Now extract patches form X_disc
    X_disc = extract_patches(X_disc, patch_size)

    return X_disc, y_disc


def gen_batch(X1, X2, batch_size):

    while True:
        idx = np.random.choice(X1.shape[0], batch_size, replace=False)
        yield X1[idx], X2[idx]


def extract_patches(X, patch_size):
    list_X = []
    list_row_idx = [(i * patch_size[0], (i + 1) * patch_size[0])
                    for i in range(X.shape[1] // patch_size[0])]
    list_col_idx = [(i * patch_size[1], (i + 1) * patch_size[1])
                    for i in range(X.shape[2] // patch_size[1])]

    for row_idx in list_row_idx:
        for col_idx in list_col_idx:
            list_X.append(X[:, row_idx[0]:row_idx[1],
                            col_idx[0]:col_idx[1], :])

    return list_X


def vectorized_read_img(img_dir, neg_norm=False , unit_vec=False):
    '''Reading pictures and normalizes in neither range of [0 , 1] or [-1 , 1]'''
    func_normal = normalization
    if neg_norm:
        func_normal = tanh_normalization
    result = func_normal(
        read_img(
            img_dir,
            (256, 256, 3)
        ), 255.0, 0.0)
    if unit_vec: # convert (h,w,3) to (1,h,w,3)
        return np.array([result])
    return result


def plot_generated_batch(
        X,
        y,
        generator_model,
        batch_size,
        suffix,
        model_name,
        self):
    '''Plotting image that be generated generator_model with given batch and saved i'''
    # Generate images
    y_gen = generator_model.predict(X)
    if self.reverse_norm:
        y = tanh_inverse_normalization(y, self.max, self.min)
        X = tanh_inverse_normalization(X, self.max, self.min)
        y_gen = tanh_inverse_normalization(y_gen, self.max, self.min)
    else:
        y = inverse_normalization(y, self.max, self.min)
        X = inverse_normalization(X, self.max, self.min)
        y_gen = inverse_normalization(y_gen, self.max, self.min)

    ys = y[:8]
    yg = y_gen[:8]
    Xr = X[:8]

    X = np.concatenate((ys, yg, Xr), axis=0)
    list_rows = []
    for i in range(int(X.shape[0] // 4)):
        Xr = np.concatenate([X[k] for k in range(4 * i, 4 * (i + 1))], axis=1)
        list_rows.append(Xr)

    Xr = np.concatenate(list_rows, axis=0)

    if Xr.shape[-1] == 1:
        plt.imshow(Xr[:, :, 0], cmap="gray")
    else:
        plt.imshow(Xr)
    plt.axis("off")
    check_folders("../figures/%s" % (model_name) )
    plt.savefig("../figures/%s/current_batch_%s.png" % (model_name,suffix))
    plt.clf()
    plt.close()


def print_text_progress_bar(percentage, **kwargs):
    '''
        Prints a progress bar to the console. Expected to be called once per iteration to update the progress bar.
        The parameter 'percentage' should be in [0, 1] inclusive.
        *** Only works in the terminal/console and not in an IDE like IDLE.

        bar_char = the character that is used to represent one 'unit' of the progress bar that has already passed
        bar_space = the character which is used to represent one 'unit' of the progress bar that has not yet passed
        bar_length = the number of bar_char that we print
    '''
    bar_name = kwargs.get('bar_name', 'Progress')
    bar_char = kwargs.get('bar_char', '#')
    bar_space = kwargs.get('bar_space', ' ')
    bar_length = kwargs.get('bar_length', 50)
    debug_msg = kwargs.get('debug_msg', '')

    progress_sofar = bar_char * int(round(percentage * bar_length))
    progress_left = bar_space * (bar_length - len(progress_sofar))

    # TODO: figure a better to handle both python2 and python3
    # current the best solution is to just comment lines out
    #print('\r'+bar_name+'[{0}] {1}% '.format(progress_sofar + progress_left, round(percentage*100))),
    print(
        '\r' +
        bar_name +
        '[{0}] {1}% '.format(
            progress_sofar +
            progress_left,
            round(
                percentage *
                100)) +
        debug_msg,
        end='')



# ==============================
#  data_preprocessing utils
# ==============================


def compute_img_diff(label, prev_label, labels, shape=(240, 320)):
    tmp = np.zeros(shape, dtype=np.uint8)
    tmp[labels == label] = 255
    # #====================================
    # # find the standard deviation error==
    # #====================================
    # # doesnt work when similarity is too huge
    # _, std = cv2.meanStdDev(prev_label)
    # _, std2 = cv2.meanStdDev(tmp)
    # diff = np.abs(std2-std)

    # # ====================================
    # # mean_square_error # i doesnt work all of the value is simlar
    # # ====================================
    # err = np.sum((prev_label.astype(np.float64) - labels.astype(np.float64)) **2 )
    # err /=float(prev_label.shape[0] * prev_label.shape[1] )

    # =====================================
    # histograms==========================
    # =====================================
    prev_hist, next_hist = cv2.calcHist(
        [prev_label], [0], None, [256], [
            0, 256]), cv2.calcHist(
        [tmp], [0], None, [256], [
            0, 256])
    d = cv2.compareHist(prev_hist, next_hist, cv2.HISTCMP_CORREL)
    tmp[labels == 1] = 255
    return d


def click_region_call_bk(event, x, y, flags, param):
    '''
    click_region_call_bk
        call back for find the region of interests
        param accept a call back function
    '''

    if event == cv2.EVENT_LBUTTONDOWN:
        if (len(param) != 0 and param[1].frame_num ==
                0 and param[1].current_mask[param[2]] is None):
            mask_cam1 = np.zeros(param[0].shape, dtype=np.uint8)
            mask_cam1[param[0] == param[0][y, x]] = 255
            param[1].callback(mask_cam1, param[2])
    return


def unproject_pointcloud(ptcloud, camera_params, scaleFactor=1000):
    img_pts = []  # im_x, im_y, im_z, r, g, b

    for p in ptcloud:
        x, y, z, r, g, b = p

        if z > 0:  # reverse projection equations
            im_x = ((x / z) * camera_params.fx) + camera_params.cx
            im_y = ((y / z) * camera_params.fy) + camera_params.cy

            img_pts.append([im_x, im_y, z, r, g, b])

    return np.array(img_pts)


def reproject_ptcloud(index, src, dest, radius=2, suffix=""):
    img_h, img_w, _ = dest.shape
    start = time()
    debug_dest = dest[:].copy()
    for ip in src:
        x, y, z, r, g, b = ip
        in_xrange = (x > 0) and (x < img_w)
        in_yrange = (y > 0) and (y < img_h)
        if in_xrange and in_yrange:
            # found the pt that is in range of radius 
            in_radius_xrange = [
                int(x) + i for i in range(1, radius)] + [int(x) - i for i in range(1, radius)]
            in_radius_yrange = [
                int(y) + i for i in range(1, radius)] + [int(x) - i for i in range(1, radius)]
            # put it into opencv bgr ordering
            dest[int(y), int(x)] = np.array([b, g, r])
            debug_dest[int(y), int(x)] = np.array([b, g, r])
            if radius != 0:
                for (x, y) in zip(in_radius_xrange, in_radius_yrange):
                    in_xrange, in_yrange = (
                        x > 0) and (x < img_w),\
                        (y > 0) and (y < img_h)
                    if in_xrange and in_yrange:
                        # put it into opencv bgr ordering
                        dest[y, x] = np.array([b, g, r])

    end = time()
    time_taken = round(end - start, 2)
    if suffix != "":
        print(
            "successifully reprojected ptcloud " +
            str(index) + " " + suffix +
            " time taken :" +
            str(time_taken))

    return debug_dest


def convert_depth_2_rgb(img_depth, max_depth=3000):
    # colors will repeat if the depth can be measured beyond max_depth
    # (default = 10 meters)
    img_depth_rgb = img_depth * (255 / max_depth)
    img_depth_rgb = np.uint8(img_depth_rgb)
    img_depth_rgb = cv2.applyColorMap(img_depth_rgb, cv2.COLORMAP_JET)
    return img_depth_rgb


def transform_pointcloud_vectorized(
        ptcloud,
        rotMat,
        transMat,
        scaleFactor=1000):
    xyz, rgb = np.hsplit(ptcloud, 2)
    # apply rotation
    rotatedXYZ = np.matmul(xyz, rotMat[0:3, 0:3])

    # apply translation
    rotatedXYZ[..., 0] -= (transMat[0] * scaleFactor)
    rotatedXYZ[..., 1] += (transMat[1] * scaleFactor)
    rotatedXYZ[..., 2] += (transMat[2] * scaleFactor)

    transformedPtcloud = np.hstack([rotatedXYZ, rgb])

    return transformedPtcloud


def scatter_point_filtering(img, close=(4, 4),
                            open_=(10, 10), grad=50,
                            dsize=3, bilater=(30, 50),
                            ksize=(4, 4), grad_filtering=False):
    img = cv2.morphologyEx(img, cv2. MORPH_OPEN, open_)
    img = cv2.morphologyEx(img, cv2. MORPH_CLOSE, close)
    if grad_filtering:
        img = cv2.bilateralFilter(
            img.astype(
                np.float32),
            dsize,
            bilater[0],
            bilater[1])
        img = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, ksize)
        img[img > grad] = 0  # remove some scatter points
    return img

def black_bg(img_reproj, img_clr, mask_front, mask_bk):
    img_reproj[mask_bk == 0] = 0
    img_clr[mask_front == 0] = 0
    return img_reproj, img_clr


def white_bg(img_reproj, img_clr, mask_front, mask_bk):
    img_reproj[mask_bk == 0] = 255
    img_clr[mask_front == 0] = 255
    return img_reproj, img_clr
