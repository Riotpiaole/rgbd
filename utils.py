import os
import sys
from os import listdir, makedirs
from os.path import isfile, join, exists
import shutil
import numpy as np

import math
import random
import re , cv2 , sys
# import matplotlib.pyplot as plt 

def break_point():
    sys.exit(0)

def chunks(l, n):
    """ Yield successive n-sized chunks from l. """
    for i in range(0, len(l), n):
        yield l[i:i+n]

def tryint(s):
    try:
        return int(s)
    except:
        return s

def alphanum_key(s):
    '''
        Turn a string into a list of string and number chunks.
        "z23a" -> ["z", 23, "a"]
    '''
    return [ tryint(c) for c in re.split('([0-9]+)', s) ]

def sort_numerically(l):
    ''' Sort the given list in the way that humans expect. For example we expect [20,10,100] to be sorted as [10, 20, 100] rather than [10, 100, 20]. '''
    l.sort(key=alphanum_key)

def number_of_digits(n):
    ''' Takes a number n as input and returns the number of digits n has. '''
    if n > 0: return int(math.log10(n))+1
    elif n == 0: digits = 1
    else: return int(math.log10(-n))+2 # +1 if you don't count the '-' 


def time_in_seconds_to_d_h_m_s(seconds):
    ''' Return the tuple of days, hours, minutes and seconds '''
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    days, hours = divmod(hours, 24)
    return days, hours, minutes, seconds # print("{0[0]} days, {0[1]} hours, {0[2]} minutes, {0[3]} seconds".format(time_in_seconds_to_d_h_m_s(seconds)))


def print_text_progress_bar(percentage, **kwargs):
    '''
        Prints a progress bar to the console. Expected to be called once per iteration to update the progress bar. 
        The parameter 'percentage' should be in [0, 1] inclusive.
        *** Only works in the terminal/console and not in an IDE like IDLE.

        bar_char = the character that is used to represent one 'unit' of the progress bar that has already passed
        bar_space = the character which is used to represent one 'unit' of the progress bar that has not yet passed
        bar_length = the number of bar_char that we print
    '''
    bar_name=kwargs.get('bar_name', 'Progress')
    bar_char=kwargs.get('bar_char', '#')
    bar_space=kwargs.get('bar_space', ' ')
    bar_length=kwargs.get('bar_length', 50)
    debug_msg=kwargs.get('debug_msg', '')

    progress_sofar = bar_char * int(round(percentage * bar_length))
    progress_left = bar_space * (bar_length - len(progress_sofar))

    # TODO: figure a better to handle both python2 and python3
    # current the best solution is to just comment lines out
    #print('\r'+bar_name+'[{0}] {1}% '.format(progress_sofar + progress_left, round(percentage*100))),
    print('\r'+bar_name+'[{0}] {1}% '.format(progress_sofar + progress_left, round(percentage*100)) + debug_msg, end='')

def listfiles_nohidden(inputPath, includeInputPath=False, ext=''):
    '''
        Return a list of files in a given directory ignoring the hidden files.
        Optional agrument ext is to ensure that the files also end with a certain extension.
    '''
    #return [ f for f in listdir(inputPath) if isfile(join(inputPath,f)) and not f.startswith('.') and f.endswith(ext)]
    return [ join(inputPath,f) if includeInputPath else f for f in listdir(inputPath) if isfile(join(inputPath,f)) and not f.startswith('.') and f.endswith(ext)]

def extract_subset_of_files(inputPath, outputPath, expectedNumber):
    ''' Given an input path take a random sample of size expectedNumber and copy them to outputPath'''
    files = listfiles_nohidden(inputPath)
    if len(files) <= expectedNumber: raise Exception("Expected number of samples ("+str(expectedNumber)+") greater than number of files ("+str(len(files))+").")

    indices = random.sample(range(0, len(files)), expectedNumber)
    subset_files = [files[i] for i in indices]

    for sf in subset_files: shutil.copyfile(inputPath + sf, outputPath + sf)

def decimate_fileset(inputPath, outputPath, nKeep=10):
    files = listfiles_nohidden(inputPath)
    for i in range(0, len(files), nKeep):
        shutil.copyfile(inputPath + files[i], outputPath + files[i])


def split_folder(inputPath, outputPath, nFolders):
    ''' Given an input folder path, take all the files in there and place them into N different folders preserving order. '''
    files = listfiles_nohidden(inputPath)
    sort_numerically(files)

    nFiles = len(files)
    nChunkSize = int(nFiles/nFolders)

    blocks = list(chunks(files, nChunkSize))

    if not exists(outputPath): makedirs(outputPath)

    for i in range(0, len(blocks)):
        strPath = outputPath  + "%03d/" %(i)
        if not exists(strPath): makedirs(strPath)
        for f in blocks[i]: 
            shutil.copyfile(inputPath + f, strPath + f)

        print_text_progress_bar((i+1)/len(blocks))
    print()
             
def showImage(img,name="Output_Image"):
    cv2.imshow(name,img)
    key = cv2.waitKey( 0 )& 0xFF 
    if key == ord( 'q' ):
        cv2.destroyAllWindows() 

def showImageSet(imgs,names,destroy=True ):
    for img , n in zip(imgs,names):
        cv2.imshow(n , img)

    key = cv2.waitKey( 0 )& 0xFF 
    if key == ord('q'):
        cv2.destroyAllWindows()

def check(arr):
    print("checking ",np.mean(arr) , np.max(arr) , np.min(arr))

from  time import time 
from functools import wraps

def ms_to_hr_mins(t_taken):
    t_taken = int(t_taken)
    seconds=(t_taken/1000)%60
    minutes=(t_taken/(1000*60))%60
    hours=(t_taken/(1000*60*60))%24
    return "{}hr{}min and {}s".format(hours,minutes,seconds)

def timeit(log_info=None,flag=False):
    def wrapper( func ):
        @wraps( func )
        def inner_wrapper( *args , **kwargs ):
            start = time()
            result = func( *args , **kwargs) # recalling the function 
            end = time() 
            time_taken =  round (end - start ,2)
            if log_info and not flag:print ( "{} elapsed time: {} ms".format(log_info,time_taken ))
            elif not log_info and not flag:print ( "elapsed time: {} ms".format(time_taken ))
            elif not log_info and flag:print("elapsed time: {}".format(ms_to_hr_mins(time_taken)))
            else: print("{} elapsed time: {}".format(log_info,ms_to_hr_mins(time_taken)))
            return result 
        return inner_wrapper
    return wrapper

def create_folder(indx , folder_path, verbose=False):
    '''Recursive function take number of / tag and created the folders if not found ...'''
    strRecurPath = "/".join(folder_path[:indx])
    if indx == 2: # reach ../sth
        if not os.path.exists(strRecurPath):
            if verbose: print("Folder {} not found, Create the folder {}.".format(strRecurPath,strRecurPath))
            os.mkdir(strRecurPath)
            pass
        else: 
            if verbose: print("Folder {} found Skipping.".format(strRecurPath))
            pass
        return 
    else:
        if not os.path.exists(strRecurPath):
            create_folder(indx-1,folder_path)
            if verbose: print("Folder {} not found, Creating the folder {}.".format(strRecurPath,strRecurPath))
            os.mkdir(strRecurPath)
            pass
        else:
            if verbose: print("Folder {} found Skipping check folder.".format(strRecurPath))
            pass
        return
    

def check_folders(folder_paths):
    '''Checking the Given Path existence if not create the folder'''
    create_folder(len(folder_paths.split("/")),folder_paths.split("/"))

def training_wrapper(func):
    def innerwrapper(*args , **kwargs):
        result = -1
        try:
            os.system("clear") # for linux and mac only
            print("")
            print("====================================================================================================")
            print("Start training.")
            print("====================================================================================================\n")
            print("\n")
            
            result = func(*args , **kwargs)
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
            y_disc[:, 1] = np.random.uniform(low=0.9, high=1, size=y_disc.shape[0])
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


def extract_patches(X , patch_size):
    list_X = []
    list_row_idx = [(i * patch_size[0], (i + 1) * patch_size[0]) for i in range(X.shape[1] // patch_size[0])]
    list_col_idx = [(i * patch_size[1], (i + 1) * patch_size[1]) for i in range(X.shape[2] // patch_size[1])]

    for row_idx in list_row_idx:
        for col_idx in list_col_idx:
            list_X.append(X[:, row_idx[0]:row_idx[1], col_idx[0]:col_idx[1], :])
    
    return list_X

def plot_generated_batch(X, y, generator_model, batch_size, image_data_format, suffix,model_name,self):

    # Generate images
    y_gen = generator_model.predict(X)
    if self.reverse_norm:
        y = neg_inverse_normalization(y,self.max , self.min )
        X = neg_inverse_normalization(X,self.max , self.min )
        y_gen = neg_inverse_normalization(y_gen,self.max , self.min )
    else:        
        y = inverse_normalization(y,self.max , self.min )
        X = inverse_normalization(X,self.max , self.min )
        y_gen = inverse_normalization(y_gen,self.max , self.min )
        
    ys = y[:8]
    yg = y_gen[:8]
    Xr = X[:8]

    if image_data_format == "channels_last":
        X = np.concatenate((ys, yg, Xr), axis=0)        
        list_rows = []
        for i in range(int(X.shape[0] // 4)):
            Xr = np.concatenate([X[k] for k in range(4 * i, 4 * (i + 1))], axis=1)
            list_rows.append(Xr)

        Xr = np.concatenate(list_rows, axis=0)

    # if Xr.shape[-1] == 1:
    #     plt.imshow(Xr[:, :, 0], cmap="gray")
    # else:
    #     plt.imshow(Xr)     
    # plt.axis("off")
    # check_folders("../figures/%s" % (model_name) )
    # plt.savefig("../figures/%s/current_batch_%s.png" % (model_name,suffix))
    # plt.clf()
    # plt.close()

def normalization(arr , arr_max , arr_min): # normalized between 0 and 1 
    result = (arr - arr_min)/(arr_max - arr_min)
    return result.astype(np.float64)

def inverse_normalization(arr , arr_max , arr_min):
    result = (arr_max - arr_min) * ( arr ) + arr_min
    return result.astype(np.uint8)


def neg_normalization(arr , arr_max , arr_min): # normalized between 0 and 1 
    result = (2*(arr - arr_min)/(arr_max - arr_min)) -1
    return result.astype(np.float64)

def neg_inverse_normalization(arr , arr_max , arr_min):
    result = (1/2*(arr + 1 ))*(arr_max - arr_min) + arr_min 
    return result.astype(np.uint8)

def std_normalization(arr , arr_mean , arr_std):
    result = (arr-arr_mean)/arr_std
    return result.astype(np.uint8)

def inverse_std_normalization(arr , arr_mean , arr_std):
    result = (arr+arr_mean)*arr_std
    return result.astype(np.uint8)


def bgr_to_rgb(img):
    b , g , r =  np.dsplit((img),3)
    return np.dstack((r,g,b))

def rgb_to_bgr(img):
    r , g , b = np.dsplit((img),3)
    return np.dstack((b,g,r))



def read_img(filedir,name,img_shape, blur=True):
    img_shape = ( img_shape[0] ,img_shape[1])
    img = cv2.imread(os.path.join(filedir,name),-1).astype(np.float64)
    resize_img = cv2.resize(img, img_shape)
    if blur :resize_img = cv2.blur(resize_img , (3,3))
    resize_img = bgr_to_rgb(resize_img)
    return resize_img
