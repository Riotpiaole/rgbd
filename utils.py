import sys
from os import listdir, makedirs
from os.path import isfile, join, exists
import os
import shutil
import numpy as np

import math
import random
import re , cv2 , sys


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
    millis = int(millis)
    seconds=(millis/1000)%60
    seconds = int(seconds)
    minutes=(millis/(1000*60))%60
    minutes = int(minutes)
    hours=(millis/(1000*60*60))%24
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

def check_folders(folder_paths):
    for path in folder_paths:
        if not os.path.exists(path):os.mkdir(path)



