import os
import numpy as np
import argparse
from config import streams_config
from data_preprocessing import DataPreprocessor
from multiprocessing import Process
from utils import ( 
    timeit,
    multi_process_wrapper,
    multi_threads_wrapper,
    chunks
)

# global variable for loading all images
configs = streams_config()

@multi_process_wrapper(configs.to_list)
def make_datasets():
    '''create datasets when the images.npy was not present'''
    processes = []
    for data in configs.to_list:
        loader = DataPreprocessor(data)  # creating the preprocessor
        loader.load_data()
        process = Process(target=loader.make_dataset)
        process.start()
        print("Starting process on %s" % loader.config.strFolderName)

    for process in processes:
        process.join()

    print("Thread Finished")

@multi_process_wrapper(configs.to_list)
def unpack_imgs(*args):
    config, _ = args
    loader = DataPreprocessor(config)
    loader.unzip_npy_to_imgs()

if __name__ == "__main__":
    unpack_imgs()
