import os
import numpy as np
import argparse
from config import streams_config
from data_preprocessing import DataPreprocessor
from multiprocessing import Process
from utils import timeit

# global variable for loading all images
configs = streams_config()


def make_datasets():
    processes = []
    for data in configs.to_list:
        loader = DataPreprocessor(data)  # creating the preprocessor
        process = Process(target=loader.make_dataset)
        process.start()
        print("Starting process on %s" % loader.config.strFolderName)

    for process in processes:
        process.join()

    print("Thread Finished")


@timeit(log_info="loading one set of images")
def bench_mark():
    data = np.load("./data/ImgSeq_Po_02_Bag/images.npy")
    more_data = np.load("./data/ImgSeq_Po_00_first_test/images.npy")


if __name__ == "__main__":
    bench_mark()
