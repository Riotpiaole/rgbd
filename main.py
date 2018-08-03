import os 
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Training the model')
    parser.add_argument('-batch',default=4, type=int , help="batch_size for trainning")
    parser.add_argument('-dset', default='all', type=int, help="which collection of images to be process")
    parser.add_argument('-nb_epoch', default=5500, type=int, help="numbers of epochs")
    parser.add_argument('-use_mbd', action="store_true", help="Whether to use minibatch discrimination")
    parser.add_argument('-lr' ,default=1.e-3 , type=float ,help="Learning rate for the model")
    parser.add_argument('-model' ,  choices=['auto_encoder' , 'KDCGAN'] , default='auto_encoder', help="choice of model only KDCGAN and auto_encoder")
    # parser.add_argument('-')