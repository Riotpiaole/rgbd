from keras.layers import Dense, Input
from keras.layers import Conv2D, Flatten, Lambda
from keras.layers import Reshape, Conv2DTranspose
from keras.models import Model
from keras.datasets import mnist
from keras.losses import mse 
from keras import backend as K

import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

