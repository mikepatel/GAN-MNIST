"""
Michael Patel
August 2019

Python 3.6.5
TensorFlow 1.12.0

Project description:
   To conduct GAN experimentation for learning purposes.
   Based on the DCGAN paper: https://arxiv.org/pdf/1511.06434.pdf

File description:
    To run preprocessing and training algorithm

Dataset: MNIST handwritten digits

"""
################################################################################
# Imports
import os
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import imageio  # generate gifs

import tensorflow as tf

from parameters import *


################################################################################
# Main
if __name__ == "__main__":
    """
    # enable eager execution
    tf.enable_eager_execution()
    """

    # print out TF version
    print("TF version: {}".format(tf.__version__))

    # create directory for checkpoints, results
    dir_name = "Results\\" + datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    # ----- ETL ----- #
    # ETL = Extraction, Transformation, Load
    (train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()

    print("Shape of training images before reshape: {}".format(train_images.shape))

    # Reshape: (28, 28) --> (28, 28, 1)
    train_images = train_images.reshape(
        train_images.shape[0], IMAGE_ROWS, IMAGE_COLS, IMAGE_CHANNELS
    ).astype("float32")

    print("Shape of training images after reshape: {}".format(train_images.shape))

    # Normalize images to [-1, 1] --- tanh activation
    train_images = (train_images - 127.5) / 127.5

