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
