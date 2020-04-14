"""
Michael Patel
April 2020

Project description:
    Build a GAN using the MNIST data set

File description:
    For model preprocessing and training
"""
################################################################################
# Imports
import os
import numpy as np
from datetime import datetime
import glob
from PIL import Image
import matplotlib.pyplot as plt

import tensorflow as tf

from parameters import *
from model import build_discriminator, build_generator


################################################################################
# Main
if __name__ == "__main__":
    # print TF version
    print(f'TF version: {tf.__version__}')

    # create output directory for results
    output_dir = "results\\" + datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # ----- ETL ----- #
    # ETL = Extraction, Transformation, Load
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

    size_train_dataset = len(train_images)

    # rescale images from [0, 255] to [-1, 1]
    train_images = (train_images.astype(np.float32) - 127.5) / 127.5
    test_images = (test_images.astype(np.float32) - 127.5) / 127.5

    # augment data set
    train_dataset = tf.data.Dataset.from_tensor_slices(train_images)
    train_dataset = train_dataset.shuffle(buffer_size=size_train_dataset)
    train_dataset = train_dataset.batch(batch_size=BATCH_SIZE)

    # ----- MODEL ----- #
