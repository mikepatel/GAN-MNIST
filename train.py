"""
Michael Patel
June 2019

Python 3.6.5
TF 1.12.0

Project description:

Datasets:
    - MNIST

Notes:
    - DCGAN: https://arxiv.org/pdf/1511.06434.pdf
    - gif => imagemagick
    -

Things to examine:

"""
################################################################################
# Imports
import os
import numpy as np
from datetime import datetime

import tensorflow as tf


################################################################################
# Model hyperparameters
NUM_EPOCHS = 1
BATCH_SIZE = 64
BUFFER_SIZE = 60000  # size of training set

# based on MNIST images: 28x28 greyscale
num_rows = 28
num_cols = 28
num_channels = 1


################################################################################
# Main
if __name__ == "__main__":
    # enable eager execution
    tf.enable_eager_execution()

    # print out TF version
    print("TF version: {}".format(tf.__version__))

    # ----- ETL ----- #
    # ETL = Extraction, Transformation, Load
    (train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()

    # Reshape: (28, 28) => (28, 28, 1)
    print("Shape of training images before reshape: {}".format(train_images[0].shape))

    train_images = train_images.reshape(
        train_images.shape[0], num_rows, num_cols, num_channels
    ).astype("float32")

    print("Shape of training images after reshape: {}".format(train_images[0].shape))

    # Normalize images to [-1, 1]
    train_images = (train_images - 127.5) / 127.5

    # use tf.data.Dataset to create batches and shuffle => TF model
    train_dataset = tf.data.Dataset.from_tensor_slices(train_images)
    train_dataset = train_dataset.shuffle(buffer_size=BUFFER_SIZE)
    train_dataset = train_dataset.batch(batch_size=BATCH_SIZE)

    print("Shape of batches: {}".format(train_dataset))

    # ----- MODEL ----- #

    # ----- TRAINING ----- #

    # ----- GENERATE ----- #




