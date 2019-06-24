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
from parameters import *
from model import build_generator, build_discriminator


################################################################################
# generator loss
def g_loss(fake_output):
    cross_entropy = tf.keras.losses.binary_crossentropy(True)
    loss = cross_entropy(tf.ones_like(fake_output), fake_output)

    return loss


# discriminator loss
def d_loss(real_output, fake_output):
    cross_entropy = tf.keras.losses.binary_crossentropy(True)
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss

    return total_loss


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
    g = build_generator()
    d = build_discriminator()

    # Loss

    # Optimizer
    g_optimizer = tf.keras.optimizers.Adam(lr=1e-4)
    d_optimizer = tf.keras.optimizers.Adam(lr=1e-4)

    # ----- TRAINING ----- #
    # Saving model, checkpoints

    # ----- GENERATE ----- #






