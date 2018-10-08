# DCGAN
# generate photo-realistic handwritten digits

# using tf.keras
# using eager execution
# https://github.com/tensorflow/tensorflow/blob/r1.11/tensorflow/contrib/eager/python/examples/generative_examples/dcgan.ipynb
# DCGAN paper: https://arxiv.org/pdf/1511.06434.pdf

# dataset: MNIST

# Notes:
#   -

################################################################################
# IMPORTs
from __future__ import absolute_import, division, print_function

import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Conv2DTranspose, \
    BatchNormalization, Dropout, Flatten, LeakyReLU, Reshape
from tensorflow.keras.activations import relu, softmax, sigmoid
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard

import os
import time
import numpy as np
import glob
import matplotlib.pyplot as plt
import PIL
import imageio
from IPython import display

################################################################################
# load dataset
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# reshape and normalize
train_images = train_images.reshape(train_images.reshape[0], 28, 28, 1)
train_images = train_images.astype(np.float32)
train_images = (train_images - 127.5) / 127.5  # [-1, 1]

BUFFER_SIZE = 60000
BATCH_SIZE = 256

# use tf.data to create batches of dataset
train_dataset = tf.data.Dataset.from_tensor_slices(train_images)
train_dataset = train_dataset.shuffle(BUFFER_SIZE)
train_dataset = train_dataset.batch(BATCH_SIZE)


################################################################################
# generator
def build_generator():
    m = Sequential()

    m.add(Dense(
        units=7*7*64,
        input_dim=100,
        activation=LeakyReLU(0.1)
    ))

    m.add(BatchNormalization())

    m.add(Reshape(
        target_shape=(-1, 7, 7, 64)
    ))

    # deconvolution
    m.add(Conv2DTranspose(
        filters=64,
        kernel_size=[5, 5],
        padding="same",
        activation=LeakyReLU(0.1)
    ))

    m.add(BatchNormalization())

    m.add(Conv2DTranspose(
        filters=32,
        kernel_size=[5, 5],
        padding="same",
        activation=LeakyReLU(0.1)
    ))

    m.add(BatchNormalization())

    m.add(Conv2DTranspose(
        filters=1,
        kernel_size=[5, 5],
        padding="same",
        activation=LeakyReLU(0.1)
    ))

    m.summary()
    return m


################################################################################
# discriminator
def build_discriminator():
    m = Sequential()

    m.add(Conv2D(
        filters=32,
        kernel_size=[5, 5],
        input_shape=(28, 28, 1),
        padding="same",
        activation=LeakyReLU(0.1)
    ))

    m.add(BatchNormalization())

    m.add(MaxPool2D(
        pool_size=[2, 2],
        strides=2
    ))

    m.add(Conv2D(
        filters=64,
        kernel_size=[5, 5],
        padding="same",
        activation=LeakyReLU(0.1)
    ))

    m.add(BatchNormalization())

    m.add(MaxPool2D(
        pool_size=[2, 2],
        strides=2
    ))

    m.add(Conv2D(
        filters=128,
        kernel_size=[5, 5],
        padding="same",
        activation=LeakyReLU(0.1)
    ))

    m.add(Flatten())

    m.add(Dense(
        units=128,
        activation=LeakyReLU(0.1)
    ))

    m.add(Dense(
        units=1,
        activation=sigmoid
    ))

    m.summary()
    return m


################################################################################
