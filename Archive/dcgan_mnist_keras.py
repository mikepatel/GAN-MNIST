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
from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Conv2DTranspose, \
    BatchNormalization, Dropout, Flatten, LeakyReLU, Reshape, Input, UpSampling2D
from tensorflow.keras.activations import relu, softmax, sigmoid
from tensorflow.keras.losses import sparse_categorical_crossentropy
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
train_images = train_images.reshape(train_images.shape[0], 28, 28, 1)
train_images = train_images.astype(np.float32)
train_images = (train_images - 127.5) / 127.5  # [-1, 1]

BUFFER_SIZE = 60000
BATCH_SIZE = 256
NUM_EPOCHS = 200

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
        target_shape=(7, 7, 64)
    ))

    m.add(UpSampling2D())

    # deconvolution
    m.add(Conv2DTranspose(
        filters=64,
        kernel_size=[5, 5],
        padding="same",
        activation=LeakyReLU(0.1)
    ))

    m.add(BatchNormalization())

    m.add(UpSampling2D())

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

    m.compile(
        loss=sparse_categorical_crossentropy,
        optimizer=Adam()
    )

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

    m.compile(
        loss=sparse_categorical_crossentropy,
        optimizer=Adam()
    )

    m.summary()
    return m


################################################################################
# build GAN
gen = build_generator()
disc = build_discriminator()

disc.trainable = False

gan_input = Input(
    shape=(100,)
)

g = gen(gan_input)
gan_output = disc(g)

gan = Model(
    inputs=gan_input,
    outputs=gan_output
)

gan.compile(
    loss=sparse_categorical_crossentropy,
    optimizer=Adam()
)

gan.summary()


################################################################################
# training
def train():
    # adversarial ground truths
    valid = np.ones((BATCH_SIZE, 1))
    fake = np.zeros((BATCH_SIZE, 1))

    for e in range(NUM_EPOCHS):
        # input for discriminator
        indices = np.random.randint(0, train_images.shape[0], size=BATCH_SIZE)
        real_images = train_images[indices]

        # input for generator
        noise_input = np.random.normal(0, 1, size=[BATCH_SIZE, 100])
        fake_images = gen.predict(noise_input)

        # train discriminator
        disc.trainable = True
        disc_loss_real = disc.train_on_batch(real_images, valid)
        disc_loss_fake = disc.train_on_batch(fake_images, fake)
        disc_loss = np.add(disc_loss_real, disc_loss_fake) * 0.5

        # train generator
        noise_input = np.random.normal(0, 1, size=[BATCH_SIZE, 100])
        disc.trainable = False
        gen_loss = gan.train_on_batch(noise_input, valid)

        # save checkpoints
        if e % 50 == 0:
            print("Epoch: {}".format(e))

        # tensorboard


################################################################################
# plot
def plot_output():
    t = np.random.rand(10, 100)
    preds = gen.predict(t)

    plt.figure(figsize=(10, 10))

    for i in range(preds.shape[0]):
        plt.subplot(10, 10, i+1)
        plt.imshow(preds[i, :, :, 0], cmap="gray")
        plt.axis("off")
    plt.show()


################################################################################
train()
plot_output()
