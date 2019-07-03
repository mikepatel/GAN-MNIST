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
    - Not using eager execution

Things to examine:

"""
################################################################################
# Imports
import os
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf

from tf_1_12.parameters import *
from tf_1_12.model import build_generator, build_discriminator


################################################################################
# print image to screen
def print_image(image):
    plt.figure(figsize=(2, 2))  # 2x2 inches
    plt.imshow(image, cmap="gray")
    plt.show()


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

    #
    tf.reset_default_graph()
    tf.keras.backend.set_learning_phase(1)  # 1=train, 0=test

    # ----- ETL ----- #
    # ETL = Extraction, Transformation, Load
    (train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()

    '''
    # show a training image before Preprocessing Transformation
    i = np.random.randint(low=0, high=len(train_images))
    random_image = train_images[i]
    print_image(random_image)
    quit()
    '''

    # Reshape: (28, 28) => (28, 28, 1)
    print("Shape of training images before reshape: {}".format(train_images[0].shape))

    train_images = train_images.reshape(
        train_images.shape[0], IMAGE_ROWS, IMAGE_COLS, IMAGE_CHANNELS
    ).astype("float32")

    print("Shape of training images after reshape: {}".format(train_images[0].shape))

    # Normalize images to [-1, 1] - tanh activation
    train_images = (train_images - 127.5) / 127.5

    # use tf.data.Dataset to create batches and shuffle => TF model
    train_dataset = tf.data.Dataset.from_tensor_slices(train_images)
    train_dataset = train_dataset.shuffle(buffer_size=BUFFER_SIZE)
    train_dataset = train_dataset.batch(batch_size=BATCH_SIZE)

    print("Shape of batches: {}".format(train_dataset))

    # ----- MODEL ----- #
    # Placeholders
    real_image_pl = tf.placeholder(dtype=tf.float32,
                                   shape=[None, IMAGE_ROWS, IMAGE_COLS, IMAGE_CHANNELS],
                                   name="real_image_placeholder")

    noise_pl = tf.placeholder(dtype=tf.float32,
                              shape=[None, Z_DIM],
                              name="noise_placeholder")

    # placeholder inputs to the models
    g_out = build_generator(noise_pl, reuse=False)

    d_real_out = build_discriminator(real_image_pl, reuse=False)

    d_fake_out = build_discriminator(g_out, reuse=True)

    # ----- TRAINING ----- #

    # ----- GENERATE ----- #
