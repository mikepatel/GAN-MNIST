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
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf

from parameters import *
from model import build_generator, build_discriminator


################################################################################
# print image to screen
def print_image(image):
    plt.figure(figsize=(2, 2))  # 2x2 inches
    plt.imshow(image, cmap="gray")
    plt.show()


################################################################################
# Callbacks
def build_callbacks(chkpt_dir):
    history_file = os.path.join(chkpt_dir, "checkpoint_{epoch}")

    # save callback
    sc = tf.keras.callbacks.ModelCheckpoint(
        filepath=history_file,
        save_weights_only=True,
        period=1,
        verbose=1
    )

    # TensorBoard callback
    tb = tf.keras.callbacks.TensorBoard(log_dir=chkpt_dir)

    return sc, tb


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
        train_images.shape[0], num_rows, num_cols, num_channels
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
    g = build_generator()
    d = build_discriminator()

    # build gan model
    d.trainable = False
    gan = tf.keras.Sequential()
    gan.add(g)
    gan.add(d)

    # configure model training
    gan.compile(
        loss=tf.keras.losses.binary_crossentropy,
        optimizer=tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5),
        metrics=["accuracy"]
    )

    gan.summary()

    # ----- TRAINING ----- #
    # callbacks for checkpoints, TensorBoard
    dir_name = "Results\\" + datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    checkpoint_dir = os.path.join(os.getcwd(), dir_name)
    save_callback, tb_callback = build_callbacks(checkpoint_dir)

    # training loop
    for epoch in range(NUM_EPOCHS+1):
        # generator
        noise_vector = np.random.normal(size=(BATCH_SIZE, Z_DIM))  # Gaussian noise
        gen_image = g.predict(noise_vector)  # generate 1 image

        # discriminator
        idx = np.random.randint(low=0, high=len(train_images), size=BATCH_SIZE)
        real_images = train_images[idx]  # batch of real images

        # concatenate real and generated images for discriminator
        d_images = np.concatenate((real_images, gen_image))

        # labels "real", "fake"
        real_labels = np.ones(shape=(BATCH_SIZE, 1))
        fake_labels = np.zeros(shape=(BATCH_SIZE, 1))

        # add random noise to labels

        # concatentate labels
        d_labels = np.concatenate((real_labels, fake_labels))

        # train discriminator
        d.trainable = True
        d_loss = d.train_on_batch(d_images, d_labels)

        # misleading labels for generator: "all these images are real" -- obviously a lie
        misleading_labels = np.ones(shape=(BATCH_SIZE, 1))

        # train generator
        d.trainable = False
        noise_vector = np.random.normal(size=(BATCH_SIZE, Z_DIM))  # Gaussian noise
        g_loss = gan.train_on_batch(noise_vector, misleading_labels)

        #
        # print metrics
        print("\nEpoch {}".format(epoch))
        print("Discriminator :: Loss: {:.4f}, Accuracy: {:.4f}".format(d_loss[0], d_loss[1]))
        print("Generator :: Loss: {:.4f}, Accuracy: {:.4f}".format(g_loss[0], g_loss[1]))

    # ----- GENERATE ----- #
