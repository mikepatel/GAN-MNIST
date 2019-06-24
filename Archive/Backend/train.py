# Notes:
#   - images are 28x28x1 (MNIST data set)


################################################################################
# IMPORTS
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.preprocessing import image

from build_model import *

import os
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt


################################################################################
BATCH_SIZE = 256
NUM_EPOCHS = 100000
Z_DIM = 100
LEARNING_RATE = 0.0002
BETA1 = 0.5

# MNIST images are 28x28x1
IMAGE_ROWS = 28
IMAGE_COLS = 28
IMAGE_CHANNELS = 1


################################################################################
# TRAIN GAN
def train_gan():
    # SETUP
    print("\nTF version: {}".format(tf.__version__))

    save_folder = os.path.join(os.getcwd(), datetime.now().strftime("%d-%m-%Y_%H-%M-%S"))
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    tf.reset_default_graph()  # clears default graph stack
    sess = tf.Session()

    # LOAD DATA SET
    (train_images, train_labels), (_, _) = mnist.load_data()
    print(train_images.shape)

    # rescale -1 to 1 and normalize
    # use in coordination with tanh activation
    train_images = train_images.reshape(train_images.shape[0], 28, 28, 1)
    #train_images = tf.image.resize_image_with_pad(train_images, target_height=32, target_width=32).eval(session=sess)
    train_images = train_images.astype(np.float32)
    train_images = (train_images - 127.5) / 127.5
    #print(type(train_images))
    print(train_images.shape)

    # placeholders to feed model
    noise_pl = tf.placeholder(dtype=tf.float32, shape=[None, Z_DIM])
    image_pl = tf.placeholder(dtype=tf.float32, shape=[None, IMAGE_ROWS, IMAGE_COLS, IMAGE_CHANNELS])

    # instantiate models
    g = build_generator(noise_pl, reuse=False)
    d_real = build_discriminator(image_pl, reuse=False)
    d_fake = build_discriminator(g, reuse=True)

    # LOSS FUNCTIONS
    d_real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        logits=d_real,
        labels=tf.ones_like(d_real)
    ))
    d_fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        logits=d_fake,
        labels=tf.zeros_like(d_fake)
    ))

    d_loss = d_real_loss + d_fake_loss

    g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        logits=d_fake,
        labels=tf.ones_like(d_fake)
    ))

    # Adam Optimizer
    d_optimizer = tf.train.AdamOptimizer(
        learning_rate=LEARNING_RATE,
        beta1=BETA1
    ).minimize(
        loss=d_loss,
        var_list=[tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="discriminator")]
    )
    g_optimizer = tf.train.AdamOptimizer(
        learning_rate=LEARNING_RATE,
        beta1=BETA1
    ).minimize(
        loss=g_loss,
        var_list=[tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="generator")]
    )

    # Session and TensorBoard
    # track loss values
    tf.summary.scalar(name="D Loss", tensor=d_loss)
    tf.summary.scalar(name="G Loss", tensor=g_loss)

    # log generated images
    tf.summary.image(
        name="Generated Images",
        tensor=build_generator(noise_pl, reuse=True),
        max_outputs=20
    )

    tb = tf.summary.merge_all()

    tb_writer = tf.summary.FileWriter(logdir=save_folder, graph=sess.graph)

    sess.run(tf.global_variables_initializer())

    #
    for epoch in range(NUM_EPOCHS+1):
        #indices = np.random.randint(0, train_images.shape[0], size=BATCH_SIZE)
        indices = np.random.randint(0, 60000, size=BATCH_SIZE)
        real_images = train_images[indices]

        z = np.random.normal(size=(BATCH_SIZE, Z_DIM))  # Gaussian noise

        d_error, _ = sess.run(
            [d_loss, d_optimizer],
            feed_dict={
                image_pl: real_images,
                noise_pl: z
            }
        )

        z = np.random.normal(size=(BATCH_SIZE, Z_DIM))  # Gaussian noise

        g_error, _ = sess.run(
            [g_loss, g_optimizer],
            feed_dict={
                noise_pl: z
            }
        )

        if epoch % 100 == 0:
            print("\nEpoch: {}".format(epoch))
            print("D LOSS: {}, G LOSS: {}".format(d_error, g_error))

            # generate images to TensorBoard
            z = np.random.normal(size=(BATCH_SIZE, Z_DIM))  # Gaussian noise

            tb_summary = sess.run(
                tb,
                feed_dict={
                    image_pl: real_images,
                    noise_pl: z
                }
            )

            tb_writer.add_summary(summary=tb_summary, global_step=epoch)

        if epoch % 100 == 0:
            z = np.random.normal(size=(BATCH_SIZE, Z_DIM))  # Gaussian noise

            gen_img = sess.run(
                [build_generator(noise_pl, reuse=True)],
                feed_dict={
                    noise_pl: z
                }
            )

            row = 8
            col = 8

            '''
            # real images
            figure = plt.figure(figsize=(10, 10))
            for i in range(1, col*row+1):
                img = image.array_to_img(train_images[i-1] * 255., scale=False)
                figure.add_subplot(row, col, i)
                plt.imshow(img)
                plt.axis("off")
                plt.subplots_adjust(wspace=0, hspace=0)
            plt.savefig(save_folder + "\\" + str(epoch) + "_real" + ".png")
            plt.close()
            '''

            # generated images
            figure = plt.figure(figsize=(4, 4))
            for i in range(1, col*row+1):
                img = image.array_to_img(gen_img[0][i-1] * 255.0, scale=False)
                figure.add_subplot(row, col, i)
                plt.imshow(img)
                plt.axis("off")
                plt.subplots_adjust(wspace=0, hspace=0)
            plt.savefig(save_folder + "\\" + str(epoch) + "_gen" + ".png")
            plt.close()


################################################################################
# MAIN
if __name__ == "__main__":
    train_gan()
