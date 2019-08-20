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
import time
import matplotlib.pyplot as plt
import imageio  # generate gifs
import glob  # module finds all pathnames matching a specified pattern

import tensorflow as tf

from parameters import *
from model import Generator, Discriminator


################################################################################
# Discriminator Loss Function
def discriminator_loss(real_output, generated_output):
    real_loss = tf.losses.sigmoid_cross_entropy(
        multi_class_labels=tf.ones_like(real_output),
        logits=real_output
    )

    generated_loss = tf.losses.sigmoid_cross_entropy(
        multi_class_labels=tf.zeros_like(generated_output),
        logits=generated_output
    )

    total_loss = real_loss + generated_loss

    return total_loss


# Generator Loss Function
def generator_loss(generated_output):
    generated_loss = tf.losses.sigmoid_cross_entropy(
        multi_class_labels=tf.ones_like(generated_output),
        logits=generated_output
    )

    return generated_loss


# Generate and Save Images
def generate_and_save_images(model, epoch, test_input, save_dir):
    # set training parameter to False b/c don't want to train batchnorm layer when doing inference
    predictions = model(test_input, training=False)

    # plot 4x4 grid of generated images
    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i+1),
        plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap="gray")
        plt.axis("off")

    # save 4x4 grid of generated images
    fig_name = os.path.join(save_dir, "Epoch {:04d}.png".format(epoch))
    plt.savefig(fig_name)
    plt.close()


# Training
def train(dataset, epochs, noise_dim, discriminator, generator, save_dir):
    for epoch in range(epochs):
        start = time.time()

        for images in dataset:
            # generate noise from uniform distribution
            noise = tf.random_normal(shape=[BATCH_SIZE, noise_dim])

            # GradientTape -->
            with tf.GradientTape() as g_tape, tf.GradientTape() as d_tape:
                # generator
                generated_images = generator(noise, training=True)

                # discriminator
                real_output = discriminator(images, training=True)
                generated_output = d(generated_images, training=True)

                # loss functions
                g_loss = generator_loss(generated_output)
                d_loss = discriminator_loss(real_output, generated_output)

            g_gradients = g_tape.gradient(g_loss, g.variables)
            d_gradients = d_tape.gradient(d_loss, d.variables)

            #
            g_optimizer.apply_gradients(zip(g_gradients, g.variables))
            d_optimizer.apply_gradients(zip(d_gradients, d.variables))

        if epoch % 1 == 0:
            # generate and save image per each epoch
            generate_and_save_images(g, epoch+1, random_vector_for_generation, save_dir)

        # save checkpoints
        if (epoch+1) % 1 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)

        print("Time taken for epoch {} is {:.4f}s".format(epoch+1, time.time()-start))

    # generate and save images after final epoch
    generate_and_save_images(g, epochs, random_vector_for_generation, save_dir)


################################################################################
# Main
if __name__ == "__main__":
    # enable eager execution
    tf.enable_eager_execution()

    # print out TF version
    print("TF version: {}".format(tf.__version__))

    # create output directory for checkpoints, results, images
    output_dir = "Results\\" + datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

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

    # use tf.data.Dataset to create batches and shuffle --> data pipeline to TF model
    train_dataset = tf.data.Dataset.from_tensor_slices(train_images)
    train_dataset = train_dataset.shuffle(buffer_size=BUFFER_SIZE)
    train_dataset = train_dataset.batch(batch_size=BATCH_SIZE)

    print("Shape of batches: {}".format(train_dataset.output_shapes))

    # ----- MODEL ----- #
    g = Generator()
    d = Discriminator()

    # defun gives 10s per epoch performance boost
    g.call = tf.contrib.eager.defun(g.call)
    d.call = tf.contrib.eager.defun(d.call)

    # Optimizers
    d_optimizer = tf.train.AdamOptimizer(learning_rate=D_LEARNING_RATE)
    g_optimizer = tf.train.AdamOptimizer(learning_rate=G_LEARNING_RATE)

    # Checkpoints
    checkpoint_dir = output_dir
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(
        g_optimizer=g_optimizer,
        d_optimizer=d_optimizer,
        g=g,
        d=d
    )

    # ----- TRAINING ----- #
    # keep random vector constant for generation to track gan improvement easier
    random_vector_for_generation = tf.random_normal(shape=[NUM_GEN_IMAGES, NOISE_DIM])

    # Train function that will save results per epoch
    train(train_dataset, NUM_EPOCHS, NOISE_DIM, d, g, output_dir)

    # Restore latest checkpoint
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir=checkpoint_dir))

    # Generate gif of all saved images
    gif_filename = os.path.join(output_dir, "dcgan.gif")
    with imageio.get_writer(gif_filename, mode="I") as writer:
        image_files_pattern = output_dir + "\\Epoch*.png"
        filenames = glob.glob(image_files_pattern)
        filenames = sorted(filenames)

        last = -1

        for i, filename in enumerate(filenames):
            frame = 2*(i**0.5)

            if round(frame) > round(last):
                last = frame

            else:
                continue

            image = imageio.imread(filename)
            writer.append_data(image)
