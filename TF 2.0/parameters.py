"""
Michael Patel
April 2020

Project description:
    Build a GAN using the MNIST data set

File description:
    For model and training parameters
"""
################################################################################
NUM_EPOCHS = 1
BATCH_SIZE = 16

LEARNING_RATE = 0.0002
BETA_1 = 0.5

LEAKY_ALPHA = 0.2  # default is 0.3
DROPOUT_RATE = 0.3

IMAGE_WIDTH = 28
IMAGE_HEIGHT = 28
NUM_CHANNELS = 1

NOISE_DIM = 100
