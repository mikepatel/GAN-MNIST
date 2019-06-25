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

Things to examine:

"""
################################################################################
# based on MNIST images: 28x28 greyscale
num_rows = 28
num_cols = 28
num_channels = 1

# Model hyperparameters
NUM_EPOCHS = 1
BATCH_SIZE = 64
BUFFER_SIZE = 60000  # size of training set
Z_DIM = 100
NUM_GEN = 16
