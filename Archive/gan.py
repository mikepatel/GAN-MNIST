'''
GAN
CSC 522

use MNIST data set to classify handwritten digits (0-9)
training set size: 60,000
test set size: 10,000
28 x 28 pixel images
monochrome images (greyscale)

v1 - use basic implementation from OReilly to see how runs/how well
v2 - max pool 2D improved
v3 - tweak leaky ReLU, tweak noise input
v4 - regular optimizer, tf.train.GradientDescentOptimizer
v5 - Adagrad, tf.train.AdagradOptimizer
v6 - softplus, tf.nn.softplus
v7 - sigmoid, tf.sigmoid
v8 - take out D sigmoid

'''
###################################################################################
# IMPORTS
import numpy as np
import tensorflow as tf
import datetime
#import matplotlib.pyplot as plt

###################################################################################
# load training and test data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/")

###################################################################################
# DISCRIMINATOR
def discriminator(images, reuse=False):
	if(reuse): # reset the runs
		tf.get_variable_scope().reuse_variables()
		
	# build as CNN binary classifier
	# CONVOLUTIONAL Layer #1
	'''
	- computes 32 features using 5x5 filter w/ ReLU activation
	- padding is added to preserve width and height
	- Input tensor shape: [batch_size, 28, 28, 1]
	  Output tensor shape: [batch_size, 28, 28, 32]
	'''
	# weight 1
	d_w1 = tf.get_variable(
		'd_w1', 
		[5,5,1,32], 
		initializer=tf.truncated_normal_initializer(stddev=0.05)) # increase standard deviation to get more variety?
	
	# bias 1
	d_b1 = tf.get_variable(
		'd_b1', 
		[32], 
		initializer=tf.constant_initializer(0))
							
	# convolution call
	d1 = tf.nn.conv2d(
		input=images, 
		filter=d_w1, 
		strides=[1, 1, 1, 1], 
		padding='SAME')
		
	d1 = d_b1 + d1 # add the bias term
	
	# perform activation
	#d1 = tf.nn.relu(d1)
	d1 = tf.nn.leaky_relu(
		features=d1,
		alpha=0.2,
		#alpha=0.1,
		#alpha=0.05,
		#alpha=0.01,
		#alpha=0.5,
		name=None
	)
	
	# POOLING Layer #1
	'''
	- first max pooling layer w/ 2x2 filter and stride=2
	- Input tensor shape: [batch_size, 28, 28, 32]
	  Output tensor shape: [batch_size, 14, 14, 32]
	'''
	'''
	d1 = tf.nn.avg_pool(
		d1, 
		ksize=[1, 2, 2, 1], 
		strides=[1, 2, 2, 1], 
		padding='SAME')
	'''
	
	d1 = tf.layers.max_pooling2d(
		inputs=conv1, 
		pool_size=[2,2], 
		strides=2)
	
	##
	# CONVOLUTIONAL Layer #2
	'''
	- computes 64 features using 5x5 filter w/ ReLU activation
	- padding is added to preserve width and height
	- Input tensor shape: [batch_size, 14, 14, 32]
	  Output tensor shape: [batch_size, 14, 14, 64]
	'''
	# weight 2
	d_w2 = tf.get_variable(
		'd_w2', 
		[5, 5, 32, 64], 
		initializer=tf.truncated_normal_initializer(stddev=0.05))
		
	# bias 2
    d_b2 = tf.get_variable(
		'd_b2', 
		[64], 
		initializer=tf.constant_initializer(0))
		
	# convolution call
    d2 = tf.nn.conv2d(
		input=d1, # get previous layer
		filter=d_w2, # use second layer
		strides=[1, 1, 1, 1], 
		padding='SAME')
		
	# add the bias term
    d2 = d2 + d_b2
	
    # perform activation
	#d2 = tf.nn.relu(d2)
	d2 = tf.nn.leaky_relu(
		features=d2,
		alpha=0.2,
		#alpha=0.1,
		#alpha=0.05,
		#alpha=0.01,
		#alpha=0.5,
		name=None
	)
	
	# POOLING Layer #2
	'''
	- second max pooling layer w/ 2x2 filter and stride=2
	- Input tensor shape: [batch_size, 14, 14, 64]
	  Output tensor shape: [batch_size, 7, 7, 64]
	'''
	pool2 = tf.layers.max_pooling2d(
		inputs=conv2, 
		pool_size=[2,2], 
		strides=2)
	
	##
	# FULLY CONNECTED Layer #1
	'''
	- densely connected layer w/ 1024 neurons
	- Input tensor shape: [batch_size, 7*7*64]
	  Output tensor shape: [batch_size, 1024]
	'''
	# weight 3
    d_w3 = tf.get_variable(
		'd_w3', 
		[7 * 7 * 64, 1024], 
		initializer=tf.truncated_normal_initializer(stddev=0.02))
	
	# bias 3
    d_b3 = tf.get_variable('d_b3', [1024], initializer=tf.constant_initializer(0))
    
	# flatten tensor into batch of vectors
	d3 = tf.reshape(d2, [-1, 7 * 7 * 64])
	
	# dot product
    d3 = tf.matmul(d3, d_w3)
	
	# add the bias term
    d3 = d3 + d_b3
	
	# batch normalization
	# reduce amount of hidden layer value shifting
	d3 = tf.contrib.layers.batch_norm(
		inputs=d3, 
		#decay=0.999,
		#decay=0.99,
		#decay=0.85,
		decay=0.88,
		center=True, 
		scale=True, 
		is_training=True)
	
	# perform activation
    d3 = tf.nn.relu(d3)
	d3 = tf.nn.leaky_relu(
		features=d3,
		alpha=0.2,
		#alpha=0.1,
		#alpha=0.05,
		#alpha=0.01,
		#alpha=0.5,
		name=None
	)

    ##
	# FULLY CONNECTED Layer #2	
	# weight 4
    d_w4 = tf.get_variable(
		'd_w4', 
		[1024, 1], 
		initializer=tf.truncated_normal_initializer(stddev=0.02))
	
	# bias 4
    d_b4 = tf.get_variable('d_b4', [1], initializer=tf.constant_initializer(0))
    
	# dot product
	d4 = tf.matmul(d3, d_w4)
	
	# add the bias term
	d4 = d4 + d_b4
	
	# sigmoid to help classify - take out otherwise D gets too good -> no G learning
	#d4 = tf.nn.sigmoid(d4)
	
	return d4

###################################################################################
# GENERATOR
def generator(z_noise, batch_size, z_noise_dim):
	# Layer #1
	
	# weight 1
	g_w1 = tf.get_variable(
		'g_w1', 
		#[z_noise_dim, 28*28*4], 
		[z_noise_dim, 28*28*3],
		dtype=tf.float32, 
		initializer=tf.truncated_normal_initializer(stddev=0.05))
	
	# bias 1
    g_b1 = tf.get_variable(
		'g_b1', 
		[28*28*3], 
		initializer=tf.truncated_normal_initializer(stddev=0.05))
	
	# dot product
    g1 = tf.matmul(z_noise, g_w1)
	
	# add the bias term
	g1 = g1 + g_b1
	
	# upsample (via reshape)
    g1 = tf.reshape(g1, [-1, 56, 56, 1])
	
	# batch normalization
    g1 = tf.contrib.layers.batch_norm(
		g1, 
		#epsilon=1e-5
		epsilon=1e-4
		)
		
    # perform activation
    #g1 = tf.nn.relu(g1)
	g1 = tf.nn.leaky_relu(
		features=g1,
		#alpha=0.2,
		alpha=0.1,
		#alpha=0.05,
		#alpha=0.01,
		#alpha=0.5,
		name=None
	)
	
	##
	# Layer 2
	# weight 2
	g_w2 = tf.get_variable(
		'g_w2', 
		#[3, 3, 1, z_noise_dim/2],
		shape=[1024, 7*7*64], 
		dtype=tf.float32, 
		initializer=tf.truncated_normal_initializer(stddev=0.05))
		
	# bias 2
    g_b2 = tf.get_variable(
		'g_b2', 
		#[z_noise_dim/2], 
		shape=[1024],
		initializer=tf.truncated_normal_initializer(stddev=0.05))
		
    g2 = tf.nn.conv2d(
		g1, 
		g_w2, 
		strides=[1, 2, 2, 1], 
		padding='SAME')
	
	# add the bias term
    g2 = g2 + g_b2
	
	# batch normalization
    g2 = tf.contrib.layers.batch_norm(
		g2, 
		#epsilon=1e-5
		epsilon=1e-4
		)
		
	# perform activation
	#g2 = tf.nn.relu(g2)
	g2 = tf.nn.leaky_relu(
		features=g2,
		#alpha=0.2,
		alpha=0.1,
		#alpha=0.05,
		#alpha=0.01,
		#alpha=0.5,
		name=None
	)
	
	# reshape 
    g2 = tf.image.resize_images(g2, [56, 56])
	#g2 = tf.reshape(g2, [batch_size,7,7,64])
	
	##
	# Layer 3
	# weight 3
	g_w3 = tf.get_variable(
		'g_w3', 
		#[3, 3, z_noise_dim/2, z_noise_dim/4], 
		shape=[5,5,32, 64],
		dtype=tf.float32, 
		initializer=tf.truncated_normal_initializer(stddev=0.05))
		
	# bias 3
    g_b3 = tf.get_variable(
		'g_b3', 
		#[z_nosie_dim/4], 
		shape=[32],
		initializer=tf.truncated_normal_initializer(stddev=0.05))
		
    g3 = tf.nn.conv2d(
		g2, 
		g_w3, 
		strides=[1, 2, 2, 1], 
		padding='SAME')
		
	# add the bias term
    g3 = g3 + g_b3
    
	# batch normalization
    g3 = tf.contrib.layers.batch_norm(
		g3, 
		#epsilon=1e-5
		epsilon=1e-4
		)
		
	# perform activation
	#g3 = tf.nn.relu(g3)
	g3 = tf.nn.leaky_relu(
		features=g3,
		#alpha=0.2,
		alpha=0.1,
		#alpha=0.05,
		#alpha=0.01,
		#alpha=0.5,
		name=None
	)
	
	# reshape
    g3 = tf.image.resize_images(g3, [56, 56])
	
	##
	# Layer 4
	# weight 4
	g_w4 = tf.get_variable(
		'g_w4', 
		#[1, 1, z_noise_dim/4, 1], 
		shape=[5,5,1,32],
		dtype=tf.float32, 
		initializer=tf.truncated_normal_initializer(stddev=0.05))
    
	# bias 4
	g_b4 = tf.get_variable(
		'g_b4', 
		[1], 
		initializer=tf.truncated_normal_initializer(stddev=0.05))
		
	
    g4 = tf.nn.conv2d(
		g3, 
		g_w4, 
		strides=[1, 2, 2, 1], 
		padding='SAME')
		
	# add the bias term
    g4 = g4 + g_b4
	
	#perform activation
	#g4 = tf.nn.relu(g4)
	'''
	g4 = tf.nn.leaky_relu(
		features=g4,
		#alpha=0.2,
		alpha=0.1,
		#alpha=0.05,
		#alpha=0.01,
		#alpha=0.5,
		name=None
	)
	'''
    g4 = tf.sigmoid(g4)
	# sigmoid helps to push grey pixels to black/white extremes, better contrast
	
	return g4 # generated image
	
###################################################################################
# SETUP
#z_noise = 100
#z_noise = 300
z_noise = 128

#batch_size = 50
batch_size = 64
#batch_size = 32

# the input to D
x_in = tf.placeholder(
	tf.float32,
	shape=[None, 28, 28, 1],
	name='x_placeholder'
)

# the input to G
z_noise_in = tf.placeholder(
	tf.float32,
	shape=[None, z_noise],
	name='z_noise_placeholder'
)

# generator - noise
G = generator(
	z_noise_in,
	batch_size,
	z_noise
)

# discriminator - real (MNIST)
Dx = discriminator(x_placeholder)

# discriminator - gen (G)
Dg = discriminator(G, reuse=True)

###################################################################################
# TRAINING

# Loss functions (objective functions)
d_loss_real = tf.reduce_mean(
	tf.nn.sigmoid_cross_entropy_with_logits(
		logits=Dx, 
		labels=tf.ones_like(Dx))) # cross real w/ 1

d_loss_gen = tf.reduce_mean(
	tf.nn.sigmoid_cross_entropy_with_logits(
		logits=Dg, 
		labels=tf.zeros_like(Dg))) # cross gen w/ 0

g_loss = tf.reduce_mean(
	tf.nn.sigmoid_cross_entropy_with_logits(
		logits=Dg, 
		labels=tf.ones_like(Dg))) # cross gen from G w/ 1
		
# variable lists
tvars = tf.trainable_variables()
d_vars = [var for var in tvars if 'd_' in var.name]
g_vars = [var for var in tvars if 'g_' in var.name]

# Optimizers
'''
# Gradient Descent optimizer
learn = 0.001
d_trainer_real = tf.train.GradientDescentOptimizer(learning_rate=learn).minimize(d_loss_real, var_list=d_vars)
d_trainer_gen = tf.train.GradientDescentOptimizer(learning_rate=learn).minimize(d_loss_gen, var_list=d_vars)
g_trainer = tf.train.GradientDescentOptimizer(learning_rate=learn).minimize(g_loss, var_list=g_vars)

'''

'''
# Adagrad optimizer
#learn = 0.001
learn = 0.0002
accum = 0.1

d_trainer_real = tf.train.AdagradOptimizer(
	learning_rate=learn,
	initial_accumulator_value=accum).minimize(d_loss_real, var_list=d_vars)
	
d_trainer_gen = tf.train.AdagradOptimizer(
	learning_rate=learn,
	initial_accumulator_value=accum).minimize(d_loss_gen, var_list=d_vars)
	
g_trainer = tf.train.AdagradOptimizer(
	learning_rate=learn,
	initial_accumulator_value=accum).minimize(g_loss, var_list=g_vars)
'''

# Adam optimizer
d_trainer_real = tf.train.AdamOptimizer(0.0002, beta1=0.9).minimize(d_loss_real, var_list=d_vars)

d_trainer_gen = tf.train.AdamOptimizer(0.0002, beta1=0.9).minimize(d_loss_gen, var_list=d_vars)

g_trainer = tf.train.AdamOptimizer(0.0002, beta1=0.9).minimize(g_loss, var_list=g_vars)

# run training
sess = tf.Session()

# tensorboard to view training
tf.summary.scalar('Generator_loss', g_loss)
tf.summary.scalar('Discriminator_loss_real', d_loss_real)
tf.summary.scalar('Discriminator_loss_fake', d_loss_gen)

images_for_tensorboard = generator(z_noise_in, batch_size, z_noise)
tf.summary.image('Generated_images', images_for_tensorboard, 20)
merged = tf.summary.merge_all()
logdir = "tensorboard/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "/"
writer = tf.summary.FileWriter(logdir, sess.graph)

sess.run(tf.global_variables_initializer())

# training iterations
for(i in range(1000000)):
#for(i in range(10000000)):
#for(i in range(50000)):
	real_image_batch = mnist.train.next_batch(batch_size)[0].reshape([batch_size, 28, 28, 1]) # for D
	
	z_noise_batch = np.random.normal(0, 1, size=[batch_size, z_noise]) # for G
	#z_noise_batch = np.random.normal(1, 2, size=[batch_size, z_noise]) # for G
	
	# train discriminator on both real and fake images
    _, __, dLossReal, dLossFake = sess.run(
		[d_trainer_real, d_trainer_gen, d_loss_real, d_loss_gen],
		{x_in: real_image_batch, z_noise_in: z_noise_batch})
		
	# train generator
	z_noise_batch = np.random.normal(0, 1, size=[batch_size, z_noise])
		_ = sess.run(g_trainer, feed_dict={z_noise_in: z_noise_batch})
		
		
	if((i % 100) == 0): # how often to update tensorboard
		z_noise_batch = np.random.normal(0, 1, size=[batch_size, z_noise])
        summary = sess.run(merged, {z_noise_in: z_noise_batch, x_in: real_image_batch})
		writer.add_summary(
			summary=summary, 
			global_step=i)