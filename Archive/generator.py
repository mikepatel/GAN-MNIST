import tensorflow as tf
import tensorflow as tf
import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt

def generator(noise, batch_size, keep_prob, reuse = False):
	with tf.variable_scope('generator') as scope:
		if(reuse):
			tf.get_variable_scope().reuse_variables()
		
		input_vector = noise
		generator_weight1 = tf.get_variable('generator_weight1', shape = [32,1024], initializer=tf.contrib.layers.xavier_initializer())
		generator_bias1 = tf.get_variable('generator_bias1', shape = [1024], initializer = tf.constant_initializer(0))
		fc1_output = tf.matmul(input_vector, generator_weight1) + generator_bias1
		#fc1_output = tf.nn.dropout(fc1_output, keep_prob = keep_prob)
		fc1_output_bn = tf.contrib.layers.batch_norm(inputs = fc1_output, decay = 0.9, center=True, scale=True, is_training=True, scope = 'generator_bn1')
		fc2_input = tf.nn.leaky_relu(features = fc1_output_bn, alpha = 0.2)

		generator_weight2 = tf.get_variable('generator_weight2', shape = [1024,7*7*64], initializer=tf.contrib.layers.xavier_initializer())
		generator_bias2 = tf.get_variable('generator_bias2', shape = [7*7*64], initializer = tf.constant_initializer(0))
		fc2_output = tf.matmul(fc2_input, generator_weight2) + generator_bias2
		#fc2_output = tf.nn.dropout(fc2_output, keep_prob = keep_prob)
		fc2_output_bn = tf.contrib.layers.batch_norm(inputs = fc2_output, decay = 0.9, center=True, scale=True, is_training=True, scope = 'generator_bn2')
		deconv1_input = tf.nn.leaky_relu(features = fc2_output_bn, alpha = 0.2)
		deconv1_input_reshape = tf.reshape(deconv1_input, [batch_size,7,7,64])

		generator_weight3 = tf.get_variable('generator_weight3', shape = [5, 5, 32, 64], initializer=tf.contrib.layers.xavier_initializer())
		generator_bias3 = tf.get_variable('generator_bias3', shape = [32], initializer = tf.constant_initializer(0))
		deconv1_output = tf.nn.conv2d_transpose(value = deconv1_input_reshape, filter = generator_weight3, output_shape = [batch_size, 14, 14, 32], strides = [1,2,2,1], padding = 'SAME') + generator_bias3
		#deconv1_output = tf.nn.dropout(deconv1_output, keep_prob = keep_prob)
		deconv1_output_bn = tf.contrib.layers.batch_norm(inputs = deconv1_output, decay = 0.9, center=True, scale=True, is_training=True, scope = 'generator_bn3')
		deconv2_input = tf.nn.leaky_relu(features = deconv1_output_bn, alpha = 0.2)

		generator_weight4 = tf.get_variable('generator_weight4', shape = [5, 5, 1, 32], initializer=tf.contrib.layers.xavier_initializer())
		generator_bias4 = tf.get_variable('generator_bias4', shape = [1], initializer = tf.constant_initializer(0))
		deconv2_output = tf.nn.conv2d_transpose(value = deconv2_input, filter = generator_weight4, output_shape = [batch_size, 28, 28, 1], strides = [1,2,2,1], padding = 'SAME') + generator_bias4
		#generator_conv4_output = tf.contrib.layers.batch_norm(generator_conv4_output, decay = 0.9)
		generated_image = tf.nn.sigmoid(deconv2_output)

		return generated_image

'''
sess = tf.Session()
noise_dimension = 100
noise_value = tf.placeholder(tf.float32, [None, noise_dimension])
generator_output = generator(noise_value,1)
test_input = np.random.normal(-1, 1, [1,noise_dimension])
sess.run(tf.global_variables_initializer())
test = (sess.run(generator_output, feed_dict = {noise_value: test_input}))
#print(test)
plt.imshow(test.squeeze(), cmap = 'gray_r')
plt.show()
'''


