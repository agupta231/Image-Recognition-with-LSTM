import time
import tensorflow as tf
import numpy as np
import plotly.plotly as py
import plotly.graph_objs as go

imageWidth = 640
imageHeight = 480
pixelCount = imageHeight * imageWidth

auxCount = 4
channels = 1


def weight(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W, stride=1, padding=0):
    x_padded = tf.pad(x, [[padding, padding], [padding, padding]], "CONSTANT")
    return tf.nn.conv2d(x_padded, W, strides=[1, stride, stride, 1], padding='VALID')


def max_pool(x, size, stride=2, padding=0):
    x_padded = tf.pad(x, [[padding, padding], [padding, padding]], "CONSTANT")
    return tf.nn.max_pool(x_padded, ksize=[1, size, size, 1], strides=[1, stride, stride, 1], padding='VALID')


session = tf.InteractiveSession()

# Y'ALL GOTTA DEAL WITH THIS BAD LARRY LATER
input_image = tf.reshape(indep, [-1, 28, 28, 1])


W_conv1 = weight([4, 4, channels, 96])
b_conv1 = bias([96])

h_conv1 = tf.nn.relu(conv2d(input_image, W_conv1, stride=4) + b_conv1)
h_pool1 = max_pool(h_conv1, 2)

W_conv2 = weight([5, 5, 96, 128])
b_conv2 = bias([128])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2, padding=2) + b_conv2)
h_pool2 = max_pool(h_conv2, 4)

W_conv3 = weight([3, 3, 128, 256])
b_conv3 = bias([256])

h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3, padding=1) + b_conv3)

W_conv4 = weight([3, 3, 256, 384])
b_conv4 = bias([384])

h_conv4 = tf.nn.relu(conv2d(h_conv3, W_conv4, padding=1) + b_conv4)

W_conv5 = weight([3, 3, 384, 256])
b_conv5 = bias([256])

h_conv5 = tf.nn.relu(conv2d(h_conv4, W_conv5, padding=1) + b_conv5)

h_pool3 = max_pool(h_conv5, 4)

W_fc1 = weight([11 * 11 * 256, 4096])
b_fc1 = bias([4096])

h_pool3_flat = tf.reshape(h_pool3, [-1, 11 * 11 * 256])
input_tensor_final = tf.concat(0, [h_pool3_flat, ])

h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc1) + b_fc1)