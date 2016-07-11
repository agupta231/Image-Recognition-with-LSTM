import tensorflow as tf

imageWidth = 400
imageHeight = 400
pixelCount = imageHeight * imageWidth

# Aux count should be 4 if ticks are enabled, 3 if ticks disabled
# auxVariables = 4
auxVariables = 3
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

# Make the Tensorflow session
session = tf.InteractiveSession()

# Make placeholders for the model
image = tf.placeholder(tf.int16, shape=[None, imageWidth, imageHeight, channels])
rightMotorPower = tf.placeholder(tf.int16, shape=[None, 1])
leftMotorPower = tf.placeholder(tf.int16, shape=[None, 1])
duration = tf.placeholder(tf.int16, shape=[None, 1])
# Don't want to use ticks yet, as lighting didn't change during the experiment

output_image_actual = tf.placeholder(tf.int16, shape=[None, imageWidth, imageHeight, channels])

W_conv1 = weight([4, 4, channels, 96])
b_conv1 = bias([96])

h_conv1 = tf.nn.relu(conv2d(image, W_conv1, stride=4) + b_conv1)
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

h_pool3_flat = tf.reshape(h_pool3, [-1, 11 * 11 * 256])
input_tensor_final = tf.concat(1, [h_pool3_flat, rightMotorPower, leftMotorPower, duration])

W_fc1 = weight([11 * 11 * 256 + auxVariables, 4096])
b_fc1 = bias([4096])

h_fc1 = tf.nn.relu(tf.matmul(input_tensor_final, W_fc1) + b_fc1)

keep_prob_fc1 = tf.placeholder(tf.float32)
h_fc1_dropout = tf.nn.dropout(h_fc1, keep_prob_fc1)

W_fc2 = weight([4096, 8192])
b_fc2 = bias([8192])

h_fc2 = tf.nn.relu(tf.matmul(h_fc1_dropout, W_fc2) + b_fc2)

keep_prob_fc2 = tf.placeholder(tf.float32)
h_fc2_dropout = tf.nn.dropout(h_fc2, keep_prob_fc2)

W_fc3 = weight([8192, pixelCount])
b_fc3 = bias([pixelCount])

output = tf.nn.relu(tf.matmul(h_fc2_dropout, W_fc3) + b_fc3)
output_image = x_image = tf.reshape(output, [-1, imageWidth, imageHeight, channels])

cross_entropy = tf.reduce_mean(-tf.reduce_sum(output_image_actual * tf.log(output_image), reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
