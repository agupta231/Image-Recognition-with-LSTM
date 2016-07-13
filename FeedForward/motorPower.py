import tensorflow as tf
from DataImport import DataImport
import glob as glob
import os

imageWidth = 400
imageHeight = 400
pixelCount = imageHeight * imageWidth

# Aux count should be 4 if ticks are enabled, 3 if ticks disabled
# auxVariables = 4
auxVariables = 3
channels = 1
epochs = 20000
batchSize = 50


# IMPORTING THE DATA
DI = DataImport("cropped_frames")

DI.setImage(imageWidth, imageHeight, channels)
dataFolders = glob.glob(os.getcwd() + "/../Experiments/Power_Frames/Basement*")

for path in dataFolders:
    DI.importFolder(path)

##############
# Code for Neural Network starts now
##############


def weight(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W, stride=1.0, padding=0):
    x = tf.to_float(x)
    x_padded = tf.pad(x, [[0, 0], [padding, padding], [padding, padding], [0, 0]], "CONSTANT")
    return tf.nn.conv2d(x_padded, W, strides=[1.0, stride, stride, 1.0], padding='VALID')


def max_pool(x, size, stride=2, padding=0):
    x_padded = tf.pad(x, [[0, 0], [padding, padding], [padding, padding], [0, 0]], "CONSTANT")
    return tf.nn.max_pool(x_padded, ksize=[1, size, size, 1], strides=[1, stride, stride, 1], padding='VALID')

# Make the Tensorflow session
session = tf.InteractiveSession()

# Make placeholders for the model
image = tf.placeholder(tf.int16, shape=[None, imageWidth, imageHeight, channels])
rightMotorPower = tf.placeholder(tf.float32, shape=[None, 1])
leftMotorPower = tf.placeholder(tf.float32, shape=[None, 1])
duration = tf.placeholder(tf.float32, shape=[None, 1])
# Don't want to use ticks yet, as lighting didn't change during the experiment
output_flattened_actual = tf.placeholder(tf.float32, shape=[None, pixelCount * channels])

W_conv1 = weight([4, 4, channels, 96])
b_conv1 = bias([96])

h_conv1 = tf.nn.relu(conv2d(image, W_conv1, stride=4.0) + b_conv1)
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

W_fc3 = weight([8192, pixelCount * channels])
b_fc3 = bias([pixelCount * channels])

output_flattened = tf.nn.relu(tf.matmul(h_fc2_dropout, W_fc3) + b_fc3)

# This tensor is used for dividing all of the difference values by a denominator to get a fractional error
denom_tensor = tf.fill([pixelCount * channels], 255.0)

cross_entropy = tf.reduce_mean(-tf.reduce_sum(output_flattened_actual * tf.log(output_flattened), reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
accuracy = tf.constant(1.0) - tf.reduce_mean(tf.abs(tf.div(tf.sub(output_flattened_actual, output_flattened), denom_tensor)))

session.run(tf.initialize_all_variables())
for i in range(epochs):
    batch = DI.next_batch(batchSize)

    if i % 100 == 0:
        train_accuracy = accuracy.eval(feed_dict={
            image: batch[0],
            rightMotorPower: batch[1],
            leftMotorPower: batch[2],
            duration: batch[3],
            output_flattened_actual: batch[4],
            keep_prob_fc1: 1.0,
            keep_prob_fc2: 1.0
        })
        print("Step %d, training accuracy %g" % (i, train_accuracy))

    train_step.run(feed_dict={
        image: batch[0],
        rightMotorPower: batch[1],
        leftMotorPower: batch[2],
        duration: batch[3],
        output_flattened_actual: batch[4],
        keep_prob_fc1: 0.5,
        keep_prob_fc2: 0.5
    })
