import time
import tensorflow as tf
import numpy as np
import plotly.plotly as py
import plotly.graph_objs as go

imageWidth = 640
imageHeight = 480
pixelCount = imageHeight * imageWidth

auxCount = 4


def weight(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

session = tf.InteractiveSession()

W_conv1 = weight([15, 15, 1, 64])
b_conv1 = bias([64])

input_image = tf.reshape(indep, [-1, 28, 28, 1])
