import time
import tensorflow as tf
import numpy as np

## Constants
imageWidth = 640
imageHeight = 480
pixelCount = imageHeight * imageWidth

auxCount = 4
inputSize = auxCount + pixelCount

## Functions


def weight(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

session = tf.InteractiveSession()

input = tf.placeholder(tf.float32, shape=[None, inputSize])
