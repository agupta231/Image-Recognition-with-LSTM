import tensorflow as tf
import os
import glob
import numpy as np

# Setup seed
np.random.seed(420)

# Setup parameters
IMAGE_HEIGHT = 150
IMAGE_WIDTH = 150
IMAGE_CHANNELS = 1
PIXEL_COUNT = IMAGE_HEIGHT * IMAGE_WIDTH * IMAGE_CHANNELS
AUX_INPUTS = 2
FREQUENCY = 60

LEARNING_RATE = 0.001
TIME_STEPS = 4
BATCH_SIZE = 16
LOG_STEP = 10
ITERATIONS = 10000

CELL_SIZE = 128
CELL_LAYERS = 10
HIDDEN_SIZE = 900
OUTPUT_SIZE = 2


# Helper functions
def create_weight(shape):
    initial = tf.truncated_normal(shape=shape, stddev=0.1)
    return tf.Variable(initial)


def create_bias(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


# The actual model
input_sequence = tf.placeholder(tf.int8, [BATCH_SIZE, TIME_STEPS, PIXEL_COUNT + AUX_INPUTS])
output_actual = tf.placeholder(tf.int8, [BATCH_SIZE, OUTPUT_SIZE])

lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(CELL_SIZE, state_is_tuple=True)
stacked_lstm = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * CELL_LAYERS, state_is_tuple=True)

initial_state = state = stacked_lstm.zero_state(BATCH_SIZE, tf.float32)
outputs = []

with tf.variable_scope("LSTM"):
    for step in xrange(TIME_STEPS):
        if step > 0:
            tf.get_variable_scope().reuse_variables()
        cell_output, state = stacked_lstm(input_sequence[:, step, :], state)
        outputs.append(cell_output)

final_state = state

output = tf.reshape(tf.concat(1, outputs), [-1, HIDDEN_SIZE])

softmax_w = tf.get_variable("softmax_w", [HIDDEN_SIZE, OUTPUT_SIZE], dtype=tf.float32)
softmax_b = tf.get_variable("softmax_b", [OUTPUT_SIZE], dtype=tf.float32)
prediction = tf.nn.softmax(tf.matmul(output, softmax_w) + softmax_b)

cross_entropy = tf.reduce_mean(-tf.reduce_sum(output_actual * tf.log(prediction), reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(output_actual, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    numpy_state = initial_state.eval()

    for i in xrange(ITERATIONS):
        numpy_state, train_step = sess.run([numpy_state, train_step], feed_dict={

        })