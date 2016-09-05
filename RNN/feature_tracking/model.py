from data_import import DataImport
import tensorflow as tf
import numpy as np
import threading
import Queue
import glob
import os

# Setup seed
np.random.seed(420)

# Setup parameters
IMAGE_HEIGHT = 150
IMAGE_WIDTH = 150
IMAGE_CHANNELS = 1
PIXEL_COUNT = IMAGE_HEIGHT * IMAGE_WIDTH * IMAGE_CHANNELS
AUX_INPUTS = 2
FREQUENCY = 60

FRAMES_FOLDER = "edges_1.75"
DISTANCE_DATA = "distances_sigma_2.25_1.5.txt"
THRESHOLD = 30

LEARNING_RATE = 0.001
SEQUENCE_SPACING = 1.024  # In seconds
TIME_STEPS = 4
BATCH_SIZE = 32
LOG_STEP = 5
ITERATIONS = 10000

CELL_SIZE = 256
CELL_LAYERS = 10
HIDDEN_SIZE = 256
OUTPUT_SIZE = 2

REGENERATE_CHUNKS = True

summary_save_dir = os.getcwd() + "/summaries/" + FRAMES_FOLDER + "_" + DISTANCE_DATA + "_lr" + str(LEARNING_RATE) + "_t" + str(THRESHOLD) + "_bs" + str(BATCH_SIZE) + "_ts" + str(TIME_STEPS) + "_p" + str(SEQUENCE_SPACING) + "_cs" + str(CELL_SIZE) + "x" + str(CELL_LAYERS) + "x" + str(HIDDEN_SIZE)
os.mkdir(summary_save_dir)

DI = DataImport(FRAMES_FOLDER, SEQUENCE_SPACING, DISTANCE_DATA, THRESHOLD, BATCH_SIZE, TIME_STEPS, channels=IMAGE_CHANNELS, image_size=IMAGE_WIDTH)

if REGENERATE_CHUNKS:
    os.mkdir(os.getcwd() + "/chunks")

    dataFolders = [path for path in glob.glob(os.getcwd() + "/*") if
                   os.path.isdir(path) and not "chunks" in path and not "done" in path]
    for path in dataFolders:
        DI.import_folder(path)


# Helper functions
def create_weight(shape):
    initial = tf.truncated_normal(shape=shape, stddev=0.1)
    return tf.Variable(initial)


def create_bias(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


# The actual model
input_sequence = tf.placeholder(tf.float32, [BATCH_SIZE, TIME_STEPS, PIXEL_COUNT + AUX_INPUTS])
output_actual = tf.placeholder(tf.float32, [BATCH_SIZE, OUTPUT_SIZE])

lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(CELL_SIZE, state_is_tuple=False)
stacked_lstm = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * CELL_LAYERS, state_is_tuple=False)

initial_state = state = stacked_lstm.zero_state(BATCH_SIZE, tf.float32)
outputs = []

with tf.variable_scope("LSTM"):
    for step in xrange(TIME_STEPS):
        if step > 0:
            tf.get_variable_scope().reuse_variables()
        cell_output, state = stacked_lstm(input_sequence[:, step, :], state)
        outputs.append(cell_output)

final_state = state

# output = tf.reshape(tf.concat(1, outputs), [-1, HIDDEN_SIZE])
output = tf.reshape(outputs[-1], [BATCH_SIZE, HIDDEN_SIZE])

softmax_w = tf.get_variable("softmax_w", [HIDDEN_SIZE, OUTPUT_SIZE], dtype=tf.float32)
softmax_b = tf.get_variable("softmax_b", [OUTPUT_SIZE], dtype=tf.float32)
prediction = tf.nn.softmax(tf.matmul(output, softmax_w) + softmax_b)

cross_entropy = tf.reduce_mean(-tf.reduce_sum(output_actual * tf.log(prediction), reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(output_actual, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

tf.histogram_summary("softmax weights", softmax_w)
tf.histogram_summary("softmax biases", softmax_b)
tf.scalar_summary('accuracy', accuracy)
tf.scalar_summary('cross entropy', cross_entropy)

merged = tf.merge_all_summaries()

with tf.Session() as session:
    train_writer = tf.train.SummaryWriter(summary_save_dir, session.graph)

    session.run(tf.initialize_all_variables())
    numpy_state = initial_state.eval()

    for i in xrange(ITERATIONS):
        batch = DI.next_batch()

        if i % LOG_STEP == 0:
            train_accuracy, summary = session.run([accuracy, merged], feed_dict={
                initial_state: numpy_state,
                input_sequence: batch[0],
                output_actual: batch[1]
            })
            train_writer.add_summary(summary, i)

            print "Iteration " + str(i) + " Training Accuracy " + str(train_accuracy)

        numpy_state, _ = session.run([final_state, train_step], feed_dict={
            initial_state: numpy_state,
            input_sequence: batch[0],
            output_actual: batch[1]
            })
