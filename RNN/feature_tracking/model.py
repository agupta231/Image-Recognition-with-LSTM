from data_import import DataImport
import tensorflow as tf
import numpy as np
import threading
import glob
import os

# Setup seed
np.random.seed(0)

# Setup parameters
IMAGE_HEIGHT = 150
IMAGE_WIDTH = 150
IMAGE_CHANNELS = 1
PIXEL_COUNT = IMAGE_HEIGHT * IMAGE_WIDTH * IMAGE_CHANNELS
AUX_INPUTS = 2
FREQUENCY = 60

FRAMES_FOLDER = "resize150"
DISTANCE_DATA = "distances_sigma_2.25_1.5.txt"
THRESHOLD = 60

LEARNING_RATE = 0.001
SEQUENCE_SPACING = 0.256 # In seconds
TIME_STEPS = 4
BATCH_SIZE = 22
LOG_STEP = 10
ROC_COLLECT = 15
ITERATIONS = 5000000

CELL_SIZE = 348
CELL_LAYERS = 64
HIDDEN_SIZE = 11251
DEEP_CON_1 = 128
DEEP_CON_2 = 64
DEEP_CON_3 = 32
DEEP_CON_4 = 16
SOFTMAX_SIZE = 256
OUTPUT_SIZE = 2

BATCH_REDUCE_ITERATION = 50
BATCH_REDUCE_STEP = 4
ACCURACY_CACHE_SIZE = 5
STOPPING_THRESHOLD = 0

REGENERATE_CHUNKS = True

# Generate folder for tensorboard summary files
summary_save_dir = os.getcwd() + "/summaries/" + FRAMES_FOLDER + "_" + DISTANCE_DATA + "_lr" + str(LEARNING_RATE) + "_t" + str(THRESHOLD) + "_bs" + str(BATCH_SIZE) + "_ts" + str(TIME_STEPS) + "_p" + str(SEQUENCE_SPACING) + "_cs" + str(CELL_SIZE) + "x" + str(CELL_LAYERS) + "x" + str(HIDDEN_SIZE)
os.mkdir(summary_save_dir)

# Create data importer object
DI = DataImport(FRAMES_FOLDER, SEQUENCE_SPACING, DISTANCE_DATA, THRESHOLD, BATCH_SIZE, TIME_STEPS, channels=IMAGE_CHANNELS, image_size=IMAGE_WIDTH)

# Generate chunks for more efficient loaded
if REGENERATE_CHUNKS:
    os.mkdir(os.getcwd() + "/chunks")

    dataFolders = [path for path in glob.glob(os.getcwd() + "/*") if
                   os.path.isdir(path) and not "chunks" in path and not "summaries" in path]
    for path in dataFolders:
        DI.import_folder(path)


# Helper functions
def load_batch(sess, coord, op):
    batch_count = 0
    batch_size = BATCH_SIZE

    while not coord.should_stop():
        # if batch_count % BATCH_REDUCE_ITERATION == 0 and batch_size >= BATCH_REDUCE_STEP + 1:
        #     batch_size -= BATCH_REDUCE_STEP

        batch = DI.next_batch()

        sess.run(op, feed_dict={queue_input: batch[0], queue_output: batch[1]})


def weight_variable(shape):
    return tf.Variable(tf.truncated_normal(shape=shape, stddev=0.1))


def bias_variable(shape):
    return tf.Variable(tf.constant(0.1, shape=shape))

# The actual model

# Create queue for mulithreaded batch loaded
queue_input = tf.placeholder(tf.float32, [BATCH_SIZE, TIME_STEPS, PIXEL_COUNT + AUX_INPUTS])
queue_output = tf.placeholder(tf.float32, [BATCH_SIZE, OUTPUT_SIZE])
queue = tf.RandomShuffleQueue(250, 2, [tf.float32, tf.float32], shapes=[[BATCH_SIZE, TIME_STEPS, PIXEL_COUNT + AUX_INPUTS], [BATCH_SIZE, OUTPUT_SIZE]])
# queue = tf.FIFOQueue(250, 2, [tf.float32, tf.float32], shapes=[[BATCH_SIZE, TIME_STEPS, PIXEL_COUNT + AUX_INPUTS], [BATCH_SIZE, OUTPUT_SIZE]])

queue_op = queue.enqueue([queue_input, queue_output])

# input_sequence = tf.placeholder(tf.float32, [BATCH_SIZE, TIME_STEPS, PIXEL_COUNT + AUX_INPUTS])
# output_actual = tf.placeholder(tf.float32, [BATCH_SIZE, OUTPUT_SIZE])


# Create placeholders
raw = queue.dequeue()
input_sequence = raw[0]
output_actual = raw[1]
dropout = tf.placeholder(tf.float32)

# This is the actual LSTM cell
lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(CELL_SIZE, state_is_tuple=False)

# Create dropout to prevent dead neurons
dropout_lstm = tf.nn.rnn_cell.DropoutWrapper(lstm_cell, output_keep_prob=dropout)

# Layer the LSTM to give it more processing power
stacked_lstm = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * CELL_LAYERS, state_is_tuple=False)

# initial_state = state = stacked_lstm.zero_state(BATCH_SIZE, tf.float32)
# outputs = []

print input_sequence
print input_sequence[:, 0, :]

# Get the output of the cell
output, _ = tf.nn.dynamic_rnn(stacked_lstm, input_sequence, dtype=tf.float32)

# Transposed the output to get the last output in the sequence (Tensorflow can't do list[-1])
output_transposed = tf.transpose(output, [1, 0, 2])

# Get the final output of the LSTM
last = tf.gather(output_transposed, int(output_transposed.get_shape()[0]) - 1)

print last

# with tf.variable_scope("LSTM"):
#     for step in xrange(TIME_STEPS):
#         if step > 0:
#             tf.get_variable_scope().reuse_variables()
#
#         input_pre = tf.reshape(input_sequence[:, step, :], [-1, HIDDEN_SIZE])
#
#         input_weight = tf.get_variable("lstm_w_" + str(step), [HIDDEN_SIZE, CELL_SIZE], dtype=tf.float32)
#         input_bias = tf.get_variable("lstm_b" + str(step), [CELL_SIZE], dtype=tf.float32)
#
#         # input_post = tf.matmul(input_sequence[:, step, :], input_weight) + input_bias
#         input_post = tf.matmul(input_pre, input_weight) + input_bias
#
#         # cell_output, state = stacked_lstm(input_sequence[:, step, :], state)
#         cell_output, state = stacked_lstm(input_post, state)
#         outputs.append(cell_output)

# final_state = state

# output = tf.reshape(tf.concat(1, outputs), [-1, HIDDEN_SIZE])
#output = tf.reshape(outputs[-1], [BATCH_SIZE, SOFTMAX_SIZE])

# softmax_w = tf.get_variable("softmax_w", [HIDDEN_SIZE, OUTPUT_SIZE], dtype=tf.float32)
# softmax_b = tf.get_variable("softmax_b", [OUTPUT_SIZE], dtype=tf.float32)

dc1_w = weight_variable([CELL_SIZE, DEEP_CON_1])
dc1_b = bias_variable([DEEP_CON_1])
dc1_out = tf.nn.relu(tf.matmul(last, dc1_w) + dc1_b)

dc2_w = weight_variable([DEEP_CON_1, DEEP_CON_2])
dc2_b = bias_variable([DEEP_CON_2])
dc2_out = tf.nn.relu(tf.matmul(dc1_out, dc2_w) + dc2_b)

dc3_w = weight_variable([DEEP_CON_2, DEEP_CON_3])
dc3_b = bias_variable([DEEP_CON_3])
dc3_out = tf.nn.relu(tf.matmul(dc2_out, dc3_w) + dc3_b)

dc4_w = weight_variable([DEEP_CON_3, DEEP_CON_4])
dc4_b = bias_variable([DEEP_CON_4])
dc4_out = tf.nn.relu(tf.matmul(dc3_out, dc4_w) + dc4_b)

final_dropout = tf.nn.dropout(dc4_out, keep_prob=dropout)

# Create softmax Layer
softmax_w = weight_variable([DEEP_CON_4, OUTPUT_SIZE])
softmax_b = bias_variable([OUTPUT_SIZE])
# prediction = tf.nn.softmax(tf.matmul(output, softmax_w) + softmax_b)
prediction = tf.nn.softmax(tf.matmul(final_dropout, softmax_w) + softmax_b)

# cross_entropy = tf.reduce_mean(-tf.reduce_sum(output_actual * tf.log(prediction), reduction_indices=[1]))
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(prediction, output_actual))
# train_step = tf.train.RMSPropOptimizer(LEARNING_RATE).minimize(cross_entropy)
train_step = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(cross_entropy)

# Compare the value that has the largest probablity in the prediction and compare that to the actual answer
correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(output_actual, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Write summaries for tensorboard and accuary tracking

tf.histogram_summary("Densly Connected Layer 1 weights", dc1_w)
tf.histogram_summary("Densly Connected Layer 1 biases", dc1_b)
tf.histogram_summary("Densly Connected Layer 2 weights", dc2_w)
tf.histogram_summary("Densly Connected Layer 2 biases", dc2_b)
tf.histogram_summary("Densly Connected Layer 3 weights", dc3_w)
tf.histogram_summary("Densly Connected Layer 3 biases", dc3_b)
tf.histogram_summary("Densly Connected Layer 4 weights", dc4_w)
tf.histogram_summary("Densly Connected Layer 4 biases", dc4_b)
tf.histogram_summary("softmax weights", softmax_w)
tf.histogram_summary("softmax biases", softmax_b)
tf.scalar_summary('accuracy', accuracy)
tf.scalar_summary('cross entropy', cross_entropy)

merged = tf.merge_all_summaries()

with tf.Session() as session:
    # Create object to write session summaries
    train_writer = tf.train.SummaryWriter(summary_save_dir, session.graph)

    # Create coordinator to be able manage mulitple queues
    coordinator = tf.train.Coordinator()

    session.run(tf.initialize_all_variables())
    # numpy_state = initial_state.eval()

    t = threading.Thread(target=load_batch, args=(session, coordinator, queue_op))
    t.start()

    # Create value to cache previous accuracies
    previous_accuracies = [0] * ACCURACY_CACHE_SIZE

    for i in xrange(ITERATIONS):
        if i % LOG_STEP == 0:
            # train_accuracy, summary = session.run([accuracy, merged], feed_dict={initial_state: numpy_state})
            train_accuracy, summary = session.run([accuracy, merged], feed_dict={dropout: 0.5})

            for j in xrange(1, len(previous_accuracies) - 1):
                previous_accuracies[j - 1] = previous_accuracies[j]

            previous_accuracies[-1] = train_accuracy

            if i > (ACCURACY_CACHE_SIZE + 1) * LOG_STEP:
                combined_delta = 0

                for j in xrange(1, len(previous_accuracies) - 1):
                    delta = previous_accuracies[j] - previous_accuracies[j - 1]
                    combined_delta += abs(delta)

                if combined_delta / ACCURACY_CACHE_SIZE < STOPPING_THRESHOLD:
                    print "killed"
                    exit()

            train_writer.add_summary(summary, i)

            print "Iteration " + str(i) + " Training Accuracy " + str(train_accuracy)

        # Determine ROC of current iteration
        if i % ROC_COLLECT == 0:
            ROC_log_file = open(summary_save_dir + "/ROC.txt", "a")

            # Get the actual output values, the predicted values and the softmax values (just for debugging)
            output_values, prediction_values, outputs_raw = session.run([output_actual, prediction, last], feed_dict={dropout: 0.5})

            print outputs_raw
            print output_values
            print prediction_values

            actual_positives = 0
            actual_negatives = 0
            true_positives = 0
            true_negatives = 0
            false_positives = 0
            false_negatives = 0

            # Count the true positive and the true negatives
            for j in xrange(len(output_values)):
                if output_values[j][1] == 1:
                    actual_positives += 1

                    if prediction_values[j][1] > 0.5:
                        true_positives += 1

                elif output_values[j][0] == 1:
                    actual_negatives += 1

                    if prediction_values[j][0] > 0.5:
                        true_negatives += 1

            # Prevent 0 / 0 error
            if actual_positives == 0:
                if true_positives == 0:
                    TPR = 1
                else:
                    TPR = 0
            else:
                TPR = true_positives / float(actual_positives)

            if actual_negatives == 0:
                if true_negatives == 0:
                    TNR = 1
                else:
                    TNR = 0
            else:
                TNR = true_negatives / float(actual_negatives)

            # Calculate false positive rate and the false negative rate by subtracting the values from 1
            FPR = 1 - TPR
            FNR = 1 - TNR

            print TPR, FPR, true_positives, actual_positives, true_negatives, actual_negatives

            # Write data to log step
            ROC_log_file.write(str(i) + "," + str(TPR) + "," + str(FPR) + "\n")
            ROC_log_file.close()

        # numpy_state, _ = session.run([final_state, train_step], feed_dict={initial_state: numpy_state})

        # Run the training step optimizer
        session.run(train_step, feed_dict={dropout: 0.5})

    coordinator.request_stop()
    coordinator.join([t])
