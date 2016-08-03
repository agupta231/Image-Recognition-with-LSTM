import tensorflow as tf
from DataImport import DataImport
import os
import glob

# Config parameters
IMAGE_WIDTH = 150
IMAGE_HEIGHT = 150
IMAGE_CHANNELS = 1
PIXEL_COUNT = IMAGE_WIDTH * IMAGE_HEIGHT * IMAGE_CHANNELS
AUX_INPUTS = 2
FRAME_FREQUENCY = 60

LEARNING_RATE = 0.001
TIME_STEPS = 2

# [[Size, Layers], [Size, Layers]]
RNN_SIZE = [[128, 10]]

BATCH_SIZE = 20
ITERATIONS = 10000
LOG_STEP = 5

# Model Generation
input_raw = tf.placeholder(tf.float32, [BATCH_SIZE, TIME_STEPS, PIXEL_COUNT + AUX_INPUTS])
output_flattened = tf.placeholder(tf.float32, [BATCH_SIZE, PIXEL_COUNT])

print input_raw

cells = []
outputs = []
states = []

os.system("mkdir " + os.getcwd() + "/chunks")

DI = DataImport("resize150", os.getcwd() + "/chunks")
DI.set_image_settings(IMAGE_WIDTH, IMAGE_CHANNELS)

dataFolders = [path for path in glob.glob(os.getcwd() + "/*") if os.path.isdir(path) and not "chunks" in path and not "done" in path]
for path in dataFolders:
    print path
    DI.importFolder(path)

def cell(size, layers):
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(size)

    if layers > 1:
        lstm_cell_final = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * layers)

    else:
        lstm_cell_final = lstm_cell

    return lstm_cell_final

inputs = [tf.reshape(i, (BATCH_SIZE, PIXEL_COUNT + AUX_INPUTS)) for i in tf.split(0, TIME_STEPS, input_raw)]
print "Inputs length: " + str(len(inputs))
outputs.append(inputs)

# Data Initialization
for i in range(len(RNN_SIZE)):
    if(RNN_SIZE[i][1] != 1):
        ltsm = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.BasicLSTMCell(RNN_SIZE[i][0])] * RNN_SIZE[i][1])
    else:
        ltsm = tf.nn.rnn_cell.BasicLSTMCell(RNN_SIZE[i][0])

    cells.append(ltsm)

    if i == 0:
        print cells[0].state_size
        states.append(tf.zeros([BATCH_SIZE, cells[-1].state_size]))
    else:
        states.append(None)

    outputs.append(None)

for i in range(len(RNN_SIZE)):
    output, state = tf.nn.rnn(
        tf.nn.rnn_cell.BasicLSTMCell(RNN_SIZE[i][0]),
        outputs[i],
        initial_state=states[i])

    outputs[i + 1] = output
    states[i] = state

# Linear Activation Layer
weightFinal = tf.Variable(tf.truncated_normal([RNN_SIZE[-1][0], PIXEL_COUNT], stddev=0.1))
print weightFinal.get_shape()

biasFinal = tf.Variable(tf.constant(0.1, shape=[PIXEL_COUNT]))
print biasFinal.get_shape()

print outputs[1]

flattened = tf.nn.relu(tf.matmul(outputs[-1][-1], weightFinal) + biasFinal)

# Normalize the data
flattened_normal = tf.div(tf.sub(flattened, tf.reduce_min(flattened)), tf.sub(tf.reduce_max(flattened), tf.reduce_min(flattened))) * 255

print flattened_normal

denom_tensor = tf.fill([PIXEL_COUNT * IMAGE_CHANNELS], 255.0)

#cross_entropy = tf.reduce_mean(-tf.reduce_sum(output_flattened * tf.log(flattened_normal), reduction_indices=[1]))
cross_entropy = tf.reduce_mean(-tf.reduce_sum(output_flattened * tf.log(tf.clip_by_value(flattened_normal, 1e-10, 1.0)), reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(cross_entropy)

accuracy = tf.reduce_mean(tf.abs(tf.div(tf.sub(output_flattened, flattened_normal), denom_tensor)))

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())

    for i in range(ITERATIONS):
        batch = DI.next_batch(BATCH_SIZE, TIME_STEPS)

        if i % LOG_STEP == 0:
            train_accuary = accuracy.eval(feed_dict={
                input_raw: batch[0],
                output_flattened: batch[1]
            })

            print "Step " + str(i) + " Error " + str(train_accuary)

        train_step.run(feed_dict={
            input_raw: batch[0],
            output_flattened: batch[1]
        })
