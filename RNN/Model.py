import tensorflow as tf

# Config parameters
IMAGE_WIDTH = 150
IMAGE_HEIGHT = 150
IMAGE_CHANNELS = 1
PIXEL_COUNT = IMAGE_WIDTH * IMAGE_HEIGHT * IMAGE_HEIGHT
AUX_INPUTS = 2
FRAME_FREQUENCY = 60

LEARNING_RATE = 0.001
TIME_STEPS = 5

# [[Size, Layers], [Size, Layers]]
RNN_SIZE = [[10, 32]]

BATCH_SIZE = 25
ITERATIONS = 10000
LOG_STEP = 50

# Model Generation
input_flattened = tf.placeholder(tf.int16, [None, TIME_STEPS, PIXEL_COUNT + AUX_INPUTS])
cells = []
outputs = []
states = []


def cell(size, layers):
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(size)

    if layers > 1:
        lstm_cell_final = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * layers)

    else:
        lstm_cell_final = lstm_cell

    return lstm_cell_final


# Data Initialization
for i in range(len(RNN_SIZE)):
    if(RNN_SIZE[i][1] != 1):
        ltsm = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.BasicLSTMCell(RNN_SIZE[i][0])] * RNN_SIZE[i][1])
    else:
        ltsm = tf.nn.rnn_cell.BasicLSTMCell(RNN_SIZE[i][0])

    cells.append(ltsm)

    if i == 0:
        states.append(tf.zeros([BATCH_SIZE, cells[-1].state_size]))

#outputs.append(input)

for i in range(len(RNN_SIZE)):
    output = tf.nn.rnn(cells[i], outputs[i], initial_state=states[i])
    outputs[i + 1] = output
    states[i] = output

# Linear Activation Layer
weightFinal = tf.Variable(tf.truncated_normal([RNN_SIZE[-1][0], PIXEL_COUNT], stddev=0.1))
biasFinal = tf.Variable(tf.constant(0.1, shape=[PIXEL_COUNT]))

flattened = tf.nn.relu(tf.matmul(outputs[-1], weightFinal) + biasFinal)

# Normalize the data
flattened_normal = tf.div(tf.sub(flattened, tf.reduce_min(flattened)), tf.sub(tf.reduce_max(flattened), tf.reduce_min(flattened))) * 255

denom_tensor = tf.fill([PIXEL_COUNT * IMAGE_CHANNELS], 255.0)

cross_entropy = tf.reduce_mean(-tf.reduce_sum(output_flattened * tf.log(flattened_normal), reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(cross_entropy)

accuracy = tf.constant(1.0) - tf.reduce_mean(tf.abs(tf.div(tf.sub(output_flattened, flattened_normal), denom_tensor)))

with tf.Session as sess:
    sess.run(tf.initialize_all_variables())

    for i in range(ITERATIONS):

        train_step.run(feed_dict={

        })
