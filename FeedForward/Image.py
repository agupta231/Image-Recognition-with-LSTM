import tensorflow as tf


class Image:
    channels = 1
    image_width = 200
    image_height = 200

    def __init__(self, imagePath, leftMotorPower, rightMotorPower, duration, ticks):
        self.path = imagePath
        self.rightMotorPower = tf.constant([rightMotorPower])
        self.leftMotorPower = tf.constant([leftMotorPower])
        self.duration = tf.constant([duration])
        self.ticks = tf.constant([ticks])

    def to_tensor(self):
        return tf.image.decode_jpeg(tf.read_file(self.path), Image.channels)

    def to_flattened_tensor(self):
        return tf.reshape(tf.image.decode_jpeg(tf.read_file(self.path), Image.channels), [-1, Image.image_width * Image.image_height * Image.channels])