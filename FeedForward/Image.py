import tensorflow as tf
import numpy as np


class Image:
    channels = 1
    image_width = 200
    image_height = 200

    def __init__(self, imagePath, leftMotorPower, rightMotorPower, duration, ticks):
        self.path = imagePath
        self.rightMotorPower = np.array([rightMotorPower])
        self.leftMotorPower = np.array([leftMotorPower])
        self.duration = np.array([duration])
        self.ticks = np.array([ticks])

    def to_tensor(self):
        return tf.image.decode_jpeg(tf.read_file(self.path), Image.channels).eval()

    def to_flattened_tensor(self):
        return tf.reshape(self.to_tensor(), [-1]).eval()
