import tensorflow as tf
import numpy as np


class Image:
    channels = 1
    image_size = 150
    threshold = 30

    def __init__(self, image_count, image_path, left_motor_power, right_motor_power, distance):
        self.count = image_count
        self.image_path = image_path
        self.left_motor_power = left_motor_power
        self.right_motor_power = right_motor_power
        self.distance = distance

    def to_tensor_with_aux_info(self):
        return tf.concat(0, [self._to_flattened_tensor(), np.array(self.left_motor_power), np.array(self.right_motor_power)]).eval()

    def crash_one_hot(self):
        # One hot vector format:
        # 0 - Not a crash
        # 1 - Is a crash
        # Huh! not that bad...

        if self.distance <= Image.threshold:
            return np.array([0, 1])
        else:
            return np.array([1, 0])

    def _to_tensor(self):
        tensor = tf.image.decode_jpeg(tf.read_file(self.image_path), channels=Image.channels).eval()
        tensor_flattened = tf.reshape(tensor, [-1])

        return tensor_flattened

    def _to_flattened_tensor(self):
        return tf.reshape(self._to_tensor(), [-1])

    @staticmethod
    def set_parameters(channels, image_size, threshold):
        Image.channels = channels
        Image.image_size = image_size
        Image.threshold = threshold