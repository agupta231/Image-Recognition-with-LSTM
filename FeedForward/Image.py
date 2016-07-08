import tensorflow as tf

class Image:
    def __init__(self, imagePath, rightMotorPower, leftMotorPower, duration):
        self.path = imagePath
        self.rightMotorPower = tf.constant(rightMotorPower)
        self.leftMotorPower = tf.constant(leftMotorPower)
        self.duration = tf.constant(duration)