import tensorflow as tf
import numpy as np
from skimage.measure import compare_ssim as ssim
from skimage.feature import canny

class DataCompare:
    @staticmethod
    def ssim_compare(computed_image, actual_image):
        means = []

        for i in range(len(computed_image)):
            means.append((ssim(computed_image[i], actual_image[i]) + 1.0) / 2.0)

        return sum(means) / float(len(means))

    @staticmethod
    def edge_detection_ssim(computed_image, actual_image, sigma=0.75):
        means = []

        for i in range(len(computed_image)):
            computed_image_edges = canny(computed_image[i], sigma)
            actual_image_edges = canny(actual_image[i], sigma)

            means.append((ssim(computed_image_edges, actual_image_edges) + 1.0) / 2.0)

        return sum(means) / float(len(means))