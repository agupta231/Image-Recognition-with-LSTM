import matplotlib.pyplot as plot
import tensorflow as tf
import numpy as np
from skimage.measure import compare_ssim as ssim
from skimage.feature import canny

class DataCompare:
    def __init__(self):
        plot.ion()

        self.figure = plot.figure("Images")
        self.plots = []
        self.plot_names = ["Original", "Computed", "Heat Signature"]

        self.generate_plots()
    def ssim_compare(self, computed_image, actual_image):
        return (1.0 - ssim(computed_image, actual_image) + 1.0) / 2.0

    def edge_detection_ssim(self, computed_image, actual_image, sigma=0.75):
        computed_image_edges = canny(computed_image, sigma)
        actual_image_edges = canny(actual_image, sigma)

        return (1.0 - (ssim(computed_image_edges, actual_image_edges) + 1.0) / 2.0)

    def mutli_accurary_compare(self, computed_image, actual_image, sigma=0.75):
            edge_detection_ssim_means = []
            pixel_by_pixel_means = []

            for i in xrange(len(computed_image)):
                edge_detection_ssim_means.append(self.ssim_compare(computed_image[i], actual_image[i]))
                pixel_by_pixel_means.append(self.pixel_by_pixel_compare(computed_image[i], actual_image[i]))

                if i == (len(computed_image) - 1):
                    for image in self.plots:
                        image.clear()

                    self.plots[0].imshow(actual_image[i], cmap=plot.cm.gray)
                    self.plots[0].set_title(self.plot_names[0])

                    self.plots[1].imshow(computed_image[i], cmap=plot.cm.gray)
                    self.plots[1].set_title(self.plot_names[1])

                    self.plots[2].imshow(self.generate_heat_plot(actual_image[i], computed_image[i]), plot.cm.hot)
                    self.plots[2].set_title(self.plot_names[2])

                    plot.pause(0.001)

            edge_detection_ssim_final = sum(edge_detection_ssim_means) / float(len(edge_detection_ssim_means))
            pixel_by_pixel_final = sum(pixel_by_pixel_means) / float(len(pixel_by_pixel_means))

            return edge_detection_ssim_final, pixel_by_pixel_final

    
    def generate_plots(self):
        for i in xrange(len(self.plot_names)):
            self.plots.append(self.figure.add_subplot(2, 2, i + 1))
            self.plots[-1].set_title(self.plot_names)

    def generate_heat_plot(self, actual_image, computed_image):
        actual_image_numpy = np.array(actual_image)
        computed_image_numpy = np.array(computed_image)

        return np.absolute(np.divide(np.subtract(actual_image_numpy, computed_image_numpy), 255.0)).tolist()

    def pixel_by_pixel_compare(self, actual_image, computed_image):
        actual_image_numpy = np.array(actual_image)
        computed_image_numpy = np.array(computed_image)

        return np.average(np.absolute(np.divide(np.subtract(actual_image_numpy, computed_image_numpy), 255.0)))
