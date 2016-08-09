import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plot
import matplotlib.animation as animation
from skimage.measure import compare_ssim as ssim
from skimage.feature import canny

class DataCompare:
    def __init__(self):
        plot.ion()

        self.figure = plot.figure("Images")
        self.plots = []

        self.generate_plots(["Original", "Computed", "Heat Signature"])

    def ssim_compare(self, computed_image, actual_image):
        return (1.0 - ssim(computed_image, actual_image) + 1.0) / 2.0

    def edge_detection_ssim(self, computed_image, actual_image, sigma=0.75):
        computed_image_edges = canny(computed_image, sigma)
        actual_image_edges = canny(actual_image, sigma)

        return (1.0 - ssim(computed_image_edges, actual_image_edges) + 1.0) / 2.0

    def mutli_accurary_compare(self, computed_image, actual_image, sigma=0.75):
            edge_detection_ssim_means = []

            for i in xrange(len(computed_image)):
                edge_detection_ssim_means.append(self.edge_detection_ssim(computed_image[i], actual_image[i], sigma=sigma))

                if i == (len(computed_image) - 1):
                    plot_animate = animation.FuncAnimation(self.figure, self.animatePlot(computed_image[i], actual_image[i]))
                    plot.show()

            edge_detection_ssim_final = sum(edge_detection_ssim_means) / float(len(edge_detection_ssim_means))

            return edge_detection_ssim_final

    
    def generate_plots(self, plots):
        for i in xrange(len(plots)):
            self.plots.append(self.figure.add_subplot(2, 2, i + 1))
            self.plots[-1].set_title(plots[i])
            self.plots[-1].axis("off")
            
    def generate_heat_plot(self, actual_image, computed_image):
        actual_image_numpy = np.array(actual_image)
        computed_image_numpy = np.array(computed_image)

        return np.absolute(np.divide(np.subtract(actual_image_numpy, computed_image_numpy), 255.0)).astype(int).tolist()

    def animatePlot(self, computed_image, actual_image):
        for image in self.plots:
            image.clear()

        self.plots[0].imshow(actual_image, cmap=plot.cm.gray)
        self.plots[1].imshow(computed_image, cmap=plot.cm.gray)
        self.plots[2].imshow(self.generate_heat_plot(actual_image, computed_image), plot.cm.hot)