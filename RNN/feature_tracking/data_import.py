from image import Image
import tensorflow as tf
import numpy as np
import random
import pickle
import glob
import re


class DataImport:
    def __init__(self, frames_folder, chunks_folder, distances_file):
        self.frames_folder = frames_folder
        self.chunks_folder = chunks_folder
        self.distances_file = distances_file

    def import_folder(self, folder_path):
        master_data_array = []
        motor_powers = []

        # [i][0] = time (ms)
        # [i][1] = right motor power
        # [i][2] = left motor power

        with open(folder_path + "/motor_powers.txt") as log_file:
            for line in log_file:
                line_data = log_file.split(":")

                motor_powers.append([
                    float(line_data[2]),
                    int(line_data[1]),
                    int(line_data[0])
                ])

        distance_readings = []

        # [i][0] = frame number
        # [i][1] = distance

        with open(folder_path + "/" + self.distances_file) as distances:
            for line in distances:
                line_data = line.split(":")

                distance_readings.append([
                    int(line_data[0]),
                    int(line_data[1])
                ])

        files = glob.glob(folder_path + "/" + self.frames_folder + "/*")

        for image_path in files:
            path_sections = []

            for string in image_path:
                if re.search(r'\d+', string) is not None:
                    path_sections.append(int(re.search(r'\d+', string).group()))

            frameTime = path_sections[-1]
            distance = None

            for i in xrange(len(distance_readings)):
                if frameTime == distance_readings[i][0]:
                    distance = distance_readings[i][1]
                    break

            for i in xrange(len(motor_powers)):
                if frameTime >= motor_powers[i][0] and frameTime < motor_powers[i + 1][0]:
                    master_data_array.append(Image(image_path, motor_powers[i][2], motor_powers[i][1], distance))
                    break
