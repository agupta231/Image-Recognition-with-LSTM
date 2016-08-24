from image import Image
import tensorflow as tf
import numpy as np
import random
import pickle
import glob
import re
import os


class DataImport:
    def __init__(self, frames_folder, period, distances_file, threshold, batch_size, timesteps, sets_per_chunk=200, channels=1, image_size=150):
        self.frames_folder = frames_folder
        self.distances_file = distances_file
        self.period = period
        self.sets_per_chunk = sets_per_chunk
        self.time_steps = timesteps
        self.threshold = threshold
        self.batch_size = batch_size

        Image.set_parameters(channels, image_size, self.threshold)

    def import_folder(self, folder_path):
        master_data_array = []
        motor_powers = []

        # [i][0] = time (ms)
        # [i][1] = right motor power
        # [i][2] = left motor power

        with open(folder_path + "/motor_powers.txt") as log_file:
            for line in log_file:
                line_data = line.split(":")

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
            path_sections = image_path.split("/")
            int_array = []

            for string in path_sections:
                if re.search(r'\d+', string) is not None:
                    int_array.append(int(re.search(r'\d+', string).group()))

            frameTime = int_array[-1]
            distance = None

            for i in xrange(len(distance_readings)):
                if frameTime == distance_readings[i][0]:
                    distance = distance_readings[i][1]
                    break

            for i in xrange(len(motor_powers)):
                if frameTime >= motor_powers[i][0] and frameTime < motor_powers[i + 1][0]:
                    master_data_array.append(Image(frameTime, image_path, motor_powers[i][2], motor_powers[i][1], distance))
                    break

        master_data_array = sorted(master_data_array, key=lambda x: x.count)
        self._generate_chunks(master_data_array)

        del master_data_array

    def _generate_chunks(self, arg_array):
        chunk_cutoff = self.period * (self.time_steps + 1) * 1000
        chunk_size = None

        while True:
            if chunk_size is None:
                for i in xrange(len(arg_array)):
                    if arg_array[i].count > chunk_cutoff:
                        chunk_size = i + self.sets_per_chunk + 50
                        break

            if len(arg_array) / chunk_size > 1:
                chunk = open(os.getcwd() + "/chunks/chunk" + str(len(glob.glob(os.getcwd() + "/chunks/*"))), "wb")
                data = arg_array[0:chunk_size]

                pickle.dump(data, chunk)

                del arg_array[0:chunk_size]
                chunk.close()
            else:
                chunk = open(os.getcwd() + "/chunks/chunk" + str(len(glob.glob(os.getcwd() + "/chunks/*"))), "wb")
                pickle.dump(arg_array, chunk)
                chunk.close()
                return

    def next_batch(self):
        input_sequences = []
        output_sequences = []
        count = 0

        while count < self.batch_size:
            chunk = open(os.getcwd() + "/chunks/chunk" + str(random.randint(0, len(glob.glob(os.getcwd() + "/chunks/*")) - 1)))
            data = pickle.load(chunk)

            batch_end = random.randint(len(data) - self.sets_per_chunk, len(data) - 1)

            batch_times = [int(data[batch_end].count - (self.period * 1000 * i)) for i in xrange(self.time_steps + 1)]
            batch_times = list(reversed(batch_times))

            batch_images = []
            previous_key = 0
            for time in batch_times[:len(batch_times) - 1]:
                for j in xrange(previous_key, len(data) - 1):
                    if abs(data[j].count - time) <= 10:
                        batch_images.append(data[j])
                        previous_key = j + 1
                        break

            for j in xrange(previous_key, len(data) - 1):
                if data[j].count == batch_times[-1]:
                    output_sequences.append(data[j].crash_one_hot())
                    break

            if len(batch_images) == 4:
                input_sequences.append([batch_images[i].to_tensor_with_aux_info() for i in xrange(len(batch_images))])
                count += 1

        return [input_sequences, output_sequences]
