from frame import Frame
import tensorflow as tf
import numpy as np
import random
import pickle
import glob
import re
import os

# Object to handle the data importing of files and the batch processing
class DataImport:
    def __init__(self, frames_folder, period, distances_file, threshold, batch_size, timesteps, sets_per_chunk=200, channels=1, image_size=150):
        # Set parameters for the object

        self.frames_folder = frames_folder
        self.distances_file = distances_file
        self.period = period
        self.sets_per_chunk = sets_per_chunk
        self.time_steps = timesteps
        self.threshold = threshold
        self.batch_size = batch_size

        Frame.set_parameters(channels, image_size, self.threshold)

    # Method to import the folders of data and convert that data into chunks to prevent the program from filling up RAM
    def import_folder(self, folder_path):
        # Array to hold all of the data
        master_data_array = []

        # Array of the motor powers as well as the time associated with them. This allows for matching each frame to its
        # corresponding motor power
        motor_powers = []

        # [i][0] = time (ms)
        # [i][1] = right motor power
        # [i][2] = left motor power

        # Import the motor powers
        with open(folder_path + "/motor_powers.txt") as log_file:
            for line in log_file:
                # Split the raw data from the raspberry pi into motors powers and times
                line_data = line.split(":")

                # Populate the motor power array
                motor_powers.append([
                    float(line_data[2]),
                    int(line_data[1]),
                    int(line_data[0])
                ])

        # Similar process for the distance readings. This array stores the time as well as the distance for each frame
        distance_readings = []

        # [i][0] = frame number
        # [i][1] = distance

        with open(folder_path + "/" + self.distances_file) as distances:
            for line in distances:
                # Ditto for the motor power file reading
                line_data = line.split(":")

                # Populate the distance readings array
                distance_readings.append([
                    int(line_data[0]),
                    int(line_data[1])
                ])

        # Create an array of all of the frames in the folder
        files = glob.glob(folder_path + "/" + self.frames_folder + "/*")

        for image_path in files:
            # These next couple of lines split up the file path and determine the time associated with each frame
            path_sections = image_path.split("/")
            int_array = []

            for string in path_sections:
                if re.search(r'\d+', string) is not None:
                    int_array.append(int(re.search(r'\d+', string).group()))

            frameTime = int_array[-1]
            distance = None

            # Match each frame to its corresponding distance reading
            for i in xrange(len(distance_readings)):
                if frameTime == distance_readings[i][0]:
                    distance = distance_readings[i][1]
                    break

            # Match each frame to its corresponding motor power and append the frame to the master data array
            for i in xrange(len(motor_powers)):
                if frameTime >= motor_powers[i][0] and frameTime < motor_powers[i + 1][0]:
                    master_data_array.append(Frame(frameTime, image_path, motor_powers[i][2], motor_powers[i][1], distance))
                    break

        # Sort the master data array based of off the frame times in ascending order
        master_data_array = sorted(master_data_array, key=lambda x: x.count)

        # Write the data to chunks to prevent filling up RAM
        self._generate_chunks(master_data_array)

        # Delete master array... not really necessary, but it can't hurt
        del master_data_array

    # Method to write all of the data to storable chunks (using pickle)
    def _generate_chunks(self, arg_array):
        # Determine the how many seconds of data each chunk should store. A little extra is added to allow for some leeway
        chunk_cutoff = self.period * (self.time_steps + 5) * 1000
        chunk_size = None

        while True:
            # If the number of files per chunk hasn't been set yet (as the time is different from number of files), then
            # figure it out and set the variable to that value
            if chunk_size is None:
                for i in xrange(len(arg_array)):
                    if arg_array[i].count > chunk_cutoff:
                        chunk_size = i + self.sets_per_chunk + 50
                        break

            # Make sure that there is enough left over data to populate other chunks. It should be noted that it
            # the expression below is the same thing as len(ary_array) / chunk_size >= 2, as integer math in python
            # always rounds down
            if len(arg_array) / chunk_size > 1:
                # Determine the number of current chunks and create a new chunk with a differnet file name
                chunk = open(os.getcwd() + "/chunks/chunk" + str(len(glob.glob(os.getcwd() + "/chunks/*"))), "wb")

                # Split the array to a smaller bit to save into the chunk
                data = arg_array[0:chunk_size]

                # Save the data into the chunk
                pickle.dump(data, chunk)

                # Delete the saved data so that the loop can continue to work
                del arg_array[0:chunk_size]

                # Close and write the file buffer
                chunk.close()

            # If the size of the array isn't big enough, just ave the whole array to the chunk. The code is the same
            # as the previous code
            else:
                chunk = open(os.getcwd() + "/chunks/chunk" + str(len(glob.glob(os.getcwd() + "/chunks/*"))), "wb")
                pickle.dump(arg_array, chunk)
                chunk.close()
                return

    # Method for batch feeding
    def next_batch(self, bs=-1):
        # Allows for variable batch size if in the future, variable batch sizes are desired
        if bs == -1:
            bs = self.batch_size

        # Create the placholders for the input as well as the output batches
        input_sequences = []
        output_sequences = []

        # Run loop while the number of items in the input sequence is less than the batch size
        while len(input_sequences) < bs:
            # Randomly open a chunk and load data
            chunk = open(os.getcwd() + "/chunks/chunk" + str(random.randint(0, len(glob.glob(os.getcwd() + "/chunks/*")) - 1)))
            data = pickle.load(chunk)

            # For the sequence of images, determine the last frame. This allows for the program to count backwards and
            # make sure that there is no "index out of bounds" error when trying to retrieve data from a chunk.
            batch_end = random.randint(len(data) - self.sets_per_chunk, len(data) - 1)

            # Calculate the time for each step in the sequence
            batch_times = [int(data[batch_end].count - (self.period * 1000 * i)) for i in xrange(self.time_steps + 1)]

            # Reverse the times list to be chronological
            batch_times = list(reversed(batch_times))

            # Place holders for the images in the current input
            batch_images = []

            # Create a key placeholder that gets updated every time a value is added to the batch array. This makes sure
            # that the program doesn't traverse through unnecessary files.
            previous_key = 0

            # Go through every value in the batch times set except for the last one because that is the sample output
            # that is going to be fed into the network
            for time in batch_times[:len(batch_times) - 1]:
                for j in xrange(previous_key, len(data) - 1):
                    # So when calculating the batch times above, there is a chance that there the calculated times (above)
                    # don't match up with the actual time on the files due to a rounding error. Instead the
                    # program just chooses the file that's time is the closest to the calculated time.
                    if abs(data[j].count - time) <= 10:
                        # Append the image to the batch images
                        batch_images.append(data[j])

                        # Update the previous key
                        previous_key = j + 1
                        break

            # Check if all of the inputs have been completed
            if len(batch_images) == self.time_steps:
                output_found = False

                # Do the same thing as before, instead just getting the output, not the input
                for j in xrange(previous_key, len(data) - 1):
                    if data[j].count == batch_times[-1]:
                        output_found = True
                        output_sequences.append(data[j].crash_one_hot())
                        break

                # Assert that the output was found
                if output_found:
                    # Convert all of the data to numpy arrays and append the values to the master input and output
                    # sequences array
                    input_sequences.append([batch_images[i].to_tensor_with_aux_info() for i in xrange(len(batch_images))])

        # Return the mini batch
        return [input_sequences, output_sequences]
