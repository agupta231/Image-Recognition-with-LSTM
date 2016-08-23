import tensorflow as tf
import numpy as np
import glob
import random
import pickle
import re

class Image:
    channels = 1
    imageSize = 150

    def __init__(self, imagePath, leftMotorPower, rightMotorPower, frame_time):
        self.imagePath = imagePath
        self.leftMotorPower = leftMotorPower
        self.rightMotorPower = rightMotorPower
        self.frameTime = frame_time

    def to_tensor(self):
        tensor = tf.image.decode_jpeg(tf.read_file(self.imagePath), Image.channels).eval()
        tensor_flattened = tf.reshape(tensor, [-1])

        return tensor_flattened.eval()

    def to_tensor_with_aux_info(self):
        return tf.concat(0, [tf.to_float(self.__to_flattened_tensor()), np.array([self.leftMotorPower]), np.array([self.rightMotorPower])]).eval()

    def __to_flattened_tensor(self):
        return tf.reshape(self.to_tensor(), [-1])

    def to_flattened_tensor_numpy(self):
        return self.__to_flattened_tensor().eval()

    def to_2d_tensor(self):
        return tf.squeeze(tf.to_float(tf.image.decode_jpeg(tf.read_file(self.imagePath), Image.channels))).eval()

class DataImport:
    def __init__(self, framesFolder, chunksFolder):
        self.framesFolder = framesFolder
        self.chunksFolder = chunksFolder

    def _save_chunk(self, array):
        CHUNK_SIZE = 100

        while (len(array) > CHUNK_SIZE + 10):
            chunk = open(self.chunksFolder + "/chunk" + str(len(glob.glob(self.chunksFolder + "/*"))), "wb")
            data = array[0: CHUNK_SIZE]

            pickle.dump(data, chunk)

            del array[0: CHUNK_SIZE]
            chunk.close()

        chunk = open(self.chunksFolder + "/chunk" + str(len(glob.glob(self.chunksFolder + "/*"))), "wb")
        pickle.dump(array, chunk)

        chunk.close()

    def importFolder(self, folderPath):
        motorPowerArray = []

        # time(MS) = motorPowerArray[i][0]
        # rightMotorPower = motorPowerArray[i][1]
        # leftMotorPower = motorPowerArray[i][2]

        with open(folderPath + "/LogFile.txt") as logFile:
            for line in logFile:
                lineData = line.split(":")

                # leftMotorPower = lineData[0]
                # rightMotorPower = lineData[1]
                # time(MS) = lineData[2]

                motorPowerArray.append([
                    float(lineData[2]),
                    int(lineData[1]),
                    int(lineData[0])
                ])

        frontImages = glob.glob(folderPath + "/front/" + self.framesFolder + "/*")
        frontDataArray = []

        for imagePath in frontImages:
            timeValues = []
            pathSections = imagePath.split("/")

            for string in pathSections:
                if re.search(r'\d+', string) is not None:
                    timeValues.append(int(re.search(r'\d+', string).group()))

            frameTime = timeValues[-1]

            for i in xrange(len(motorPowerArray) - 1):
                if frameTime >= motorPowerArray[i][0] and frameTime < motorPowerArray[i + 1][0]:
                    frontDataArray.append(
                        Image(
                            imagePath,
                            motorPowerArray[i][1],
                            motorPowerArray[i][0],
                            frameTime
                        )
                    )

        frontDataArray = sorted(frontDataArray, key=lambda x: x.frameTime)
        self._save_chunk(frontDataArray)

        del frontDataArray

        backImages = glob.glob(folderPath + "/back/" + self.framesFolder + "/*")
        backDataArray = []

        for imagePath in backImages:
            timeValues = []
            pathSections = imagePath.split("/")

            for string in pathSections:
                if re.search(r'\d+', string) is not None:
                    timeValues.append(int(re.search(r'\d+', string).group()))

            frameTime = timeValues[-1]

            for i in xrange(len(motorPowerArray) - 1):
                if frameTime >= motorPowerArray[i][0] and frameTime < motorPowerArray[i + 1][0]:
                    backDataArray.append(
                        Image(
                            imagePath,
                            -1 * motorPowerArray[i][0],
                            -1 * motorPowerArray[i][1],
                            frameTime
                        )
                    )

        backDataArray = sorted(backDataArray, key=lambda x: x.frameTime)
        self._save_chunk(backDataArray)

        del backDataArray

    def next_batch(self, size, timesteps):
        input_images = []
        output_images_flattened = []
        output_images_2d = []

        for i in range(size):
            chunk = open(self.chunksFolder + "/chunk" + str(random.randint(0, len(glob.glob(self.chunksFolder + "/*")) - 1)))
            data = pickle.load(chunk)

            batchStart = random.randint(0, len(data) - timesteps - 2)
            batch = data[batchStart:batchStart + timesteps + 1]

            steps = []

            for j in range(timesteps):
                steps.append(batch[j].to_tensor_with_aux_info())

            input_images.append(steps)
            output_images_flattened.append(batch[timesteps].to_flattened_tensor_numpy())
            output_images_2d.append(batch[timesteps].to_2d_tensor())

        resultant_batch = [input_images, output_images_flattened, output_images_2d]
        return resultant_batch

    def set_image_settings(self, imageSize, channels):
        Image.channels = channels
        Image.imageSize = imageSize
