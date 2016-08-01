import tensorflow as tf
import numpy as np
import glob
import random
import os
import pickle
import re

class Image:
    channels = 1
    imageSize = 150

    def __init__(self, imagePath, leftMotorPower, rightMotorPower):
        self.imagePath = imagePath
        self.leftMotorPower = leftMotorPower
        self.rightMotorPower = rightMotorPower

    def to_tensor(self):
        tensor = tf.image.decode_jpeg(tf.read_file(self.imagePath), Image.channels).eval()
        return tf.reshape(tensor, [-1]).eval()

class DataImport:
    def __init__(self, framesFolder, chunksFolder):
        self.framesFolder = framesFolder
        self.chunksFolder = chunksFolder

    def importFolder(self, folderPath):
        motorPowerArray = []

        # time(MS) = motorPowerArray[i][0]
        # rightMotorPower = motorPowerArray[i][0]
        # leftMotorPower = motorPowerArray[i][1]

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
            frameTime = int(re.search(r'\d+', imagePath).group())

            for i in xrange(len(motorPowerArray) - 1):
                if frameTime >= motorPowerArray[i][0] and frameTime < motorPowerArray[i + 1][0]:
                    frontDataArray.append(
                        Image(
                            imagePath,
                            motorPowerArray[i][1],
                            motorPowerArray[i][0],
                        )
                    )

        chunk = open(self.chunksFolder + "/chunk" + str(len(glob.glob(self.chunksFolder + "/*"))), "wb")
        pickle.dump(frontDataArray, chunk)

        chunk.close()
        del frontDataArray

        backImages = glob.glob(folderPath + "/back/" + self.framesFolder + "/*")
        backDataArray = []

        for imagePath in backImages:
            frameTime = int(re.search(r'\d+', imagePath).group())

            for i in xrange(len(motorPowerArray) - 1):
                if frameTime >= motorPowerArray[i][0] and frameTime < motorPowerArray[i + 1][0]:
                    backDataArray.append(
                        Image(
                            imagePath,
                            -1 * motorPowerArray[i][0],
                            -1 * motorPowerArray[i][1],
                        )
                    )

        chunk = open(self.chunksFolder + "/chunk" + str(len(glob.glob(self.chunksFolder + "/*"))), "wb")
        pickle.dump(backDataArray, chunk)
        chunk.close()

    def next_batch(self, size, timesteps):
        chunk = open(self.chunksFolder + "/chunk" + str(random.randint(0, len(glob.glob(self.chunksFolder + "/*")) - 1)))
        data = pickle.load(chunk)

        batchStart = random.randint(0, len(data) - size - timesteps - 2)
        batch = data[batchStart:batchStart + size + timesteps + 1]

        input_images = []
        output_images = []

        for i in range(size):
            steps = []

            for j in range(timesteps):
                steps.append(batch[i + j].to_tensor())

            input_images.append(steps)
            output_images.append(batch[i + timesteps + 1])

        resultant_batch = [input_images, output_images]
        return resultant_batch

    def set_image_settings(self, imageSize, channels):
        Image.channels = channels
        Image.imageSize = imageSize
