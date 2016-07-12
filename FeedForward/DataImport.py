import glob
from Image import Image
import random


class DataImport:
    def __init__(self, framesFolder):
        self.framesFolder = framesFolder
        self.dataArray = []

    def importFolder(self, folderPath):
        breakMarker = False

        with open(folderPath + "/LogFile.txt") as logFile:
            for line in logFile:
                lineData = line.split(":")

                # frameNumber = lineData[0]
                # leftMotorPower = lineData[1]
                # rightMotorPower = lineData[2]
                # duration = lineData[3]
                # timestamp = lineData[4]

                file = glob.glob(folderPath + "/" + self.framesFolder + "/FRAME_" + lineData[0] + "_*")

                if len(file) != 1:
                    print ("Frame " + lineData[0] + " cannot be found")

                    if len(self.dataArray) > 1:
                        breakMarker = True

                    continue

                if breakMarker:
                    self.dataArray.append(Image(
                        file,
                        0.0,
                        0.0,
                        0.0,
                        float(lineData[4].rstrip())
                    ))
                else:
                    self.dataArray.append(Image(
                        file,
                        float(lineData[1]),
                        float(lineData[2]),
                        float(lineData[3]),
                        float(lineData[4].rstrip())
                    ))

    def next_batch(self, size):
        batchStart = random.randint(0, len(self.dataArray) - (size + 1))
        batch = self.dataArray[batchStart:batchStart + size]

        input_image = []
        rightMotorPower = []
        leftMotorPower = []
        duration = []
        output_image = []

        for i in range(len(batch)):
            input_image.append(batch[i].to_tensor())
            rightMotorPower.append(batch[i + 1].rightMotorPower)
            leftMotorPower.append(batch[i + 1].leftMotorPower)
            duration.append(batch[i + 1].duration)

            if batch[i + 1].rightMotorPower == [1] and batch[i + 1].leftMotorPower == [0]:
                output_image.append(batch[i].to_flattened_tensor())
            else:
                output_image.append(batch[i + 1].to_flattened_tensor())

        resultant_batch = [input_image, rightMotorPower, leftMotorPower, duration, output_image]
        return resultant_batch

    def setImage(self, imageWidth, imageHeight, channels):
        Image.image_width = imageWidth
        Image.image_height = imageHeight,
        Image.channels = channels