import glob
from Image import Image

class DataImport:
    def __init__(self, framesFolder):
        self.framesFolder = framesFolder

    def importFolder(self, folderPath):
        dataArray = []
        breakMarker = False

        with open(folderPath + "LogFile.txt") as logFile:
            for line in logFile:
                lineData = line.split(":")

                # frameNumber = lineData[0]
                # leftMotorPower = lineData[1]
                # rightMotorPower = lineData[2]
                # duration = lineData[3]
                # timestamp = lineData[4]

                file = glob.glob(folderPath + self.framesFolder + "/FRAME_" + lineData[0] + "_*")

                if len(file) != 1:
                    print ("Frame " + lineData[0] + " cannot be found")

                    if len(dataArray) > 1:
                        breakMarker = True

                    continue

                if breakMarker:
                    dataArray.append(Image(
                        file,
                        0,
                        0,
                        0,
                        int(lineData[4])
                    ))
                else:
                    dataArray.append(Image(
                        file,
                        int(lineData[1]),
                        int(lineData[2]),
                        int(lineData[3]),
                        int(lineData[4])
                    ))
        return dataArray
