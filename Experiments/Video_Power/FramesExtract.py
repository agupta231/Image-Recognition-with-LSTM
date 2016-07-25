import os
import cv2
import glob

print("OpenCV Version:" + cv2.__version__)

folderName = raw_input("What is the folder name?:\n")

frontDir = os.getcwd() + "\\" + folderName + "\\front\\"
backDir = os.getcwd() + "\\" + folderName + "\\back\\"

frontVidCap = cv2.VideoCapture(frontDir + "rawTrimmed.mp4")

print (frontDir + "rawTrimmed.mp4")
backVidCap = cv2.VideoCapture(backDir + "rawTrimmed.mp4")

print(frontVidCap.read())