import serial
import time
import random
import os

ser = serial.Serial('/dev/ttyACM0', 9600)

randomTime = False
time_max = 3
time_min = 0

name = raw_input("Name of Trial:\n")
os.makedirs(os.getcwd() + "/" + name)

logFile = open(os.getcwd() + "/" + name + "/" + "LogFile.txt", "a")

#########  FORMAT OF LOG FILE ##########################
#
#	Frame Number : Left Motor Power : Right Motor Power
#
########################################################

count = int(raw_input("How many photos / frames?\n"))
timeBetweenIntervals = raw_input("Time between frames: (rand for random)\n")

if(timeBetweenIntervals === "rand"):
	randomTime = True
else:
	timeBetweenIntervals = int(timeBetweenIntervals)

## First photo
os.system("fswebcam --no-banner " + os.getcwd() + "/" + name + "/frames/FRAME_0.jpg")
logFile.write("0:0:0\n")

## Rest of the frames
for i in range(1, count):
	leftMotorPower = random.randInt(-255, 255)
	rightMotorPower = random.randInt(-255, 255)

	ser.write(leftMotorPower + ":" + rightMotorPower)

	if(randomTime):
		sleepTime = random.uniform(time_min, time_max)
		time.sleep(sleepTime)
	else:
		sleepTime = timeBetweenIntervals
		time.sleep(sleepTime)

	ser.write("0:0")
	time.sleep(0.25)

	os.system("fswebcam --no-banner " + os.getcwd() + "/" + name + "/frames/FRAME_" + i +".jpg")
	logFile.write(i + ":" + leftMotorPower + ":" + rightMotorPower + "\n")
	time.sleep(0.25)