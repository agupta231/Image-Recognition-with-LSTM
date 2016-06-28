import serial
import time
import random
import os

ser = serial.Serial('/dev/ttyACM0', 9600)

randomTime = False
time_max = 3
time_min = 1.5

name = raw_input("Name of Trial:\n")
os.system("mkdir " + os.getcwd() + "/" + name)
os.system("mkdir " + os.getcwd() + "/" + name + "/frames")

logFile = open(os.getcwd() + "/" + name + "/" + "LogFile.txt", "a")

#########  FORMAT OF LOG FILE ##################################################
#
#	Frame Number : Left Motor Power : Right Motor Power : Duration : Timestamp
#
################################################################################

count = int(raw_input("How many photos / frames?\n"))
timeBetweenIntervals = raw_input("Time between frames: (rand for random)\n")

if(timeBetweenIntervals == "rand"):
	randomTime = True
else:
	timeBetweenIntervals = int(timeBetweenIntervals)

## First photo
os.system("fswebcam --no-banner " + os.getcwd() + "/" + name + "/frames/FRAME_0.jpg")
logFile.write("0:0:0:0:" + str(time.time()) + "\n")

## Rest of the frames
for i in range(1, count):
	leftMotorPower = random.randint(*random.choice([(-255, -150), (150, 255)]))
	rightMotorPower = random.randint(*random.choice([(-255, -150), (150, 255)]))

	ser.write(str(leftMotorPower) + ":" + str(rightMotorPower))

	sleepTime = timeBetweenIntervals

	if(randomTime):
		sleepTime = random.uniform(time_min, time_max)

	time.sleep(sleepTime)

	ser.write("0:0")
	time.sleep(0.25)

	os.system("fswebcam --no-banner " + os.getcwd() + "/" + name + "/frames/FRAME_" + i +".jpg")
	logFile.write(i + ":" + str(leftMotorPower) + ":" + str(rightMotorPower) + ":" + str(sleepTime) + ":" + str(time.time()) + "\n")
	time.sleep(0.25)