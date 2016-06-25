import serial
import time
import random
import os

ser = serial.Serial('/dev/ttyACM0', 9600)

randomTime = False
time_max = 3
time_min = 0

count = int(raw_input("How many photos / frames?\n"))
timeBetweenIntervals = raw_input("Time between frames: (rand for random)\n")

if(timeBetweenIntervals === "rand"):
	randomTime = True
else:
	timeBetweenIntervals = int(timeBetweenIntervals)

for i in range(count):
	leftMotorPower = random.randInt(-255, 255)
	rightMotorPower = random.randInt(-255, 255)

	ser.write(leftMotorPower + " " + rightMotorPower)

	if(randomTime):
		sleepTime = random.uniform(time_min, time_max)
		time.sleep(sleepTime)
	else:
		sleepTime = timeBetweenIntervals
		time.sleep(sleepTime)

	ser.write("0 0")
	time.sleep(0.25)

	os.system("fswebcam FRAME_" + i +".jpg")
	time.sleep(0.25)