import serial
import os
import time
import random

ser = serial.Serial('/dev/ttyACM0', 9600)

DRIVE_TIME_MIN = 0.3
DRIVE_TIME_MAX = 1.2
DRIVE_TIME_STANDARD = 0.75
TIME_RANDOM = False

trial_name = raw_input("Trial Name:\n")

os.system("mkdir " + os.getcwd() + "/" + trial_name)
logFile = open(os.getcwd() + "/" + trial_name + "/" + "LogFile.txt", "a")

timeBetweenIntervals = raw_input("Time between frames: (rand for random)\n")

if timeBetweenIntervals == "rand":
    TIME_RANDOM = True

"""
--- Log File Format ---
rightMotorPower : leftMotorPower : Milliseconds
"""

timeCount = 0.0

# Initial State
logFile.write("0:0:0.0\n")

while True:
    leftMotorPower = random.randint(*random.choice([(-255, -25), (25, 255)]))
    rightMotorPower = random.randint(*random.choice([(-255, -25), (25, 255)]))

    if TIME_RANDOM:
        sleepTime = random.uniform(DRIVE_TIME_MIN, DRIVE_TIME_MAX)
    else:
        sleepTime = DRIVE_TIME_STANDARD

    time.sleep(sleepTime)

    timeCount += sleepTime

    ser.write(str(leftMotorPower) + ":" + str(rightMotorPower))
    logFile.write(str(leftMotorPower) + ":" + str(rightMotorPower) + ":" + str(timeCount))