import serial
import os
import time
import random

ser = serial.Serial('/dev/ttyACM0', 9600)

FPS = 60.0
MINIMUM_MULTIPLE = 40
MAXIMUM_MULTIPLE = 325
DRIVE_TIME_STANDARD = 0.75
TIME_RANDOM = False
MOTOR_POWER_MINIMUM = 200
MOTOR_POWER_MAXIMUM = 255

trial_name = raw_input("Trial Name:\n")

os.system("mkdir " + os.getcwd() + "/" + trial_name)
logFile = open(os.getcwd() + "/" + trial_name + "/" + "LogFile.txt", "a")

timeBetweenIntervals = raw_input("Time between frames: (rand for random)\n")

if timeBetweenIntervals == "rand":
    TIME_RANDOM = True

# Go forward a bit to sync the video and log files
ser.write("255:255")
time.sleep(1)

ser.write("0:0")

"""
--- Log File Format ---
rightMotorPower : leftMotorPower : Milliseconds
"""

timeCount = 0.0

# Initial State
logFile.write("0:0:0.0\n")

while True:
    leftMotorPower = random.randint(*random.choice([(-MOTOR_POWER_MAXIMUM, -MOTOR_POWER_MINIMUM), (MOTOR_POWER_MINIMUM, MOTOR_POWER_MAXIMUM)]))
    rightMotorPower = random.randint(*random.choice([(-MOTOR_POWER_MAXIMUM, -MOTOR_POWER_MINIMUM), (MOTOR_POWER_MINIMUM, MOTOR_POWER_MAXIMUM)]))

    if TIME_RANDOM:
        sleepTime = (1.0 / FPS) * random.randint(MINIMUM_MULTIPLE, MAXIMUM_MULTIPLE)
    else:
        sleepTime = DRIVE_TIME_STANDARD

    time.sleep(sleepTime)

    timeCount += sleepTime

    ser.write(str(leftMotorPower) + ":" + str(rightMotorPower))
    logFile.write(str(leftMotorPower) + ":" + str(rightMotorPower) + ":" + str(timeCount * 1000) + "\n")

    print "\n Right motor power: " + str(rightMotorPower) + "\n Left motor power: " + str(leftMotorPower) + "\n"
