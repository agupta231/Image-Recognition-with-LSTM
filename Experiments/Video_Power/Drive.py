import serial
import os
import time
import random

ser = serial.Serial('/dev/ttyACM0', 9600)

DRIVE_TIME_MIN = 0.3
DRIVE_TIME_MAX = 1.2

trial_name = raw_input("Trial Name:\n")

os.system("mkdir " + os.getcwd() + "/" + trial_name)

logFile = open(os.getcwd() + "/" + trial_name + "/" + "LogFile.txt", "a")

count = int(raw_input("How many photos / frames?\n"))

