import serial

ser = serial.Serial('/dev/ttyACM0', 9600)

logFile = open("LogFile.txt", "a")

while True:
    power = raw_input("Motor Power:\n")

    ser.write(power + ":" + power)
    logFile.write(power + "\n")
