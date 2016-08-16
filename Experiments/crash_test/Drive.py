import serial
import threading

ser = serial.Serial('/dev/ttyACM0', 9600)
logFile = open("LogFile.txt", "a")


class new_thread(threading.Thread):
    def __init__(self, process, baud_rate):
        threading.Thread.__init__(self)
        self.process = process
        self.baud_rate = baud_rate

        # self.process setup:
        # 1 - Motor Management
        # 2 - Ultra Sonic Data Management

    def run(self):
        pass
