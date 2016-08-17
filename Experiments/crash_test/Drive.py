from datetime import datetime
import serial
import threading

# If serial has 0 prefix, then code is going from raspberry pi to arduino
# If serial has 1 prefix, then code is going from arduino to raspberry pi

class new_thread(threading.Thread):
    def __init__(self, process, trial_name, baud_rate=9600):
        threading.Thread.__init__(self)
        self.process = process
        self.baud_rate = baud_rate
        self.trial_name = trial_name

        self.ser = serial.Serial('/dev/ttyACM0', self.baud_rate)

        # self.process setup:
        # 0 - Ultra Sonic Data Management
        # 1 - Motor Management

    def run(self):
        if self.process:
            logFile = open(self.trial_name + "_motor_powers.txt", "a")

            self.ser.write("0:0:0\n")
            logFile.write("0:0:0.0\n")

            current_time = datetime.now()
            time_baselime = (current_time.day * 24 * 60 * 60 + current_time.second) * 1000 + current_time.microsecond / 1000.0

            while True:
                motor_power_raw = raw_input("Motor Power (Right [space] left:\n").split(" ")
                right_motor_power = -1 * int(motor_power_raw[0])
                left_motor_power = -1 * int(motor_power_raw[1])

                current_time = datetime.now()
                current_time_milli = (current_time.day * 24 * 60 * 60 + current_time.second) * 1000 + current_time.microsecond / 1000.0
                delta = current_time_milli - time_baselime

                self.ser.write("0:" + str(left_motor_power) + ":" + str(right_motor_power) + "\n")
                logFile.write(str(left_motor_power) + ":" + str(-1 * right_motor_power) + ":" + str(delta) + "\n")

        else:
            logFile = open(self.trial_name + "_ultrasonic_sensor_data.txt", "a")

            # Log File setup:
            # time (ms) : sensor reading

            while True:
                data = self.ser.readline()
                data_array = data.split(":")

                try:
                    if int(data_array[0]) == 1:
                        logFile.write(data_array[1] + ":" + data_array[2])
                except ValueError:
                    print "Corrupted string \n"

trial_name = raw_input("Trial Name:\n")

delegates = []
delegates.append(new_thread(1, trial_name))
delegates.append(new_thread(0, trial_name))

serial.Serial('/dev/ttyACM0', 9600).write("2\n")

for delegate in delegates:
    delegate.start()

for delegate in delegates:
    delegate.join()