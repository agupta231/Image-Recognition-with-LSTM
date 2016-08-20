import os
import glob
import pickle
import matplotlib.pyplot as plot
import cv2

plot.ion()
figure = plot.figure("Distance Data")
raw_image = figure.add_subplot(221)
edges_images = figure.add_subplot(222)
text = figure.add_subplot(223)
text_ax = text.axis([-1, 1, -1, 1])

motor_powers = []

# Motor Powers Setup:
# 0 - Time
# 1 - Left Motor Power
# 2 - Right Motor Power

ultrasonic_data = []

# Ultrasonic data setup:
# 0 - Time
# 1 - Data

with open("motor_powers.txt") as logFile:
    line = logFile.readline()
    data = line.split(":")

    motor_powers.append([float(data[2]), int(data[0]), int(data[1])])

with open("ultrasonic_sensor_data.txt") as logFile:
    for line in logFile:
        data = line.split(":")
        ultrasonic_data.append([int(data[0]), int(data[1])])

txt = None
os.mkdir(os.getcwd() + "/figures/")

for i in range(len(glob.glob(os.getcwd() + "/raw/*"))):
    count = int((1000.0/60.0) * i)
    ultrasonic_reading = -3

    for j in range(len(ultrasonic_data)):
        if count == ultrasonic_data[j][0]:
            ultrasonic_reading = ultrasonic_data[j][1]
            break
        elif count > ultrasonic_data[j][0] and j is not len(ultrasonic_data) - 1:
            ultrasonic_reading = (ultrasonic_data[j][1] + ultrasonic_data[j + 1][1]) / 2
            break
        elif count > ultrasonic_data[len(ultrasonic_data) - 1][0]:
            ultrasonic_reading = ultrasonic_data[len(ultrasonic_data) - 1][1]
            break

    print str(count) + " " + str(ultrasonic_reading)

    try:
        image_raw = cv2.imread(os.getcwd() + "/raw/FRAME_" + str(count) + ".jpg")
        image_edges = cv2.imread(os.getcwd() + "/edges_1.75/FRAME_" + str(count) + ".jpg")

        raw_image.imshow(image_raw)
        raw_image.set_title("Raw Image")
        edges_images.imshow(image_edges)
        raw_image.set_title("Edges")
        txt = text.text(-0.5, 0, "Distance: " + str(ultrasonic_reading))

        figure.savefig("figures/FRAME_" + str(count) + ".png")
        txt.remove()

    except TypeError:
        print "Image " + str(count) + " not found"