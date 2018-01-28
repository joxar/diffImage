import cv2
from datetime import datetime
from time import sleep

# file for recognizing face
# on Mac OS
#CASCADE_DIR = "/usr/local/share/OpenCV/haarcascades/"
# on raspberryPi
CASCADE_DIR = "/usr/share/opencv/haarcascades"

CASCADE_FILE = CASCADE_DIR + "haarcascade_frontalface_alt.xml"
OUTPUT_DIR = "."

# don't recognize too small face
MIN_SIZE = (150, 150)

# generate recognizer
cascade = cv2.CascadeClassifier(CASCADE_FILE)

# open camera
camera = cv2.VideoCapture(0)

_, img = camera.read()
fname = OUTPUT_DIR + "/" + "fileB.jpg"
cv2.imwrite(fname, img)
