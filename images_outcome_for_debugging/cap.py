from picamera import PiCamera
import picamera.array
import time
import cv2
import numpy as np
cam=PiCamera()
for i in range(15):
    cam.start_preview()
    time.sleep(2)
    cam.capture('0test'+str(i)+'.jpg')
    cam.stop_preview()
    time.sleep(5)