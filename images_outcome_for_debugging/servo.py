import RPi.GPIO as GPIO

import time

SERVO = 17

GPIO.setmode(GPIO.BCM)

GPIO.setup(SERVO, GPIO.OUT)

p = GPIO.PWM(SERVO, 50)

p.start(2.5)



p.ChangeDutyCycle(10)#2.5 12.5

time.sleep(1)

p.ChangeDutyCycle(11.7)#2.5 12.5

time.sleep(1)
p.stop()

GPIO.cleanup()
