import serial    #import serial module
import time
ser = serial.Serial('/dev/ttyUSB0', 9600,timeout=1);   #open named port at 9600,1s timeot

#try and exceptstructure are exception handler
while 1:
    time.sleep(3)
    ser.write(b'1080')#writ a string to port
    time.sleep(0.5)
    while True:
        response = ser.readline()#read a string from port
        if response !=b'' and response !=b'\n':
            break
        
    print(str(response))
    time.sleep(0.5)
    while True:
        response = ser.readline()
        print(response)
        if response==b'Done\r\n':
            break;
    
    
