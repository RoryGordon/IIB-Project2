import serial
import time
import csv

arduino = serial.Serial('/dev/cu.usbmodem14101',38400)
start_time = time.time()
array = []
i=0
while True:
    serial_output = str(arduino.readline())
    if serial_output:
        elapsed_time = time.time()-start_time
        str_idx = serial_output.find('g:')
        mass = float(serial_output[str_idx+3:-8])
        array.append([elapsed_time,mass])
        print('Elapsed Time:',"%.3f" % round(elapsed_time,2),'s, Mass:',mass,'kg')
        with open('/Users/shuowanghe/github/IIB-Project/test.csv','a',) as newFile:
            newFileWriter = csv.writer(newFile)
            newFileWriter.writerow([elapsed_time,mass])
