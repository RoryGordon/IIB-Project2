# THIS CELL SHOULD ONLY NEED TO BE RUN ONCE, AT THE BEGINNING OF THE SESSION.

# EDIT THESE LINES

comport='/dev/cu.usbmodem14101' # Enter the COM port number. Default is comport='COM4'
watermark='WordWord'  # Enter the two 4-letter words provided to you by the demonstrator. Default is watermark='WordWord'
userids=['userid1','userid2']     # Enter list of lab group's userids; whoever is logged on should be listed first. Default is userids=['userid1','userid2']
carnumber=0            # Enter the car number (integer in the range 1-8). Default is carnumber=0
# DO NOT CHANGE ANYTHING BELOW THIS LINE

#
# Import python packages
#
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import math
import serial
import time
import numpy as np
from numpy import genfromtxt
import csv
from cobs import cobs
from scipy import integrate

#
# Open the USB port into which the radio is plugged
#
print('Wait a few seconds for serial port to open...')
try:
  ser = serial.Serial(comport, baudrate=57600)
  ser.isOpen() # try to open the port; will cause error if port is already open
except IOError: # if the port is already open, close it and open it again
  ser.close()
  ser.open()

time.sleep(2)  # pause for 2 seconds to allow port to settle down
ser.flush()    # flush the port to remove any spurious data
print('Serial port open:',ser.name) # print name of port to confirm correct opening

#
# Define a function to read a line of data from the serial port.
#
# The arduino sends a line of data from the Arduino twenty times a second.
# Each line contains: one measurement from each sensor; the time; and a checksum.
# Each line is terminated by an end of line character.
def readmotiondata():
    eol = bytes([0])  # define the end of line character as a zero byte
    leneol = len(eol)
    motiond = bytearray() # set up an array of bytes
    while True:
        c = bytes(ser.read()) # read a byte from the serial port into variable c
        if c == eol:
            break    # if the end of line character is read, return from the function with the bytes in motiond
        else:
            motiond += c  # otherwise append the byte to the array
    return bytes(motiond)
# Execute the readmotiondata function once to read the first (possibly incomplete) line of bytes and ignore it
readmotiondata()

#
# Initialise some variables
#
motiondata=np.array([],dtype='uint16') # Initialise a numpy array of unsigned 16-bit (two byte) integers
axlist=[]  # initialise some python lists (lists are computationally more efficient than numpy arrays when we don't know the final size)
aylist=[]
omegazlist=[]
tlist=[]
counterlist=[]


axlist[:]=[] # empty the lists
aylist[:]=[]
omegazlist[:]=[]
tlist[:]=[]
counterlist[:]=[]

for x in range(150):    # read sufficient number of lines from serial port to flush buffer before recording/plotting
    readmotiondata()


def update():
    motioncoded = readmotiondata()  # serial read into bytes object converted to list of ints, last element is line feed
    try:
        motiondata = cobs.decode(motioncoded) #cobs
    except cobs.DecodeError:
        print('COBS DecodeError')
    else:
        motiondata = list(motiondata)  # bytes object converted to list of ints, last element is line feed
        checksumrecvd=np.sum(motiondata[0:-1],dtype=np.uint8) # checksum
        if (checksumrecvd != motiondata[-1]):
            print('Checksum error')
        else:
            millis=np.uint32(motiondata[0] | motiondata[1]<<8 | motiondata[2]<<16 | motiondata[3]<<24)
            accx=np.int16(motiondata[4] | motiondata[5]<<8)
            accy=np.int16(motiondata[6] | motiondata[7]<<8)
            gyrz=np.int16(motiondata[20] | motiondata[21]<<8)
            encoder=np.int16(motiondata[22] | motiondata[23]<<8) # 22 = 4 time bytes + 18 imu bytes

    return (millis, accx, accy, gyrz, encoder)
secs = 0


file = '/Users/shuowanghe/github/IIB-Project2/data/adafruitapril15th/freeswing.csv'

while secs<=60:
    (millis, accx, accy, gyrz, encoder) = update()
    secs = millis/1000
    accx = accx/100
    accy = accy/100
    gyrz = gyrz/900
    with open(file,'a',) as newFile:
        newFileWriter = csv.writer(newFile)
        newFileWriter.writerow([secs,accx,accy,gyrz])
    print('Elapsed Time:',"%.2f" % round(secs,2),'s, Ax:',"%.2f" % round(accx,2),'m/s/s, Ay:',"%.2f" % round(accy,2),'m/s/s, GyroZ:',"%.2f" % round(gyrz,2),'rad/s')

data = genfromtxt(file, delimiter=',')
ts = data[:,0]
ax = data[:,1]
ay = data[:,2]
gz = data[:,3]
plt.plot(ts,ax)
plt.show()
plt.plot(ts,ay)
plt.show()
plt.plot(ts,gz)
plt.show()
