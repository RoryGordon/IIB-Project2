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


# DO NOT EDIT ANY OF THIS CELL

print('Wait up to 8 s for plot to appear...')

# use 'magic' to allow matplotlib to work properly in the notebook
# ensure this line is after the opening of the serial port
# %matplotlib notebook

twidth=3 # length of time axis (plotting is slow and hence serial buffers fill up if time axis is too long)
# check this is short enough by comparing real time with time on x-axis during plotting.

axlist[:]=[] # empty the lists
aylist[:]=[]
omegazlist[:]=[]
tlist[:]=[]
counterlist[:]=[]

fig=plt.figure(0,figsize=(9.5,6))

axes1=plt.subplot(221)   # cartesian plot
line1, = axes1.plot(tlist, aylist, marker='o', markersize=3, color="red")

axes2=plt.subplot(222)   # cartesian plot
line2, = axes2.plot(tlist, counterlist, marker='o', markersize=3, color="green")

axes3=plt.subplot(223)   # cartesian plot
line3, = axes3.plot(tlist, counterlist, marker='o', markersize=3, color="blue")

axes4=plt.subplot(224)   # cartesian plot
line4, = axes4.plot(tlist, axlist, marker='o', markersize=3, color="orange")

line=[line1,line2,line3,line4] # list of line objects

anim_running=False # boolean to indicate whether animation should be running or not

for x in range(150):    # read sufficient number of lines from serial port to flush buffer before recording/plotting
    readmotiondata()

print('Press any key to start or stop recording/plotting (do not use the icons above)')

#def onClick(event):
def press(event):
    global anim_running
    if anim_running:
        anim_running = False
    else:
        axlist[:]=[]    # empty the lists before starting the recording/plotting
        aylist[:]=[]
        omegazlist[:]=[]
        tlist[:]=[]
        counterlist[:]=[]
        anim_running = True

def init():
    axes1.set_xlim(0,twidth)
    axes1.set_ylim(-1,1)
    axes1.set_ylabel(r'$a_y\ /\ \rm{m \ s^{-2}}$')
    axes2.set_xlim(0,twidth)
    axes2.set_ylim(-1,1)
    axes2.set_ylabel(r'$n$')
    axes3.set_xlim(0,twidth)
    axes3.set_ylim(-1,1)
    axes3.set_ylabel(r'$\omega_z\ /\ \rm{rad \ s^{-1}}$')
    axes3.set_xlabel(r'time $t\ /\ \rm{s}$')
    axes4.set_xlim(0,twidth)
    axes4.set_ylim(-1,1)
    axes4.set_ylabel(r'$a_x\ /\ \rm{m \ s^{-2}}$')
    axes4.set_xlabel(r'time $t\ /\ \rm{s}$')
    axes1.set_title('Lateral acceleration')
    axes2.set_title('Wheel encoder count')
    axes3.set_title('Yaw velocity')
    axes4.set_title('Longitudinal acceleration')
    return line

def update(frame):
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

    if anim_running:
        axlist.append(accx/100)      # x acceleration, /100 to convert to m/s/s
        aylist.append(accy/100)      # y acceleration, /100 to convert to m/s/s
        omegazlist.append(gyrz/900)  # z velocity, /900 to convert to rad/s
        counterlist.append(encoder)  # encoder count
        tlist.append(millis/1000)    # time, /1000 to convert to s

    taxis=tlist-tlist[0] # adjust the time at the plot origin to zero
    tmin=max(taxis[0],taxis[-1]-twidth)
    tmax=max(taxis[0]+twidth,taxis[-1])
    axes1.set_xlim(tmin,tmax)
    axes1.set_ylim(min(aylist),max(aylist))
    axes2.set_xlim(tmin,tmax)
    axes2.set_ylim(min(counterlist)-1,max(counterlist)+1)
    axes3.set_xlim(tmin,tmax)
    axes3.set_ylim(min(omegazlist),max(omegazlist))
    axes4.set_xlim(tmin,tmax)
    axes4.set_ylim(min(axlist),max(axlist))
    line1.set_data(taxis,aylist)
    line2.set_data(taxis,counterlist)
    line3.set_data(taxis,omegazlist)
    line4.set_data(taxis,axlist)
    return line

fig.canvas.mpl_connect('key_press_event', press)

animation = FuncAnimation(fig, update, init_func=init, interval=1, blit=False)
# interval is in ms, set shorter than time step of data sent by Arduino,
# so that update occurs as soon as data arrives from Arduino

plt.show()
