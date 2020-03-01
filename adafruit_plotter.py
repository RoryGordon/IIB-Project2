from numpy import genfromtxt
import numpy as np
import scipy as sp
from scipy.signal import find_peaks
import csv
import matplotlib.pyplot as plt

file = '/Users/shuowanghe/github/IIB-Project/test.csv'
data = genfromtxt(file, delimiter=',')
axpeaks,_ = sp.signal.find_peaks(data[:,1],height=5,distance=50)
aypeaks,_ = sp.signal.find_peaks(data[:,2],height=0,distance=40)
gzpeaks,_ = sp.signal.find_peaks(data[:,3],height=0,distance=50)
zeros = min(aypeaks, key=lambda x:abs(x))
print(zeros)
#plot 1 (x acceleration)
plt.plot(data[:,0],data[:,1])
plt.plot(data[axpeaks,0],data[axpeaks,1],"x")
plt.xlabel('Time (s)')
plt.ylabel('Normal Acceleration (m/s/s)')
plt.show()
#plot 2 (y acceleration)
plt.plot(data[:,0],data[:,2])
plt.plot(data[aypeaks,0],data[aypeaks,2],"x")
plt.plot(data[zeros,0],data[zeros,2],"x",color='b')
plt.xlabel('Time (s)')
plt.ylabel('Tangential Acceleration (m/s/s)')
plt.show()
#plot 3 (z angular velocity)
plt.plot(data[:,0],data[:,3])
plt.plot(data[gzpeaks,0],data[gzpeaks,3],"x")
plt.xlabel('Time (s)')
plt.ylabel('Angular velocity (rad/s)')
plt.show()
#plot 4 (all plots)
plt.plot(data[:,0],data[:,1])
plt.plot(data[:,0],data[:,2])
plt.plot(data[:,0],data[:,3])
for xc in axpeaks:
    xc = data[xc,0]
    plt.axvline(x=xc,color='b',linestyle='--')
#for xc in aypeaks:
    #xc = data[xc,0]
    #plt.axvline(x=xc,color='r',linestyle='--')
for xc in gzpeaks:
    xc = data[xc,0]
    plt.axvline(x=xc,color='g',linestyle='--')
plt.xlabel('Time (s)')
plt.show()
