#Imports
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from numpy import genfromtxt
import scipy
from scipy import integrate
from scipy.signal import find_peaks

#Gather and unpack data from CSV
file = '/Users/shuowanghe/github/IIB-Project2/data/adafruitapril15th/freeswing.csv'
data = genfromtxt(file,delimiter=',')
timestamps = data[:,0]
a_r = data[:,1]
a_theta = data[:,2]
theta_dot = data[:,3]*1.07
#Smooth the radial acceleration signal
filtered_a_r = scipy.signal.savgol_filter(a_r,window_length=21, polyorder=2)
#Differentiate gyro signal to get angular acceleration, then smooth with Sav-Gol filter
theta_double_dot = np.gradient(theta_dot,timestamps)
filtered_theta_double_dot = scipy.signal.savgol_filter(theta_double_dot,window_length=25, polyorder=3)
#Get theta=0 time stamps from filtered radial acceleration signal
theta_zeros,_ = scipy.signal.find_peaks(filtered_a_r,prominence=5)
#Integrate gyro signal from first theta=0 to get angle vs time, but will drift
theta = scipy.integrate.cumtrapz(theta_dot[theta_zeros[0]:],timestamps[theta_zeros[0]:],initial=0)

#Function for getting angle from gyro by re-integrating at every theta=0 and distributing the drift
def get_theta(data):
    timestamps = data[:,0] #unpack the data again, filter a_r and find theta=0s
    a_r = data[:,1]
    ang_vel = data[:,3]*1.07
    filtered_a_r = scipy.signal.savgol_filter(a_r,window_length=21, polyorder=2)
    theta_zeros,_ = scipy.signal.find_peaks(filtered_a_r,prominence=5)
    theta = np.zeros(len(timestamps)) #generate empty array to hold theta
    for _ in theta_zeros: #hard re-integrate at every theta=0
        time_section = timestamps[_:]
        theta_dot_section = ang_vel[_:]
        theta[_:] = scipy.integrate.cumtrapz(theta_dot_section,time_section,initial=0)
    theta_int_once = scipy.integrate.cumtrapz(ang_vel[theta_zeros[0]:],timestamps[theta_zeros[0]:],initial=0)
    theta_fix = np.zeros(len(timestamps)) #generate an array to hold the fixed theta
    theta_fix[theta_zeros[0]:] = theta_int_once #put the hard integrated theta between the first 2 zeros
    prev_zero = theta_zeros[0] #initiate the last theta=0 as the first one for the loop
    for _ in theta_zeros[1:]: #reintegrate and correct drift
        time_section = timestamps[prev_zero:_+1] #carve out the section between the 2 zeros
        theta_dot_section = ang_vel[prev_zero:_+1]
        theta_section = scipy.integrate.cumtrapz(theta_dot_section,time_section,initial=0) #make the integration
        drift = theta_section[-1] #find the drift at the end of the swing
        drift_vec = np.linspace(start=0,stop=drift,num=_-prev_zero+1) #generate a vector increasing steadily from 0 to the drift over that time frame
        theta_fix[prev_zero:_] = theta_section[:-1]-drift_vec[:-1] #make the correction so last theta=0
        prev_zero = _ #store the zero point for the next loop
    return theta,theta_fix #returns both the hard reintegrated theta and the drift-fixed theta

#Plot to compare reintegrated theta vs once integrated theta
plt.plot(timestamps,get_theta(data)[0],label=r'Re-zeroed $\theta$',linewidth=5)
plt.plot(timestamps,get_theta(data)[1],label=r'Re-zeroed and drift corrected $\theta$',linewidth=3)
plt.plot(timestamps[theta_zeros[0]:],theta,label=r'$\theta$ integrated from 1st zero')
plt.plot(timestamps[theta_zeros],np.zeros(len(theta_zeros)),'gx',label=r'$\theta=0$')
plt.legend(loc='lower right')
plt.title(r'Comparison of $\theta$ calculated from initial zero vs recalculated at every zero')
plt.xlabel('time(s)')
plt.ylabel(r'$\theta$(rad)')
plt.show()

#Plot to show filtered radial acceleration to find all theta=0 time stamps
plt.plot(timestamps,a_r,label=r'measured $a_r$')
plt.plot(timestamps,filtered_a_r,label=r'measured $a_r$ with Savitzky–Golay filter')
plt.plot(data[theta_zeros,0],filtered_a_r[theta_zeros],'x')
plt.legend()
plt.title(r'Filtered vs unfiltered $a_r$, used to find all $\theta=0$')
plt.xlabel('time(s)')
plt.ylabel(r'$a_r$(m/$s^2$)')
plt.show()
