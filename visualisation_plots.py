###-------------------------------------------------------------------------###
###---------------------------------IMPORTS---------------------------------###
###-------------------------------------------------------------------------###
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from numpy import genfromtxt
import scipy
from scipy import integrate
from scipy.signal import find_peaks
from matplotlib import animation
from statsmodels.graphics import tsaplots
###-------------------------------------------------------------------------###
###--------------------------------FUNCTIONS--------------------------------###
###-------------------------------------------------------------------------###
#Function for getting angle from gyro by re-integrating at every theta=0 and distributing the drift
def get_theta(data):
    #find theta=zeros
    a_r = data[:,1]
    filtered_a_r = scipy.signal.savgol_filter(a_r,window_length=21, polyorder=2)
    theta_zeros,_ = scipy.signal.find_peaks(filtered_a_r,prominence=5)
    #integrate theta and distribute drift before every zero
    theta_dot_correction_factor = 1.07
    timestamps = data[:,0]
    theta_dot = data[:,3]*theta_dot_correction_factor
    theta_int_once = scipy.integrate.cumtrapz(theta_dot[theta_zeros[0]:],timestamps[theta_zeros[0]:],initial=0)
    theta_fix = np.zeros(len(timestamps)) #generate an array to hold the fixed theta
    theta_fix[theta_zeros[0]:] = theta_int_once #put the hard integrated theta between the first 2 zeros
    prev_zero = theta_zeros[0] #initiate the last theta=0 as the first one for the loop
    for _ in theta_zeros[1:]: #reintegrate and correct drift
        time_section = timestamps[prev_zero:_+1] #carve out the section between the 2 zeros
        theta_dot_section = theta_dot[prev_zero:_+1]
        theta_section = scipy.integrate.cumtrapz(theta_dot_section,time_section,initial=0) #make the integration
        drift = theta_section[-1] #find the drift at the end of the section
        drift_vec = np.linspace(start=0,stop=drift,num=_-prev_zero+1) #generate a vector increasing steadily from 0 to the drift over that time frame
        theta_fix[prev_zero:_] = theta_section[:-1]-drift_vec[:-1] #make the correction so the last theta=0
        prev_zero = _ #store the zero point for the next loop
    return theta_fix #returns the drift-fixed theta

#Function for getting the gradient in the sin(theta) vs ang accel equation
def get_gradient(data):
    theta = get_theta(data) #get drift corrected theta from the data
    x = np.sin(theta)[theta_zeros[0]:]
    y = scipy.signal.savgol_filter(np.gradient(data[:,3]*1.07,data[:,0]),window_length=25, polyorder=3)[theta_zeros[0]:]
    allowed_indices = np.where(abs(x)<0.2) #find indices where line is straight
    p = np.polyfit(x[allowed_indices],y[allowed_indices],deg=1) #fit a line through all of the data points within the cutoff
    plt.plot(x,y,'x') #plot all the data points
    plt.plot(x[allowed_indices],y[allowed_indices],'gx') #plot the allowed data points for fitting
    plt.plot(x,p[0]*x,'r') #plot the fitted line, through the origin
    plt.show()
    return p[0]

###-------------------------------------------------------------------------###
###----------------------------------DATA-----------------------------------###
###-------------------------------------------------------------------------###
#Gather and unpack data from CSV
file = '/Users/shuowanghe/github/IIB-Project2/data/adafruitapril15th/2sidepush.csv'
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

#get the -mlg/J gradient from fitting the straight part of the graph
p = get_gradient(data)
#Use re-integrated, drift corrected theta from now on
theta = get_theta(data)
#Calculate some force quantity T/J
force = filtered_theta_double_dot[theta_zeros[0]:]-p*np.sin(theta[theta_zeros[0]:])
peaks,_ = scipy.signal.find_peaks(abs(scipy.signal.savgol_filter(force,window_length=25, polyorder=3)),prominence=None)



###-------------------------------------------------------------------------###
###----------------------------------PLOTS----------------------------------###
###-------------------------------------------------------------------------###
#Plot bell angle against some force calculation
force = filtered_theta_double_dot[theta_zeros[0]:]-p*np.sin(theta[theta_zeros[0]:])
peaks,_ = scipy.signal.find_peaks(abs(scipy.signal.savgol_filter(force,window_length=25, polyorder=3)),prominence=None)
tsaplots.plot_acf(force,lags=2000)
plt.show()
plt.plot(timestamps[theta_zeros[0]:],force,label="Force measurement")
#plt.plot(timestamps[theta_zeros[0]:][peaks],scipy.signal.savgol_filter(force,window_length=25, polyorder=3)[peaks],'x')
plt.plot(timestamps[theta_zeros[0]:],scipy.signal.savgol_filter(force,window_length=25, polyorder=3),label="Force measurement (Smoothed)")
plt.plot(timestamps[theta_zeros[0]:],theta[theta_zeros[0]:],label=r'$\theta$')
plt.plot(timestamps[theta_zeros[0]:],theta_dot[theta_zeros[0]:],label=r'$\dot{\theta}$')

#plt.plot(timestamps[theta_zeros[0]:],filtered_theta_double_dot[theta_zeros[0]:])
#plt.plot(timestamps[theta_zeros[0]:],p*np.sin(theta[theta_zeros[0]:]))
plt.xlabel(r't(s)')
plt.ylabel(r'$\"{\theta}+\frac{mgl}{J}sin(\theta)$(rad/$s^2$)')
plt.axhline(0,color='b',linestyle='--')
plt.title(r'$\frac{T}{J}$ vs time')
plt.legend()
plt.show()
print(-p)
plt.plot(-p*np.sin(theta[theta_zeros[0]:]),force)
plt.show()

#Other plots for visualisation
#plt.plot(timestamps,theta_dot,label='measured ang vel')
plt.plot(timestamps,scipy.signal.savgol_filter(a_theta,window_length=21, polyorder=2),label='measured tangential acc')
#plt.plot(timestamps,a_r,label='measured radial acc')
#plt.plot(timestamps_trunc1[:-1],theta1,label='theta (from integration)')
#plt.plot(timestamps_trunc2[:-1],abs(theta2),'b')
#plt.plot(data[theta_zeros,0],theta_dot[theta_zeros],'x')
#plt.plot(timestamps,theta_double_dot,label='theta double dot (from diff)')
#plt.plot(timestamps,filtered_theta_double_dot,label='theta double dot (from diff and filtered)')
#plt.plot(timestamps_trunc1[:-1],9.81*np.cos(theta1)+0.43*np.square(theta_dot_trunc1[:-1]),label='radial acc prediction')
#plt.plot(timestamps_trunc1[:-1],a_r[first_theta_zero_idx:-1]-9.81*np.cos(theta1),label='r thetadot^2')
#plt.plot(timestamps_trunc1[:-1],a_r[first_theta_zero_idx:-1]+9.81*np.sin(theta1),label='L theta double dot')
#plt.plot(timestamps,5*np.square(theta_dot),label='thetadot^2')
#plt.plot(timestamps_trunc1[:-1],np.cos(theta1),label='cos(theta)')
#plt.plot(theta_double_dot[-len(timestamps_trunc1):-1],-np.sin(theta1))
#plt.plot(a_theta[-len(timestamps_trunc1):-861],-np.sin(theta1[0:200]))
plt.legend(loc='lower left')
plt.axhline(0,color='b',linestyle='--')
#plt.xlim(0,timestamps[-1])
#plt.ylim(-5,5)
plt.title('Other temporal plots that may be of interest for visualisation')
plt.xlabel('time(s)')
plt.show()
