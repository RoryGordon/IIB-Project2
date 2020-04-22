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
def get_theta(data,theta_dot_correction_factor):
    #find theta=zeros
    a_r = data[:,1]
    filtered_a_r = scipy.signal.savgol_filter(a_r,window_length=21, polyorder=2)
    theta_zeros,_ = scipy.signal.find_peaks(filtered_a_r,prominence=5)
    #integrate theta and distribute drift before every zero
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
def get_gradient(data,theta_dot_correction_factor):
    theta = get_theta(data,theta_dot_correction_factor) #get drift corrected theta from the data
    timestamps = data[:,0]
    theta_dot = data[:,3]*theta_dot_correction_factor
    theta_double_dot = np.gradient(theta_dot,timestamps)
    x = np.sin(theta)[theta_zeros[0]:]
    y = scipy.signal.savgol_filter(theta_double_dot,window_length=25, polyorder=3)[theta_zeros[0]:]
    allowed_indices = np.where(abs(x)<0.2) #find indices where line is straight
    p = np.polyfit(x[allowed_indices],y[allowed_indices],deg=1) #fit a line through all of the data points within the cutoff
    plt.plot(x,y,'x',label='All points') #plot all the data points
    plt.plot(x[allowed_indices],y[allowed_indices],'gx',label='Points used in fitting') #plot the allowed data points for fitting
    plt.plot(x,p[0]*x,'r',label='Line fit through origin of central points') #plot the fitted line, through the origin
    plt.xlabel(r'sin($\theta$)')
    plt.ylabel(r'$\"{\theta}$(rad/$s^2$)')
    plt.title(r'$\"{\theta}$ vs sin($\theta$) with a line fitted through')
    plt.show()
    return p[0]

###-------------------------------------------------------------------------###
###----------------------------------DATA-----------------------------------###
###-------------------------------------------------------------------------###
#Gather and unpack data from CSV
file = '/Users/shuowanghe/github/IIB-Project2/data/adafruitapril15th/freeswing.csv'
data = genfromtxt(file,delimiter=',')
theta_dot_correction_factor = 1.08
gradient_correction_factor = 1.05
timestamps = data[:,0]
a_r = data[:,1]
a_theta = data[:,2]
theta_dot = data[:,3]*theta_dot_correction_factor
#Smooth the radial acceleration signal
filtered_a_r = scipy.signal.savgol_filter(a_r,window_length=21, polyorder=2)
#Differentiate gyro signal to get angular acceleration, then smooth with Sav-Gol filter
theta_double_dot = np.gradient(theta_dot,timestamps)
filtered_theta_double_dot = scipy.signal.savgol_filter(theta_double_dot,window_length=25, polyorder=3)
#Get theta=0 time stamps from filtered radial acceleration signal
theta_zeros,_ = scipy.signal.find_peaks(filtered_a_r,prominence=5)

#get the -mlg/J gradient from fitting the straight part of the graph
p = get_gradient(data,theta_dot_correction_factor)*gradient_correction_factor
#Use re-integrated, drift corrected theta from now on
theta = get_theta(data,theta_dot_correction_factor)
#Calculate some force quantity T/J
force = filtered_theta_double_dot[theta_zeros[0]:]-p*np.sin(theta[theta_zeros[0]:])


###-------------------------------------------------------------------------###
###--------------------------------ANIMATIONS-------------------------------###
###-------------------------------------------------------------------------###
#THETA_DOUBLE_DOT vs SIN(THETA) ANIMATION
fig = plt.figure()
ax = plt.axes(xlim=(-1.5, 1.5), ylim=(-10,10))
line, = ax.plot([], [], lw=2)
def init():
    line.set_data([], [])
    return line,
#Animation function. This is called sequentially
def animate(i):
    y = filtered_theta_double_dot[theta_zeros[0]:i+theta_zeros[0]]
    x = np.sin(theta[theta_zeros[0]:i+theta_zeros[0]])
    line.set_data(x, y)
    return line,
#Call the animator. blit=True means only re-draw the parts that have changed.
anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=len(theta), interval=20, blit=True)
#Save the animation
#anim.save('2sidepush.gif', fps=50, extra_args=['-vcodec', 'libx264'])
#Plot the animation
x = np.sin(theta)[theta_zeros[0]:]
plt.plot(x,p*x,'r') #plot the fitted line, through the origin
plt.xlabel(r'sin($\theta$)')
plt.ylabel(r'$\"{\theta}$(rad/$s^2$)')
plt.title(r'$\"{\theta}$ vs sin($\theta$)')
plt.show()

#PENDULUM AND FORCE ANIMATION
fig_pend = plt.figure()
ax_pend = plt.axes(xlim=(-1.5, 1.5), ylim=(-1.5,1.5))
bell, = ax_pend.plot([], [], 'bo', lw=5,label='Bell')
torque, = ax_pend.plot([], [], 'ro', lw=5, label='Force')
def init_pend():
    bell.set_data([], [])
    torque.set_data([], [])
    return bell, torque,
#Animation function. This is called sequentially
def animate_pend(i):
    y = -np.cos(theta[i]) #Bell's y position
    x = np.sin(theta[i]) #Bell's x position
    bell.set_data(x, y)
    x_torque = filtered_theta_double_dot[i]-p*np.sin(theta[i]) #Torque/J at time i
    torque.set_data(x_torque/3,0)
    return bell, torque,
#Call the animator. blit=True means only re-draw the parts that have changed.
anim_pend = animation.FuncAnimation(fig_pend, animate_pend, init_func=init_pend,
                               frames=len(theta), interval=20, blit=True)
#anim_pend.save('2sidepush_force.gif', fps=50, extra_args=['-vcodec', 'libx264'])
plt.title('Bell swinging and force applied')
plt.legend()
plt.show()

###-------------------------------------------------------------------------###
###----------------------------------PLOTS----------------------------------###
###-------------------------------------------------------------------------###
#Force, theta and theta_dot vs time
plt.plot(timestamps[theta_zeros[0]:],force,label="Force measurement")
plt.plot(timestamps[theta_zeros[0]:],scipy.signal.savgol_filter(force,window_length=25, polyorder=3),label="Force measurement (Smoothed)")
plt.plot(timestamps[theta_zeros[0]:],theta[theta_zeros[0]:],label=r'$\theta$')
plt.plot(timestamps[theta_zeros[0]:],theta_dot[theta_zeros[0]:],label=r'$\dot{\theta}$')
plt.axhline(0,color='b',linestyle='--')
plt.xlabel(r't(s)')
plt.ylabel(r'$\"{\theta}+\frac{mgl}{J}sin(\theta)$(rad/$s^2$)')
plt.title(r'$\frac{T}{J}$ vs time')
plt.axhline(0,color='b',linestyle='--')
plt.legend()
plt.show()

#Plot Force vs (mlg/J)sin(theta)
plt.plot(-p*np.sin(theta[theta_zeros[0]:]),force)
plt.xlabel(r'$\frac{mgl}{J}sin(\theta)$(rad/$s^2$)')
plt.ylabel(r'$\frac{T}{J}')
plt.title(r'$\frac{T}{J}$ vs $\frac{mgl}{J}sin(\theta)$')
plt.show()
