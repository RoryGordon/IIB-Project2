###-------------------------------------------------------------------------###
###---------------------------------IMPORTS---------------------------------###
###-------------------------------------------------------------------------###
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from numpy import genfromtxt
import scipy
from scipy import integrate
from scipy.optimize import least_squares
from scipy.optimize import basinhopping
from scipy.signal import find_peaks
from scipy.signal import savgol_filter
from scipy.integrate import cumtrapz
from matplotlib import animation
from statsmodels.graphics import tsaplots
###-------------------------------------------------------------------------###
###--------------------------------FUNCTIONS--------------------------------###
###-------------------------------------------------------------------------###
#Function for getting angle from gyro by re-integrating at every theta=0 and distributing the drift
def get_theta(data):
    #find theta=zeros
    a_r = data[:,1]
    filtered_a_r = savgol_filter(a_r,window_length=21, polyorder=2)
    theta_zeros,_ = find_peaks(filtered_a_r,prominence=5)
    #integrate theta and distribute drift before every zero
    timestamps = data[:,0]
    theta_dot = data[:,3]
    theta_int_once = cumtrapz(theta_dot[theta_zeros[0]:],timestamps[theta_zeros[0]:],initial=0)
    theta_fix = np.zeros(len(timestamps)) #generate an array to hold the fixed theta
    theta_fix[theta_zeros[0]:] = theta_int_once #put the hard integrated theta between the first 2 zeros
    prev_zero = theta_zeros[0] #initiate the last theta=0 as the first one for the loop
    for _ in theta_zeros[1:]: #reintegrate and correct drift
        time_section = timestamps[prev_zero:_+1] #carve out the section between the 2 zeros
        theta_dot_section = theta_dot[prev_zero:_+1]
        theta_section = cumtrapz(theta_dot_section,time_section,initial=0) #make the integration
        drift = theta_section[-1] #find the drift at the end of the section
        drift_vec = np.linspace(start=0,stop=drift,num=_-prev_zero+1) #generate a vector increasing steadily from 0 to the drift over that time frame
        theta_fix[prev_zero:_] = theta_section[:-1]-drift_vec[:-1] #make the correction so the last theta=0
        prev_zero = _ #store the zero point for the next loop
    return theta_fix #returns the drift-fixed theta

#Function for getting the gradient in the sin(theta) vs ang accel equation
def get_gradient(data):
    #find theta=zeros
    a_r = data[:,1]
    filtered_a_r = savgol_filter(a_r,window_length=21, polyorder=2)
    theta_zeros,_ = find_peaks(filtered_a_r,prominence=5)

    theta = get_theta(data) #get drift corrected theta from the data
    timestamps = data[:,0]
    theta_dot = data[:,3]
    theta_double_dot = np.gradient(theta_dot,timestamps)
    x = np.sin(theta)[theta_zeros[0]:]
    y = savgol_filter(theta_double_dot,window_length=25, polyorder=3)[theta_zeros[0]:]
    allowed_indices = np.where(abs(x)<20.2) #find indices where line is straight
    p = np.polyfit(x[allowed_indices],y[allowed_indices],deg=1) #fit a line through all of the data points within the cutoff
    return p[0]

###-------------------------------------------------------------------------###
###---------------------------DATA-&-OPTIMISATION---------------------------###
###-------------------------------------------------------------------------###
#Gather and unpack data from CSV
file = '/Users/shuowanghe/github/IIB-Project2/data/adafruitapril15th/freeswing.csv'
data = genfromtxt(file,delimiter=',')
timestamps,a_r,a_theta,theta_dot = data[:,0], data[:,1], data[:,2], data[:,3]
#Optimise correction parameters
def force_func(corrections):
    file = '/Users/shuowanghe/github/IIB-Project2/data/adafruitapril15th/freeswing.csv'
    data = genfromtxt(file,delimiter=',')
    theta_correction_factor,gradient_correction_factor,theta_offset = corrections[0],corrections[1],corrections[2]
    theta = get_theta(data)*theta_correction_factor+theta_offset
    theta_double_dot = np.gradient(data[:,3],data[:,0])*theta_correction_factor
    filtered_theta_double_dot = savgol_filter(theta_double_dot,window_length=25, polyorder=3)
    p = get_gradient(data)*gradient_correction_factor
    filtered_a_r = savgol_filter(data[:,1],window_length=21, polyorder=2)
    theta_zeros,_ = find_peaks(filtered_a_r,prominence=5)
    force = filtered_theta_double_dot[theta_zeros[0]:]-p*np.sin(theta[theta_zeros[0]:])
    return sum(abs(force))

res=least_squares(fun=force_func, x0=[1,1,0])
print(res.x,res.fun)
theta_correction_factor,gradient_correction_factor,theta_offset = res.x[0],res.x[1],res.x[2]

#Smooth the radial acceleration signal to find theta=zeros
filtered_a_r = savgol_filter(a_r,window_length=21, polyorder=2)
theta_zeros,_ = find_peaks(filtered_a_r,prominence=5)
#Differentiate gyro signal to get angular acceleration, then smooth with Sav-Gol filter
theta_double_dot = np.gradient(theta_dot,timestamps)
filtered_theta_double_dot = savgol_filter(theta_double_dot,window_length=25, polyorder=3)
#get the -mlg/J gradient from fitting the straight part of the graph
p = get_gradient(data)
#Use re-integrated, drift corrected theta from now on
theta = get_theta(data)
#Calculate some force quantity T/J
force = filtered_theta_double_dot[theta_zeros[0]:]-p*np.sin(theta[theta_zeros[0]:])
smooth_force = savgol_filter(force,window_length=25, polyorder=3)
#Calculate it post corrections
force_corrected = theta_correction_factor*filtered_theta_double_dot[theta_zeros[0]:]-gradient_correction_factor*p*np.sin(theta_correction_factor*theta[theta_zeros[0]:]+theta_offset)
smooth_force_corrected = savgol_filter(force_corrected,window_length=25, polyorder=3)

theta_corrected = theta*theta_correction_factor+theta_offset
p_corrected = p*gradient_correction_factor
theta_dot_corrected = theta_dot*theta_correction_factor

###-------------------------------------------------------------------------###
###--------------------------------ANIMATIONS-------------------------------###
###-------------------------------------------------------------------------###
#THETA_DOUBLE_DOT vs SIN(THETA) ANIMATION
fig = plt.figure()
ax = plt.axes(xlim=(-1.5, 1.5), ylim=(-10,10))
line, = ax.plot([], [], lw=1)
line_corrected, = ax.plot([], [], lw=1)
def init():
    line.set_data([], [])
    line_corrected.set_data([], [])
    return line, line_corrected,
#Animation function. This is called sequentially
def animate(i):
    y = filtered_theta_double_dot[theta_zeros[0]:i+theta_zeros[0]]
    x = np.sin(theta[theta_zeros[0]:i+theta_zeros[0]])
    line.set_data(x, y)
    y_corrected = filtered_theta_double_dot[theta_zeros[0]:i+theta_zeros[0]]*theta_correction_factor
    x_corrected = np.sin(theta[theta_zeros[0]:i+theta_zeros[0]]*theta_correction_factor+theta_offset)
    line_corrected.set_data(x_corrected, y_corrected)
    return line, line_corrected,
#Call the animator. blit=True means only re-draw the parts that have changed.
anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=len(theta), interval=20, blit=True)
#Save the animation
#anim.save('2sidepushfix.gif', fps=50, extra_args=['-vcodec', 'libx264'])
#Plot the animation
x = np.sin(theta_corrected)[theta_zeros[0]:]
plt.plot(x,p_corrected*x,'r') #plot the fitted line, through the origin
plt.xlabel(r'sin($\theta$)')
plt.ylabel(r'$\"{\theta}$(rad/$s^2$)')
plt.title(r'$\"{\theta}$ vs sin($\theta$)')
plt.show()

###-------------------------------------------------------------------------###
###----------------------------------PLOTS----------------------------------###
###-------------------------------------------------------------------------###
#Plot line fit through theta double dot vs sin(theta) relationship
allowed_indices = np.where(abs(x)<20.2) #find indices where line is straight
x = np.sin(theta[theta_zeros[0]:])
y = filtered_theta_double_dot[theta_zeros[0]:]
x_corrected=np.sin(theta_corrected[theta_zeros[0]:])
y_corrected=filtered_theta_double_dot[theta_zeros[0]:]*theta_correction_factor
plt.plot(x,y,'.',label='Raw measurements') #plot all the data points
plt.plot(x_corrected,y_corrected,'.',label='Corrected data') #plot all the data points
plt.plot(x_corrected[allowed_indices],y_corrected[allowed_indices],'.',label='Points used in fitting') #plot the allowed data points for fitting
plt.plot(x,p*x,label='Line fit through origin of central points') #plot the fitted line, through the origin
plt.plot(x_corrected,p_corrected*x_corrected,label='Line fit through origin of corrected points')
plt.xlabel(r'sin($\theta$)')
plt.ylabel(r'$\"{\theta}$(rad/$s^2$)')
plt.title(r'$\"{\theta}$ vs sin($\theta$) with a line fitted through')
plt.legend()
plt.show()

#Force, theta and theta_dot vs time
plt.plot(timestamps[theta_zeros[0]:],smooth_force,label="Force measurement (Smoothed)")
plt.plot(timestamps[theta_zeros[0]:],smooth_force_corrected,label="Corrected force measurement (Smoothed)")
plt.plot(timestamps[theta_zeros[0]:],theta_corrected[theta_zeros[0]:],label=r'$\theta$')
plt.plot(timestamps[theta_zeros[0]:],theta_dot_corrected[theta_zeros[0]:],label=r'$\dot{\theta}$')
plt.axhline(0,color='b',linestyle='--')
plt.xlabel(r't(s)')
plt.ylabel(r'$\"{\theta}+\frac{mgl}{J}sin(\theta)$(rad/$s^2$)')
plt.title(r'$\frac{T}{J}$ vs time')
plt.legend()
plt.show()

#Plot Force vs (mlg/J)sin(theta)
plt.plot(-p*np.sin(theta[theta_zeros[0]:]),smooth_force,label="As measured")
plt.plot(-p_corrected*np.sin(theta_corrected[theta_zeros[0]:]),smooth_force_corrected,label="Corrected")
plt.xlabel(r'$\frac{mgl}{J}sin(\theta)$(rad/$s^2$)')
plt.ylabel(r'$\frac{T}{J}$')
plt.title(r'$\frac{T}{J}$ vs $\frac{mgl}{J}sin(\theta)$')
plt.legend()
plt.show()

#Plot Force vs velocity
plt.plot(theta_dot[theta_zeros[0]:],smooth_force,label="As measured")
plt.plot(theta_dot_corrected[theta_zeros[0]:],smooth_force_corrected,label="Corrected")
plt.xlabel(r'$\dot{\theta}$')
plt.ylabel(r'$\frac{T}{J}$')
plt.title(r'Force vs $\dot{\theta}$')
plt.legend()
plt.show()
